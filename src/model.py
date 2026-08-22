"""Stage 4/5: LightGBM, called directly, on the SQL feature layer plus raw
columns. Train on train; calibration and operating point from validation;
test reported once per configuration.

The six modeling parameters and their defenses:
    num_leaves=63         moderate tree capacity; 2^6-1, default family
    learning_rate=0.05    conservative; paired with a fixed 400 trees
    n_estimators=400      fixed budget, no early stopping (nothing tunes
                          itself against val behind our back)
    min_child_samples=100 no leaf may fit a cell thinner than ~100 rows --
                          the rules stage showed what 45-row cells do
    colsample_bytree=0.8  mild decorrelation across 400+ columns
    scale_pos_weight      n_neg/n_pos on train (mandated; no SMOTE)

Probability handling, in order:
  1. scale_pos_weight shifts the implied prior; the analytic inversion
     p = q / (q + w*(1-q)) undoes the class-weight exactly IF that were the
     only distortion;
  2. measured on validation, it is not (the model is also miscalibrated in
     shape), so isotonic regression is fit on validation and applied
     everywhere. The EV rule reads p's absolute value -- an uncalibrated p
     mis-sets every implied amount bar -- so calibration is a correctness
     requirement, not polish.

The ablation retrains identically (same params, same seed) without the six
engineered SQL features; the test-cost delta is the measured value of the
Postgres feature layer.
"""

import os

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sqlalchemy import create_engine

import evaluate as ev

DB_URL = os.environ.get("DB_URL", "postgresql://fraud:fraud@localhost:5432/fraud")

PARAMS = {
    "num_leaves": 63,
    "learning_rate": 0.05,
    "n_estimators": 400,
    "min_child_samples": 100,
    "colsample_bytree": 0.8,
}

# Not features, with reasons:
#   transactionid  row identifier
#   isfraud        the target; nothing derives from it
#   transactiondt  absolute clock position -- would encode the fold/time
#                  trend itself, which cannot generalize past the data;
#                  calendar features would need the unverified start date
#   card_key(_v2)  entity identity; the model learns behaviour, not entities
#   fold           the split
EXCLUDE = {"transactionid", "isfraud", "transactiondt",
           "card_key", "card_key_v2", "fold"}

SQL_FEATURES = ["cnt_1h", "cnt_24h", "cnt_7d", "secs_since_prev",
                "amt_ratio_med", "has_identity"]

CAPACITY = [100, 500, 1000]
FRICTIONS = [10.0, 25.0, 50.0]
BASELINE_BAR = 445_281   # rules + EV decisioning, test (src/baseline.py)


def load():
    eng = create_engine(DB_URL)
    df = pd.read_sql("SELECT * FROM txn", eng)
    feat = pd.read_sql("SELECT * FROM features", eng)
    ident = pd.read_sql("SELECT * FROM txn_identity", eng)
    zero_head = pd.read_sql("""
        SELECT card_key_v2 FROM txn
        GROUP BY card_key_v2 HAVING count(*) >= 100 AND sum(isfraud) = 0
    """, eng)  # label-derived, reporting only -- never a feature
    eng.dispose()
    df = df.merge(feat, on="transactionid", how="left")
    df = df.merge(ident, on="transactionid", how="left")
    # Physical row order from a heap scan is not stable across rebuilds, and
    # LightGBM is only deterministic for identical row order -- pin it.
    df = df.sort_values("transactionid").reset_index(drop=True)
    df["transactionamt"] = df.transactionamt.astype(float)
    df["in_zero_head"] = df.card_key_v2.isin(zero_head.card_key_v2).astype(int)
    return df


def train_predict(X, y, masks, w, seed=42):
    clf = lgb.LGBMClassifier(objective="binary", scale_pos_weight=w,
                             random_state=seed, n_jobs=-1, verbose=-1, **PARAMS)
    clf.fit(X[masks["train"]], y[masks["train"]])
    q = clf.predict_proba(X)[:, 1]
    return clf, q / (q + w * (1 - q))   # undo the class-weight prior shift


def calibrated_test_ev_cost(X, y, masks, w, test, seed):
    """Train -> isotonic on val -> EV rule on test. Used by the ablation."""
    _, p_raw = train_predict(X, y, masks, w, seed)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_raw[masks["val"]], y[masks["val"]])
    p = iso.predict(p_raw)[masks["test"]]
    alert = ev.ev_alert(p, test.transactionamt)
    m = ev.metrics(test.isfraud, alert, test.transactionamt)
    return m["cost"], m["alerts"]


def decile_table(p, y, label):
    b = pd.qcut(p, 10, labels=False, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": y}).groupby(b)
    print(f"  {label}: decile mean-predicted vs actual (actual/pred)")
    for i, (mp, my) in enumerate(zip(g.p.mean(), g.y.mean())):
        ratio = my / mp if mp > 0 else float("nan")
        print(f"    d{i}: pred {mp:.4f}  actual {my:.4f}  (x{ratio:4.1f})")


def report(label, fold, alert, friction=ev.FRICTION):
    base = ev.no_alert_cost(fold.isfraud, fold.transactionamt)
    m = ev.metrics(fold.isfraud, alert, fold.transactionamt, friction)
    heads = int((np.asarray(alert) & (fold.in_zero_head == 1)).sum())
    print(f"  {label:>22}: cost ${m['cost']:>9,.0f} "
          f"({100*(base-m['cost'])/base:4.1f}% vs no-alert)  "
          f"alerts {m['alerts']:>6,}  precision {m['precision']:.3f}  "
          f"recall {m['recall']:.3f}  $recall {m['dollar_recall']:.3f}  "
          f"zero-head FPs {heads:,}")
    return m["cost"]


def main():
    df = load()
    X = df[[c for c in df.columns if c not in EXCLUDE | {"in_zero_head"}]].copy()
    for c in X.columns[X.dtypes == object]:
        X[c] = X[c].astype("category")
    y = df.isfraud.astype(int).to_numpy()
    masks = {k: (df.fold == k).to_numpy() for k in ("train", "val", "test")}
    val_m, test_m = masks["val"], masks["test"]

    n_cat = int((X.dtypes == "category").sum())
    print(f"features: {X.shape[1]} columns ({n_cat} categorical), "
          f"excluded: {sorted(EXCLUDE)}")
    w = (y[masks["train"]] == 0).sum() / (y[masks["train"]] == 1).sum()
    print(f"scale_pos_weight (train n_neg/n_pos): {w:.2f}")

    clf, p_raw = train_predict(X, y, masks, w)

    print("\ntop 10 features by gain:")
    imp = pd.Series(clf.booster_.feature_importance("gain"),
                    index=X.columns).nlargest(10)
    for name, g in imp.items():
        print(f"  {name:>16}: {g:,.0f}")

    print("\n== calibration (validation) ==")
    decile_table(p_raw[val_m], y[val_m], "before (analytic inversion only)")
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_raw[val_m], y[val_m])
    p_cal = iso.predict(p_raw)
    decile_table(p_cal[val_m], y[val_m], "after isotonic (fit on val)")

    df["p_raw"], df["p_cal"] = p_raw, p_cal
    val, test = df[val_m], df[test_m]

    # operating points: swept on validation only, one per calibration state
    t_raw, _ = ev.best_p_threshold(val.isfraud, val.p_raw, val.transactionamt)
    thr, costs = ev.sweep_cost_curve(val.isfraud, val.p_cal, val.transactionamt)
    i = int(np.argmin(costs))
    t_cal = float(thr[i])
    print(f"\ncost-optimal p threshold (val sweep): "
          f"before {t_raw:.4f}, after {t_cal:.4f}")

    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thr[1:], costs[1:], color="#2563eb", lw=2)
    base_val = ev.no_alert_cost(val.isfraud, val.transactionamt)
    ax.axhline(base_val, color="#9ca3af", lw=1.2, ls="--")
    ax.annotate("no-alert cost", xy=(0.7, base_val), xytext=(0, 5),
                textcoords="offset points", color="#6b7280", fontsize=9)
    ax.plot([t_cal], [costs[i]], "o", color="#2563eb", ms=8)
    ax.annotate(f"minimum: t={t_cal:.3f}, ${costs[i]:,.0f}",
                xy=(t_cal, costs[i]), xytext=(12, 14),
                textcoords="offset points", color="#374151", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("alert threshold on calibrated p", color="#374151")
    ax.set_ylabel("expected cost, validation fold ($)", color="#374151")
    ax.set_title("Expected cost vs. alert threshold (validation, calibrated)",
                 color="#111827", fontsize=11)
    ax.grid(True, color="#e5e7eb", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig("results/cost_curve.png", dpi=150)
    print("wrote results/cost_curve.png")

    print(f"\n[test]  no-alert cost "
          f"${ev.no_alert_cost(test.isfraud, test.transactionamt):,.0f}")
    report("before: p-threshold", test, (test.p_raw >= t_raw).to_numpy())
    report("before: expected-value", test,
           ev.ev_alert(test.p_raw, test.transactionamt))
    report("after:  p-threshold", test, (test.p_cal >= t_cal).to_numpy())
    c_ev = report("after:  expected-value", test,
                  ev.ev_alert(test.p_cal, test.transactionamt))
    print(f"  vs rules baseline (EV, ${BASELINE_BAR:,}): "
          f"{'BEATS' if c_ev < BASELINE_BAR else 'DOES NOT BEAT'} it "
          f"by ${abs(BASELINE_BAR - c_ev):,.0f}")

    heads = df[df.in_zero_head == 1]
    print(f"\nzero-fraud rail evidence: mean calibrated p on its "
          f"{len(heads):,} rows = {heads.p_cal.mean():.4f} vs population "
          f"mean {df.p_cal.mean():.4f} (fraud base rate {y.mean():.4f})")

    print("\n== capacity-constrained review (fraud dollars in top-N/day, "
          "rank by p*amount, calibrated) ==")
    for name, fold in (("val", val), ("test", test)):
        day = fold.transactiondt // 86400
        rank = fold.p_cal * fold.transactionamt
        parts = [f"{n}/day: {100*ev.capacity_dollar_recall(day, rank, fold.transactionamt, fold.isfraud, n):.1f}%"
                 for n in CAPACITY]
        print(f"  {name} -> " + "  ".join(parts))

    print("\n== friction sensitivity (test; threshold refit on val, "
          "calibrated) ==")
    for f in FRICTIONS:
        t_f, _ = ev.best_p_threshold(val.isfraud, val.p_cal,
                                     val.transactionamt, f)
        m_thr = ev.metrics(test.isfraud, test.p_cal >= t_f,
                           test.transactionamt, f)
        m_ev = ev.metrics(test.isfraud,
                          ev.ev_alert(test.p_cal, test.transactionamt, f),
                          test.transactionamt, f)
        print(f"  friction ${f:>2.0f}: p* {t_f:.4f}  "
              f"p-thr cost ${m_thr['cost']:,.0f} ({m_thr['alerts']:,} alerts)  "
              f"EV cost ${m_ev['cost']:,.0f} ({m_ev['alerts']:,} alerts)")

    # Ablation at two seeds: the with/without delta turned out to be the
    # same magnitude as seed-to-seed training noise (its sign flipped under
    # a row-order perturbation during development), so the noise scale is
    # measured and printed alongside the delta rather than asserted.
    print("\n== ablation: six SQL features removed, identical training, "
          "two seeds ==")
    X_abl = X.drop(columns=SQL_FEATURES)
    for seed in (42, 43):
        if seed == 42:
            c_full, n_full = c_ev, None
        else:
            c_full, n_full = calibrated_test_ev_cost(X, y, masks, w, test, seed)
        c_abl, n_abl = calibrated_test_ev_cost(X_abl, y, masks, w, test, seed)
        print(f"  seed {seed}: with ${c_full:,.0f}  without ${c_abl:,.0f}  "
              f"SQL layer worth ${c_abl - c_full:,.0f}")


if __name__ == "__main__":
    main()
