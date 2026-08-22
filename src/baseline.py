"""Stage 3: rules baseline. No ML -- explainable threshold rules, fitted on
train only, scored on val and test with the shared cost model.

Each rule is a single comparison. Its threshold is chosen from a small grid
by minimizing standalone train cost (alert = this rule alone fires). The
rule score is the number of fired rules; the train-set fraud rate at each
score level converts score to a probability, which feeds both decision
rules: the swept probability threshold and the expected-value rule.
"""

import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import evaluate as ev

DB_URL = os.environ.get("DB_URL", "postgresql://fraud:fraud@localhost:5432/fraud")

# (column, direction, candidate thresholds). NaN never fires a rule.
# has_identity was a rule here and was pruned: standalone train cost $4.31M
# against a $2.53M train no-alert cost -- a rule that loses money on its own
# does not belong in the score at equal weight. It remains a model feature.
RULE_GRIDS = {
    "high_amount": ("amount", ">=", [100, 200, 300, 500, 750, 1000, 1500, 2000]),
    "burst_1h": ("cnt_1h", ">=", [2, 3, 4, 6]),
    "burst_24h": ("cnt_24h", ">=", [3, 5, 8, 12]),
    "rapid_repeat": ("secs_since_prev", "<=", [30, 60, 300]),
    "amt_spike": ("amt_ratio_med", ">=", [2, 3, 5, 10]),
}

# Two guards on the score -> p map, both fixing defects measured on the
# first fit (a p of 0.0 from a four-row cell producing a permanent
# never-alert; a 45-row cell giving score 4 a HIGHER dollar bar than
# score 3):
#   * scores above SCORE_CAP collapse into one cell (kills the tiny tail);
#   * pool-adjacent-violators then merges any remaining adjacent cells
#     whose p decreases as score increases -- discrete isotonic regression,
#     done transparently on a handful of cells. More rules fired can never
#     mean a higher dollar bar.
SCORE_CAP = 3

CAPACITY = [100, 500, 1000]
FRICTIONS = [10.0, 25.0, 50.0]


def load():
    eng = create_engine(DB_URL)
    df = pd.read_sql("""
        SELECT t.transactionid, t.isfraud, t.transactiondt, t.fold,
               t.transactionamt::double precision AS amount,
               f.cnt_1h, f.cnt_24h, f.cnt_7d, f.secs_since_prev,
               f.amt_ratio_med, f.has_identity,
               -- label-derived, for REPORTING ONLY (zero-fraud head audit);
               -- never used as a model/rule input
               (t.card_key_v2 IN (
                   SELECT card_key_v2 FROM txn
                   GROUP BY card_key_v2
                   HAVING count(*) >= 100 AND sum(isfraud) = 0
               ))::int AS in_zero_head
        FROM txn t JOIN features f USING (transactionid)
        ORDER BY t.transactionid
    """, eng)
    eng.dispose()
    return df


def fire(df, col, op, x):
    v = df[col]
    return (v >= x) if op == ">=" else (v <= x)  # NaN compares False


def fit_rules(train):
    fitted = {}
    for name, (col, op, grid) in RULE_GRIDS.items():
        costs = {x: ev.total_cost(train.isfraud, fire(train, col, op, x),
                                  train.amount) for x in grid}
        x_best = min(costs, key=costs.get)
        fitted[name] = (col, op, x_best, costs[x_best])
    return fitted


def score(df, fitted):
    s = np.zeros(len(df), int)
    for col, op, x, _ in fitted.values():
        s += fire(df, col, op, x).to_numpy()
    return np.minimum(s, SCORE_CAP)


def train_p_map(train, fitted):
    s = score(train, fitted)
    grp = pd.Series(train.isfraud.to_numpy()).groupby(s)
    raw = grp.mean()
    counts = grp.size()
    # pool adjacent violators: merge neighbouring cells while p decreases
    blocks = [[[k], int(counts[k]), float(raw[k])] for k in raw.index]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][2] > blocks[i + 1][2]:
            ks, n1, p1 = blocks[i]
            ks2, n2, p2 = blocks[i + 1]
            blocks[i] = [ks + ks2, n1 + n2, (n1 * p1 + n2 * p2) / (n1 + n2)]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    pooled = pd.Series({k: p for ks, _, p in blocks for k in ks})
    return pooled, raw, s


def ensemble_ev_train_cost(train, fitted):
    p_map, _, s = train_p_map(train, fitted)
    p = pd.Series(s).map(p_map).to_numpy()
    return ev.total_cost(train.isfraud, ev.ev_alert(p, train.amount),
                         train.amount)


def report_fold(name, fold, p, alert_thr, alert_ev, friction=ev.FRICTION):
    base = ev.no_alert_cost(fold.isfraud, fold.amount)
    print(f"\n[{name}]  no-alert cost ${base:,.0f}")
    for label, alert in (("p-threshold", alert_thr), ("expected-value", alert_ev)):
        m = ev.metrics(fold.isfraud, alert, fold.amount, friction)
        head = fold.loc[np.asarray(alert) & (fold.in_zero_head == 1)]
        print(f"  {label:>14}: cost ${m['cost']:>9,.0f} "
              f"({100*(base-m['cost'])/base:4.1f}% vs no-alert)  "
              f"alerts {m['alerts']:>6,}  precision {m['precision']:.3f}  "
              f"recall {m['recall']:.3f}  $recall {m['dollar_recall']:.3f}  "
              f"zero-head FPs {len(head):,} (${(ev.REVIEW+friction)*len(head):,.0f})")


def main():
    df = load()
    folds = {k: df[df.fold == k] for k in ("train", "val", "test")}
    train = folds["train"]

    base_train = ev.no_alert_cost(train.isfraud, train.amount)
    print(f"train no-alert cost: ${base_train:,.0f}")
    print("\n== fitted rules (train only) ==")
    fitted = fit_rules(train)
    for name, (col, op, x, c) in fitted.items():
        flag = "  [worse than no-alert]" if c > base_train else ""
        print(f"  {name:>12}: {col} {op} {x}   standalone train cost "
              f"${c:,.0f}{flag}")

    # burst_24h sits within noise of no-alert standalone; keep it only if it
    # helps the ensemble on train.
    without = {k: v for k, v in fitted.items() if k != "burst_24h"}
    c_with = ensemble_ev_train_cost(train, fitted)
    c_without = ensemble_ev_train_cost(train, without)
    print(f"\nensemble EV train cost with burst_24h ${c_with:,.0f}, "
          f"without ${c_without:,.0f}", end="")
    if c_without <= c_with:
        fitted = without
        print("  -> dropped")
    else:
        print("  -> kept")

    p_map, p_raw, s_train = train_p_map(train, fitted)
    print(f"\n== score -> p (train), implied EV amount bar "
          f"A > {ev.REVIEW + ev.FRICTION:.0f}/p - {ev.FN_FIXED + ev.FRICTION:.0f} ==")
    for k, v in p_map.items():
        cap = "+" if k == SCORE_CAP else " "
        pooled = "" if abs(v - p_raw[k]) < 1e-12 else f" (raw {p_raw[k]:.4f}, pooled)"
        bar = (f"${(ev.REVIEW + ev.FRICTION) / v - (ev.FN_FIXED + ev.FRICTION):,.0f}"
               if v > 0 else "never alerts")
        print(f"  score {k}{cap}: n {(s_train == k).sum():>7,}  p {v:.4f}{pooled}  "
              f"alerts when amount > {bar}")

    def p_of(fold):
        s = pd.Series(score(fold, fitted))
        return s.map(p_map).fillna(p_map.max()).to_numpy()

    t_star, _ = ev.best_p_threshold(train.isfraud, p_of(train), train.amount)
    print(f"\ncost-optimal p threshold (train): p >= {t_star:.4f}")

    for name in ("val", "test"):
        fold = folds[name]
        p = p_of(fold)
        report_fold(name, fold, p, p >= t_star, ev.ev_alert(p, fold.amount))

    print("\n== capacity-constrained review (fraud dollars in top-N/day) ==")
    for name in ("val", "test"):
        fold = folds[name]
        p = p_of(fold)
        day = fold.transactiondt // 86400
        for label, rank in (("rank by p       ", p),
                            ("rank by p*amount", p * fold.amount.to_numpy())):
            parts = [f"{n}/day: {100*ev.capacity_dollar_recall(day, rank, fold.amount, fold.isfraud, n):.1f}%"
                     for n in CAPACITY]
            print(f"  {name} {label} -> " + "  ".join(parts))

    print("\n== friction sensitivity (test fold; rule defs fixed, "
          "operating points refit on train) ==")
    test = folds["test"]
    p_test, p_train = p_of(test), p_of(folds["train"])
    for f in FRICTIONS:
        t_f, _ = ev.best_p_threshold(train.isfraud, p_train, train.amount, f)
        m_thr = ev.metrics(test.isfraud, p_test >= t_f, test.amount, f)
        m_ev = ev.metrics(test.isfraud, ev.ev_alert(p_test, test.amount, f),
                          test.amount, f)
        print(f"  friction ${f:>2.0f}: p* {t_f:.4f}  "
              f"p-thr cost ${m_thr['cost']:,.0f} ({m_thr['alerts']:,} alerts)  "
              f"EV cost ${m_ev['cost']:,.0f} ({m_ev['alerts']:,} alerts)")


if __name__ == "__main__":
    main()
