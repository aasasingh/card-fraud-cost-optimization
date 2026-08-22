"""Cost model and evaluation helpers, shared by the rules baseline and the
model. All dollar figures are per transaction:

    true negative   $0
    true positive   $2                  (review cost -- analyst paid either way)
    false positive  $2 + friction      (review + declined-customer friction)
    false negative  amount + $25       (chargeback + fixed dispute cost)

friction defaults to $25 and is the least certain assumption; callers
sensitivity-test it.
"""

import numpy as np

REVIEW = 2.0
FN_FIXED = 25.0
FRICTION = 25.0


def total_cost(y, alert, amount, friction=FRICTION):
    y = np.asarray(y, bool)
    alert = np.asarray(alert, bool)
    amount = np.asarray(amount, float)
    tp = (y & alert).sum()
    fp = (~y & alert).sum()
    fn_amount = amount[y & ~alert].sum()
    fn = (y & ~alert).sum()
    return REVIEW * tp + (REVIEW + friction) * fp + fn_amount + FN_FIXED * fn


def metrics(y, alert, amount, friction=FRICTION):
    y = np.asarray(y, bool)
    alert = np.asarray(alert, bool)
    amount = np.asarray(amount, float)
    n_alert = int(alert.sum())
    tp = int((y & alert).sum())
    fraud_dollars = amount[y].sum()
    return {
        "cost": total_cost(y, alert, amount, friction),
        "alerts": n_alert,
        "precision": tp / n_alert if n_alert else float("nan"),
        "recall": tp / y.sum() if y.sum() else float("nan"),
        "dollar_recall": amount[y & alert].sum() / fraud_dollars
                         if fraud_dollars else float("nan"),
    }


def no_alert_cost(y, amount):
    """Reference point: cost of alerting on nothing (eat every fraud)."""
    y = np.asarray(y, bool)
    return np.asarray(amount, float)[y].sum() + FN_FIXED * y.sum()


def sweep_cost_curve(y, p, amount, friction=FRICTION):
    """Cost of alert = (p >= t) at every distinct value of p, in one pass:
    sort descending by p, take cumulative TP/FP/fraud-dollar sums, and read
    the cost off at each unique-p group boundary (ties must alert together).
    Returns (thresholds, costs), including t = +inf (alert on nothing)."""
    y = np.asarray(y, bool)
    p = np.asarray(p, float)
    amount = np.asarray(amount, float)
    order = np.argsort(-p, kind="stable")
    ps, ys, amts = p[order], y[order], amount[order]
    cum_tp = np.cumsum(ys)
    cum_fp = np.cumsum(~ys)
    cum_famt = np.cumsum(np.where(ys, amts, 0.0))
    total_famt = float(cum_famt[-1]) if len(ps) else 0.0
    n_fraud = int(ys.sum())
    ends = np.append(np.nonzero(np.diff(ps))[0], len(ps) - 1)
    costs = (REVIEW * cum_tp[ends] + (REVIEW + friction) * cum_fp[ends]
             + (total_famt - cum_famt[ends])
             + FN_FIXED * (n_fraud - cum_tp[ends]))
    thresholds = np.append(np.inf, ps[ends])
    costs = np.append(total_famt + FN_FIXED * n_fraud, costs)
    return thresholds, costs


def best_p_threshold(y, p, amount, friction=FRICTION):
    thresholds, costs = sweep_cost_curve(y, p, amount, friction)
    i = int(np.argmin(costs))
    return float(thresholds[i]), float(costs[i])


def ev_alert(p, amount, friction=FRICTION):
    """Expected-value decision rule: alert when the expected cost of not
    reviewing exceeds the expected cost of reviewing,
        p * (amount + FN_FIXED) > REVIEW + (1 - p) * friction.
    A threshold on expected dollars, not on probability."""
    p = np.asarray(p, float)
    amount = np.asarray(amount, float)
    return p * (amount + FN_FIXED) > REVIEW + (1 - p) * friction


def capacity_dollar_recall(day, rank, amount, y, n_per_day):
    """If reviewers handle only n alerts/day, review the day's top-n by
    `rank` (ties broken by amount): fraction of fraud dollars reviewed.
    Pass rank=p for probability ranking, rank=p*amount for expected-dollar
    ranking."""
    import pandas as pd
    df = pd.DataFrame({"day": day, "rank": rank, "amount": amount, "y": y})
    df = df.sort_values(["day", "rank", "amount"], ascending=[True, False, False])
    picked = df.groupby("day").head(n_per_day)
    fraud_dollars = df.loc[df.y == 1, "amount"].sum()
    return picked.loc[picked.y == 1, "amount"].sum() / fraud_dollars
