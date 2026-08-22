# Card fraud detection on IEEE-CIS, cost-first

Transaction fraud detection built as a cost-minimization problem, not a
classification benchmark: SQL feature engineering over a reconstructed card
identity, an explainable rules baseline, a LightGBM model, and an explicit
dollar cost matrix that both are judged on. Every number below is printed by
`make run`.

## Result

On the test fold (final ~15% of the timeline, 88,581 transactions, never
touched during training or threshold selection):

| policy | test cost | vs doing nothing |
|---|---|---|
| no alerts (eat every fraud) | $546,684 | — |
| rules baseline + expected-value decisioning | $445,281 | −18.5% |
| LightGBM + expected-value decisioning, calibrated | **$283,703** | **−48.1%** |

The model beats the rules baseline by **$161,578 (36% lower cost)**. At its
operating point it raises 4,966 alerts over the ~31-day test window
(precision 0.317, recall 0.511, **67.8% of fraud dollars**). Under a review
capacity of 100 alerts/day, ranking by expected dollars (p × amount) puts
59.1% of test-fold fraud dollars in front of a reviewer; 500/day reaches
88.1%.

Cost matrix (per transaction): true negative $0; true positive $2 (review);
false positive $27 (review + $25 declined-customer friction); false negative
amount + $25 (chargeback + dispute fee). The friction figure is the softest
assumption and the results pivot on it — see limitations.

## The decision rule is an amount threshold in disguise

Sweeping a probability threshold is the textbook move, but with this cost
matrix the optimal rule is `alert when p × (amount + 25) > 2 + (1 − p) × 25`,
which rearranges to **`amount > 27/p − 50`** — a dollar bar per probability
level, not a probability cutoff. For the rules baseline, whose score has
four levels, the entire policy collapses to four fixed bars
(`make baseline`):

| rules fired | train fraud rate | alert when amount exceeds |
|---|---|---|
| 0 | 0.034 | $748 |
| 1 | 0.065 | $365 |
| 2 | 0.133 | $152 |
| 3+ | 0.133 (pooled) | $152 |

The rules don't decide anything by themselves — they select which bar
applies, and the amount does the discriminating. On the test fold this
expected-value form beats the swept-threshold form of the *same* rules by
$52,895, and the gap exists because fraud amounts here are not concentrated
in big tickets (fraud p99 is $994, *below* the legitimate p99 of $1,104), so
no probability cutoff can reach the low-probability/high-amount rows that
dominate the loss. The score→probability map is monotone by construction:
thin cells are collapsed and pool-adjacent-violators is applied, after the
first fit produced a 4-row cell with p = 0 (a permanent never-alert) and a
45-row cell that gave a higher score a higher dollar bar.

## The entity key: measured, not assumed

The dataset has no card ID. The first reconstruction —
`card1..card6 | addr1 | P_emaildomain` — produced 94,873 keys whose
distribution is **bimodal** (`make profile`): median 2 transactions per key,
but the top key held 4,002 transactions across all 182 days at a
near-base-rate fraud rate — a pooled cohort, not a card. The largest 100
keys held 13.3% of all transactions.

Root cause, quantified: addr1 is 11.1% null, and null-collapse pools
distinct cards under one key. Keys with null addr1 carry 22.0% of their
transactions in 1,000+-transaction mega-keys versus 3.1% for keys with an
address — plus residual pooling from common BIN combinations.

The fix appends `day − D1` to the key. D1's meaning is deliberately not
assumed (the D columns are anonymized); instead one property was tested:
within a key, `day − D1` is either **exactly constant** (12,756 of 27,232
small keys) **or spread over months** (11,226 keys) — a range of exactly 1
day, the signature of rounding jitter splitting real cards, occurs 173
times. That bimodal signature is what a stable per-entity constant mixed
across pooled entities looks like, and constancy plus discriminative power
is all a key component needs. The refined key (278,270 entities) splits the
4,002-transaction monster into 362 pieces with card-sized medians, and cuts
the top-100-key share from 13.3% to 1.5% of transactions.

The honest cost: transactions with no prior history on their key — where
velocity features are undefined — rise from 16.1% (v1) to **47.1%** (v2).
Computing velocity over entities proven to be pools was judged worse than
having no velocity at all for more rows.

## A structurally zero-fraud rail exists, and it broke the rules

Eleven keys with ≥100 transactions carry 3,107 transactions and **zero
fraud** — the base rate predicts ~109. They are automated traffic, not
shoppers: one key runs 1,414 transactions in 52 days at a median gap of 12
*seconds* (ProductCD S, 100% identity coverage); the others are
recurring-billing-shaped (fixed amounts on ~daily cadence, ProductCD W).
Seven survive into the test window.

The rules baseline cannot tell this rail from a fraud burst: its
probability-threshold form flags 1,354 rail rows on test — $36,558 of
guaranteed false positives, 31% of its alert budget. The model produces
**zero** rail false positives at every operating point, and not merely
because the amount bar excludes them: its mean calibrated probability on the
rail's 3,107 rows is 0.0024, against a population mean of 0.0403. It learned
the behaviour is safe.

## What the SQL feature layer bought the model: nothing, within noise

Retraining identically with the six engineered features removed changes the
test EV cost by **−$1,653 at seed 42 and −$750 at seed 43** (negative =
removing them helped), while the full model's own seed-to-seed spread is
$4,891. The measured contribution of the entire Postgres velocity layer to
model cost is zero within training noise — an earlier single-seed run
showed +$8,306 before the noise scale was measured, which is exactly why
the ablation now runs at two seeds. The honest reading: C1–C14 are the
vendor's own counting features and D1–D15 its timedeltas, computed with
access to the *true* card identity — our velocity features reconstruct
approximately what the vendor already computed exactly, and `c14` and `d2`
sit in the model's top-10 gains. The Postgres layer's demonstrated value is
operational, not predictive: an explainable rules tier, dollar bars, the
split/leakage audit trail, and the entity-key analysis itself.

## Limitations

- **Calibration.** Raw model probabilities were badly miscalibrated even
  after analytically inverting the `scale_pos_weight` prior shift
  (validation deciles ran 1.8×–12.1× too low — shape-dependent, worst on
  easy negatives, i.e. genuine boosting overconfidence rather than an
  inversion bug). Isotonic regression fit on validation fixed the deciles
  and was worth $22,505 on test — the expected-value rule reads absolute
  probabilities, so calibration is a correctness requirement. The swept
  p-threshold, invariant to monotone recalibration, moved by $392.
- **Friction sensitivity.** Moving the declined-customer cost across
  $10/$25/$50 moves the model's test EV cost across
  $211,367 / $283,703 / $339,911 and roughly halves-to-doubles alert
  volume. The single softest business assumption owns a ~$129k swing.
- **Temporal decay.** The operating point and calibration both come from
  the validation month and are applied one month later; the capacity view
  shows the drift directly (fraud-dollar capture at 100 alerts/day:
  65.2% on validation, 59.1% on test).
- **Training noise.** LightGBM results move ~$5k between seeds and were
  order-sensitive until row order was pinned (`ORDER BY transactionid`);
  cost deltas smaller than that spread — the ablation above — are noise,
  and are labeled as such rather than claimed.
- **The top features are anonymized.** `v258`, `v294`, `c14` lead the gain
  ranking; nobody can say what they measure. The model is an empirical
  artifact of this dataset, not a portable fraud theory.
- **Residual entity noise.** The refined key still has a 1,414-transaction
  head entity; 0.21% of rows lack D1 and fall back to coarse identity;
  same-second peers make trailing windows fuzzy for 0.02% of rows.
- **`day − D1` constancy is demonstrated; its meaning is not.** No claim
  about what D1 is survives in this repo, only the measured property.

## Reproduce

Requires Docker and the two Kaggle files
([IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data))
placed in `data/raw/`: `train_transaction.csv`, `train_identity.csv`.

```
docker compose up -d
make setup     # venv + deps + Postgres 16 healthy
make run       # load -> migrate -> profile -> baseline -> model
```

`make run` prints every number in this README: dataset profile and entity
key validation (`profile`), rules baseline with fitted thresholds and
dollar bars (`baseline`), model with calibration, ablation, capacity view,
friction sensitivity, and the cost curve at `results/cost_curve.png`
(`model`). The full-teardown check (`docker compose down -v`, rebuild)
reproduced the profile byte-for-byte.

```
data/raw/     gitignored Kaggle CSVs
sql/          numbered migrations: typed cast, entity keys, features, split
src/          load.py, migrate.py, profile.py, baseline.py, model.py, evaluate.py
results/      cost curve plot
```

Stack: Python 3.11, Postgres 16 (Docker), psycopg2/SQLAlchemy, pandas,
LightGBM, scikit-learn, matplotlib.
