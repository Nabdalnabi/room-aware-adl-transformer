# Room-Aware ADL Forecasting

Compact PyTorch Transformer and LSTM pipelines for next-observation activities
of daily living (ADL) classification from ambient smart-home sensors.
Prepared from Nader Abdalnabi's dissertation code for research-code review.

## What this demonstrates

- Causal room-specific motion counts, door transitions, environmental summaries,
  change rates, and cyclic time-of-day features.
- Entire chronological days are held out for validation and testing **before**
  feature scaling or sequence creation. Scaling is fit on training rows only.
- Windows cannot cross houses, calendar days, or recording gaps above the configured limit.
- Only sensor columns and engineered time features enter the model: neither the
  other ADL label nor house identifiers are predictors.
- Validation-loss early stopping; a single held-out evaluation after training.
- Accuracy, micro/macro F1, one-vs-rest AUC, per-class reports, parameter counts,
  saved local weights and preprocessing statistics.

"Room-aware" means location-specific sensor features, not learned room embeddings
or a graph architecture. This cleaned demonstration does not reproduce or claim
the numerical performance of the original dissertation experiments.

## Quick start

Python 3.10+; CPU is the default. No real household data is included.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python make_synthetic_demo.py --output data/synthetic.csv
python room_aware_transformer_adl.py --csv data/synthetic.csv --epochs 2 --seq-len 10 --output-dir runs/transformer-demo
python lstm_adl_pipeline.py --csv data/synthetic.csv --epochs 2 --seq-len 10 --output-dir runs/lstm-demo
python -m pytest -q
```

## Input contract

Provide `Date`, optional `House` (otherwise one house is assumed), a `P` or `C`
target, and finite numeric sensors named with suffixes `_motion`, `_door`, `_bin`,
`_temp`, `_light`, or `_hum`. Binary sensors should use 0/1. Missing data and duplicate
house/timestamp rows are rejected. Training requires at least two classes and
more days than the combined validation and test day counts. Unseen evaluation
classes cause an error rather than being silently discarded.

`--seq-len` counts **observations**, not minutes unless sampling is one row per
minute. The target is the next observation, not the last input row. The default
`--max-gap-seconds 300` prevents windows spanning long recording outages. Rolling
statistics reset each day; this deliberately sacrifices overnight context.

## Reading results without retraining

Each run writes `metrics.json`, `history.json`, and `model.pt` under its output
directory. Open the JSON files directly to review a completed experiment.
Checkpoints also contain class order, feature order, and scaler statistics.
Use validation data for any model selection; do not tune against test scores.

## Scope and privacy

Validation: three automated tests passed, and both models completed two-epoch
CPU training/evaluation smoke runs on the fictional dataset (Python 3.12,
PyTorch 2.2.2). GPU execution and real-world performance were not validated.

Synthetic metrics demonstrate execution, not activity-recognition quality.
Chronological holdout evaluates future days, not transfer to unseen homes or
participants. Class imbalance, sampling irregularity, and repeated households
require additional study. AUC is undefined for absent classes and is reported
as null for the macro score in that case. No clinical or monitoring deployment
is implied. Keep real recordings, identity mappings, and trained weights on
approved local storage; do not upload them to this repository.
