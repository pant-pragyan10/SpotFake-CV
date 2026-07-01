# Spot the Fake Photo

Classifies one RGB image as either:

- `0`: genuine camera photo of a real-world object
- `1`: photo of a display or printout showing an image (a "recapture")

See **NOTE.md** for the half-page summary (approach, honest accuracy, latency, cost, threshold
choice). Full brief: **ASSIGNMENT.pdf** (in `../ML Latest Assignment/`).

## Usage

```
pip install -r requirements.txt
python predict.py path/to/image.jpg
```

## Live demo

A small camera + upload web UI backed by the exact same model (same `requirements.txt`, `flask`
is already in it):

```
python webapp/app.py        # http://localhost:5050
```

Deployed on Vercel: see **Deploying to Vercel** below.

"Camera" mode grabs a frame every ~700ms and scores it live; "Upload" mode lets you drop in an
existing photo. Both hit the same in-process `ScreenRecaptureDetector` used by `predict.py` — no
cloud call, no separate model.

Prints one float in `[0, 1]` (0 = real, 1 = screen). No other output.

## Architecture

- `src/preprocessing/` — image loading, resizing, normalization.
- `src/features/` — 12 forensic feature extractors (FFT/periodicity, moiré, pixel-grid,
  brightness periodicity, glare, reflections, noise, texture, sharpness, edges, contrast, colors),
  a registry (`src/registry.py`) that auto-discovers them, and a `FeatureVectorBuilder`
  (`src/features/feature_vector.py`) that runs all of them into one 91-dim vector per image.
- `src/pipeline.py` — runs preprocessing + feature extraction end to end (used by the dataset
  builder and experiment lab; not on the hot inference path).
- `src/experiments/` — Phase 5/6 lab: dataset export, statistics, correlation, importance,
  ablation, latency benchmarking, and report/figure generation. Entry points:
  `scripts/collect_dataset.py` (build `outputs/csv/features.{csv,npy}` + labels) and
  `scripts/benchmark.py` (full statistics/correlation/importance/ablation/visualization lab).
- `src/models/train.py` — compares Logistic Regression, Random Forest, and XGBoost with 5-fold CV
  + a held-out split, ranks features (mutual information, variance, RF importance), evaluates
  feature-selection and feature-family ablation variants, and picks the most accurate config among
  those that already clear a phone-friendly latency/size budget. Entry point: `scripts/train.py`.
  Writes `saved_models/trained_model.pkl` + `feature_scaler.pkl` (if needed) +
  `selected_features.json`, and `outputs/reports/training_report.md`.
- `src/models/inference.py` — the production predictor: preprocess -> extract only the selected
  features -> load the saved model once -> return a probability. `predict.py` is a thin wrapper
  around this with logging disabled and errors routed to stderr.
- `scripts/predict_benchmark.py` — runs the production predictor over a folder (with `real/` +
  `screen/` subfolders if you want accuracy too) and reports latency/throughput/accuracy.

## Reproducing from scratch

```
pip install -r requirements.txt -r requirements-train.txt   # runtime + training/analysis deps
PYTHONPATH=. python scripts/collect_dataset.py   # Phase 5: build the feature dataset
PYTHONPATH=. python scripts/benchmark.py         # Phase 5: stats/correlation/importance/ablation/figures
PYTHONPATH=. python scripts/train.py             # Phase 6: compare models, select, export artifacts
python predict.py data/raw/real/<some_image>.jpg # Phase 7: run the production predictor
```

`requirements.txt` alone is the minimal deployment set (no pandas/matplotlib/xgboost/tabulate —
those are only needed for training and the analysis lab, and are split into
`requirements-train.txt`).

## Deploying to Vercel

The webapp deploys as a single Python serverless function:

```
npx vercel login
npx vercel --prod
```

`api/index.py` re-exports the same Flask `app` from `webapp/app.py`, `vercel.json` excludes
`data/`, `outputs/figures`, and `notebooks/` from the function bundle (they're not needed at
runtime and would otherwise bloat it), and the root `requirements.txt` is pinned to the exact
versions the shipped model was trained with, since Vercel's Python runtime uses Python 3.12+
while this was developed against 3.9 -- pinning avoids scikit-learn pickle skew across versions.

## Plug-In Model

To add a new forensic feature, create one module under `src/features/`, implement the
`FeatureExtractor` interface, and register it with `register_feature_extractor`. It's picked up
automatically by the registry and included in the next `collect_dataset.py` run — no other code
changes needed.
