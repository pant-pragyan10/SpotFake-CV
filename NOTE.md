# Spot the Fake Photo — Note

**Approach.** No deep model — 91 hand-engineered forensic features grouped into 12 families
(FFT/periodicity, moiré interference, pixel-grid/scanline spacing, brightness periodicity, glare,
specular reflections, sensor noise, texture entropy, sharpness/blur, edge orientation, contrast,
color/white-balance) computed per image (`src/features/`), then a classifier on top. I compared
Logistic Regression, Random Forest, and XGBoost with stratified 5-fold CV + an 80/20 held-out
split (`src/models/train.py`); Random Forest on the full 91-feature vector had the best accuracy
and was already comfortably small/fast, so it's the deployed model (`saved_models/trained_model.pkl`,
~780 KB, no GPU, no scaler needed).

**Accuracy (honest numbers).** 354 self-collected photos (182 real, 172 screen recaptures) from
one household, a handful of phones/screens. 5-fold CV accuracy: **94.6% (±2.7 pts)**; held-out
20% split: **95.8%** (68/71, precision 100%, recall 91.2%). I expect this to be somewhat optimistic
against the graders' own held-out photos — different phones, screens, lighting and photographer
than what the model has seen. Ablation shows FFT-only and screen-specific-only feature groups are
individually weak (65-75% accuracy); color, texture and lighting features carry most of the signal,
and combining all families is what gets to ~95%.

**Latency.** Two numbers, because they differ a lot:
- **~1.3-1.8 s** for a single fresh `python predict.py image.jpg` process (laptop CPU, M-series
  Mac) — almost all of that is Python starting up and importing numpy/scipy/opencv/scikit-learn
  and loading the model, which happens once per process.
- **~153 ms/image** if the process stays warm (e.g. a long-running service or a batch loop that
  imports once and calls the predictor repeatedly) — measured over all 354 images via
  `scripts/predict_benchmark.py`. Of that, image decode + resize of full-resolution phone photos
  is ~64 ms, feature extraction across 12 extractors is ~48 ms, and the classifier itself is <1 ms.

If graders invoke `predict.py` as a fresh subprocess per image, expect the first number; if this
were deployed as a service (which is how it'd actually run in production), expect the second.

**Cost per image.** On-device: free — the model is 780 KB, sklearn-only, trivially bundleable into
a phone app with no network round-trip. Cloud fallback: assuming ~6.5 img/s per CPU core (measured),
a 4-vCPU instance parallelizes to roughly ~25 img/s; on a ~$0.05-0.10/hr small CPU VM that's on the
order of **$0.01-0.03 per 1,000 images** (rough — excludes storage, networking, and orchestration
overhead). On-device is strictly better here since there's no accuracy tradeoff for going local.

**Cutoff score.** Shipped with a default threshold of 0.5. In practice I'd calibrate it from the ROC
curve (`outputs/figures/roc_curve.png`) against the real cost ratio: falsely flagging a genuine
photo (annoys a real user) vs. missing a recapture (lets fraud through) — e.g. Youden's J for a
balanced cost, or a low-false-positive-rate operating point if user friction is the bigger concern.

**What I'd improve with more time.** (1) Much more varied training data — more phones, more display
types (OLED/LCD/e-ink), more printouts, outdoor lighting, off-axis angles — the current set is too
homogeneous to trust the 95% number at face value. (2) Track drift: as cheaters adapt (better
screens, screen protectors, cropping out bezels), periodically recollect data and retrain; a
feature-based model degrades gracefully but should be monitored, not "fire and forget." (3) A
second, orthogonal signal (EXIF/metadata plausibility, sensor-pattern-noise consistency) as a cheap
ensemble check before falling back to manual review on borderline scores.
