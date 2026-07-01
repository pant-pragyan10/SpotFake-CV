# Training Report

Dataset: 354 images, 91 raw features (real=182, screen=172).

## Model Comparison (full feature set, 80/20 held-out split + 5-fold CV)

| model               |   latency_ms |   size_kb |   test_accuracy |   test_precision |   test_recall |   test_f1 |   test_roc_auc |   cv_accuracy_mean |   cv_accuracy_std |   cv_f1_mean |   cv_roc_auc_mean |
|:--------------------|-------------:|----------:|----------------:|-----------------:|--------------:|----------:|---------------:|-------------------:|------------------:|-------------:|------------------:|
| random_forest       |    5.14206   |  660.634  |        0.957746 |         1        |      0.911765 |  0.953846 |       0.992051 |           0.918546 |         0.0269671 |     0.91817  |          0.977714 |
| xgboost             |    0.0711508 |  140.831  |        0.929577 |         0.939394 |      0.911765 |  0.925373 |       0.988076 |           0.922368 |         0.0393404 |     0.92304  |          0.975196 |
| logistic_regression |    0.0584696 |    4.2793 |        0.915493 |         0.9375   |      0.882353 |  0.909091 |       0.984897 |           0.901253 |         0.0280932 |     0.896774 |          0.96133  |

Winning model type on the full feature set: **random_forest**.

## Feature Selection

Correlation-based redundancy removal (|r| >= 0.95) kept 76/91 features.

| feature_set   | model               |   cv_accuracy_mean |   cv_f1_mean |   cv_roc_auc_mean |   latency_ms |   size_kb |   feature_count |
|:--------------|:--------------------|-------------------:|-------------:|------------------:|-------------:|----------:|----------------:|
| top_20        | random_forest       |           0.923783 |     0.923761 |          0.98337  |    4.96754   | 769.384   |              20 |
| all_features  | random_forest       |           0.946358 |     0.946487 |          0.981382 |    6.14657   | 779.853   |              91 |
| non_redundant | random_forest       |           0.937948 |     0.936485 |          0.980915 |    5.23031   | 812.353   |              76 |
| all_features  | xgboost             |           0.937948 |     0.938406 |          0.980125 |    0.0695396 | 144.363   |              91 |
| top_20        | xgboost             |           0.921046 |     0.92133  |          0.979167 |    0.06713   | 148.088   |              20 |
| non_redundant | xgboost             |           0.929497 |     0.92974  |          0.977421 |    0.0697133 | 143.961   |              76 |
| top_10        | random_forest       |           0.921006 |     0.92155  |          0.976123 |    5.13825   | 822.665   |              10 |
| top_10        | xgboost             |           0.915332 |     0.914905 |          0.973105 |    0.0670575 | 151.822   |              10 |
| top_20        | logistic_regression |           0.915372 |     0.914883 |          0.966142 |    0.0577671 |   2.06836 |              20 |
| non_redundant | logistic_regression |           0.918189 |     0.915234 |          0.961654 |    0.05739   |   3.81836 |              76 |
| all_features  | logistic_regression |           0.912555 |     0.909437 |          0.959287 |    0.0586687 |   4.2793  |              91 |
| top_10        | logistic_regression |           0.904064 |     0.904437 |          0.957236 |    0.0574525 |   1.75586 |              10 |

## Ablation (feature-family groups, 5-fold CV)

| ablation_group       | model               |   cv_accuracy_mean |   cv_f1_mean |   cv_roc_auc_mean |   latency_ms |    size_kb |   feature_count |
|:---------------------|:--------------------|-------------------:|-------------:|------------------:|-------------:|-----------:|----------------:|
| all_features         | random_forest       |           0.946358 |     0.946487 |          0.981382 |    7.93883   |  779.853   |              91 |
| all_features         | xgboost             |           0.937948 |     0.938406 |          0.980125 |    0.07917   |  144.363   |              91 |
| all_features         | logistic_regression |           0.912555 |     0.909437 |          0.959287 |    0.0562288 |    4.2793  |              91 |
| texture_only         | xgboost             |           0.884185 |     0.882841 |          0.951207 |    0.0676679 |  156.425   |              17 |
| color_only           | random_forest       |           0.889899 |     0.888877 |          0.950208 |    4.98935   | 1092.51    |               6 |
| color_only           | logistic_regression |           0.906841 |     0.905967 |          0.949838 |    0.0564671 |    1.63086 |               6 |
| texture_only         | random_forest       |           0.86169  |     0.858713 |          0.949804 |    5.02487   | 1047.04    |              17 |
| color_only           | xgboost             |           0.878632 |     0.877455 |          0.938886 |    0.0761288 |  156.01    |               6 |
| texture_only         | logistic_regression |           0.878632 |     0.879653 |          0.927876 |    0.0590871 |    1.9668  |              17 |
| lighting_only        | random_forest       |           0.709014 |     0.702168 |          0.794561 |    4.96221   | 1536.88    |               5 |
| lighting_only        | xgboost             |           0.706237 |     0.700867 |          0.773788 |    0.0678058 |  160.812   |               5 |
| screen_specific_only | random_forest       |           0.661087 |     0.651951 |          0.748951 |    5.17575   | 1472.35    |              12 |
| lighting_only        | logistic_regression |           0.692113 |     0.70301  |          0.736857 |    0.0621471 |    1.5918  |               5 |
| fft_only             | logistic_regression |           0.689256 |     0.69212  |          0.731286 |    0.0572483 |    1.7793  |              11 |
| screen_specific_only | xgboost             |           0.663662 |     0.652062 |          0.730918 |    0.0676458 |  164.893   |              12 |
| screen_specific_only | logistic_regression |           0.666398 |     0.673411 |          0.726805 |    0.0574858 |    1.81836 |              12 |
| fft_only             | random_forest       |           0.655533 |     0.656041 |          0.691233 |    5.44085   | 1408.13    |              11 |
| fft_only             | xgboost             |           0.652555 |     0.647974 |          0.673847 |    0.0723317 |  163.896   |              11 |

## Final Deployment Configuration

- Model: **random_forest**
- Feature set: **all_features** (91 features)
- Selection rule: highest CV accuracy (ROC-AUC as tie-break) among configs that clear a phone-friendly feasibility gate (<=50ms latency, <=5MB serialized size) -- every candidate here clears that gate by a wide margin, so the choice is really just the most accurate one.
- Retrained on all 354 labeled images before export (test-set metrics above come from the held-out split before this final refit).
- Inference latency: ~4.984 ms/image (feature-vector -> probability only, laptop CPU).
- Serialized size: ~779.9 KB.
- Default decision threshold: 0.5 (see note below).

## Caveats

- Only 354 labeled images from a single household/small set of devices and screens. Held-out accuracy is a useful signal but will likely be optimistic relative to the graders' own held-out photos, which come from different phones/screens/lighting.
- The decision threshold of 0.5 is a starting point, not tuned; in production, pick it from the ROC curve based on the desired false-accept vs false-reject tradeoff (e.g. Youden's J statistic, or a fixed low false-positive-rate operating point if false-flagging real users is costlier than missing some recaptures).