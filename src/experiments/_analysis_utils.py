from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def safe_mutual_information(feature_frame: pd.DataFrame, labels: pd.Series) -> np.ndarray:
    y = labels.astype(int).to_numpy()
    scores: list[float] = []
    for column_name in feature_frame.columns:
        column = feature_frame[[column_name]].fillna(feature_frame[column_name].mean()).to_numpy()
        if column.shape[0] < 2 or np.nanstd(column) < 1e-12 or len(np.unique(y)) < 2:
            scores.append(0.0)
            continue
        try:
            score = mutual_info_classif(column, y, random_state=42, discrete_features=False, n_neighbors=1)
            scores.append(float(score[0]))
        except Exception:
            scores.append(0.0)
    return np.asarray(scores, dtype=np.float64)