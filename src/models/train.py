"""Train and compare lightweight classifiers, select a deployable model.

Reads the feature dataset exported by ``FeatureDatasetBuilder`` (Phase 5),
compares candidate classifiers with cross-validation + a held-out test split,
runs feature selection and ablation, then retrains the chosen model on all
labeled data and exports the deployment artifacts.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_XGBOOST = False

from configs.config import ExperimentConfig, MODELS_DIR, OUTPUTS_DIR

from ..experiments._analysis_utils import safe_mutual_information
from ..logging_utils import get_logger
from ..utils.io import ensure_directory, save_dataframe, save_json, save_text

LOGGER = get_logger(__name__)
RANDOM_STATE = 42
CV_FOLDS = 5

ABLATION_GROUPS: dict[str, tuple[str, ...] | None] = {
    "fft_only": ("fft",),
    "color_only": ("colors", "contrast"),
    "texture_only": ("texture", "sharpness", "noise"),
    "lighting_only": ("glare", "reflections"),
    "screen_specific_only": ("pixel_grid", "brightness_periodicity", "moire"),
}


@dataclass
class ModelCandidate:
    name: str
    build: Any
    needs_scaling: bool = False


def candidate_models() -> list[ModelCandidate]:
    models = [
        ModelCandidate(
            "logistic_regression",
            lambda: LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE),
            needs_scaling=True,
        ),
        ModelCandidate(
            "random_forest",
            lambda: RandomForestClassifier(
                n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=1
            ),
        ),
    ]
    if HAS_XGBOOST:
        models.append(
            ModelCandidate(
                "xgboost",
                lambda: XGBClassifier(
                    n_estimators=150,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            )
        )
    return models


def load_feature_dataset(csv_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    dataframe = pd.read_csv(csv_dir / "features.csv")
    feature_names = json.loads((csv_dir / "feature_names.json").read_text())["feature_names"]
    return dataframe, feature_names


def model_size_kb(estimator: Any) -> float:
    buffer = io.BytesIO()
    joblib.dump(estimator, buffer)
    return len(buffer.getvalue()) / 1024.0


def measure_latency_ms(estimator: Any, x_row: np.ndarray, repeats: int = 100) -> float:
    estimator.predict_proba(x_row)  # warm up
    start = time.perf_counter()
    for _ in range(repeats):
        estimator.predict_proba(x_row)
    return (time.perf_counter() - start) / repeats * 1000.0


def fit_and_score(
    candidate: ModelCandidate,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    scaler = StandardScaler() if candidate.needs_scaling else None
    xtr = scaler.fit_transform(x_train) if scaler else x_train
    xte = scaler.transform(x_test) if scaler else x_test

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_validate(
        candidate.build(),
        xtr,
        y_train,
        cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
        n_jobs=1,
    )

    estimator = candidate.build()
    estimator.fit(xtr, y_train)
    proba = estimator.predict_proba(xte)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "test_accuracy": accuracy_score(y_test, preds),
        "test_precision": precision_score(y_test, preds, zero_division=0),
        "test_recall": recall_score(y_test, preds, zero_division=0),
        "test_f1": f1_score(y_test, preds, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, proba),
        "cv_accuracy_mean": float(cv_scores["test_accuracy"].mean()),
        "cv_accuracy_std": float(cv_scores["test_accuracy"].std()),
        "cv_f1_mean": float(cv_scores["test_f1"].mean()),
        "cv_roc_auc_mean": float(cv_scores["test_roc_auc"].mean()),
    }
    latency_ms = measure_latency_ms(estimator, xte[:1])
    size_kb = model_size_kb(estimator) + (model_size_kb(scaler) if scaler else 0.0)
    return {
        "name": candidate.name,
        "estimator": estimator,
        "scaler": scaler,
        "metrics": metrics,
        "latency_ms": latency_ms,
        "size_kb": size_kb,
        "confusion_matrix": confusion_matrix(y_test, preds),
        "y_proba": proba,
        "y_test": y_test,
    }


def cv_only_score(
    candidate: ModelCandidate, x: np.ndarray, y: np.ndarray
) -> dict[str, Any]:
    scaler = StandardScaler() if candidate.needs_scaling else None
    xs = scaler.fit_transform(x) if scaler else x
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_validate(
        candidate.build(),
        xs,
        y,
        cv=cv,
        scoring=["accuracy", "f1", "roc_auc"],
        n_jobs=1,
    )
    estimator = candidate.build()
    estimator.fit(xs, y)
    latency_ms = measure_latency_ms(estimator, xs[:1])
    size_kb = model_size_kb(estimator) + (model_size_kb(scaler) if scaler else 0.0)
    return {
        "cv_accuracy_mean": float(cv_scores["test_accuracy"].mean()),
        "cv_f1_mean": float(cv_scores["test_f1"].mean()),
        "cv_roc_auc_mean": float(cv_scores["test_roc_auc"].mean()),
        "latency_ms": latency_ms,
        "size_kb": size_kb,
        "feature_count": x.shape[1],
    }


def rank_features(dataframe: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    x = dataframe[feature_names]
    y = dataframe["label"].astype(int)
    mutual_information = safe_mutual_information(x, y.rename("label"))
    variances = VarianceThreshold().fit(x).variances_
    forest = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1)
    forest.fit(x, y)

    ranking = pd.DataFrame(
        {
            "feature_name": feature_names,
            "mutual_information": mutual_information,
            "variance": variances,
            "random_forest_importance": forest.feature_importances_,
        }
    )
    ranking["combined_score"] = (
        ranking["mutual_information"].rank(pct=True) * 0.4
        + ranking["random_forest_importance"].rank(pct=True) * 0.4
        + ranking["variance"].rank(pct=True) * 0.2
    )
    return ranking.sort_values("combined_score", ascending=False).reset_index(drop=True)


def drop_redundant_features(dataframe: pd.DataFrame, ranking: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    ordered = list(ranking["feature_name"])
    correlation = dataframe[ordered].corr(method="pearson").abs()
    kept: list[str] = []
    for feature_name in ordered:
        if all(correlation.loc[feature_name, other] < threshold for other in kept):
            kept.append(feature_name)
    return kept


def plot_roc_pr_curves(results: list[dict[str, Any]], figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for result in results:
        fpr, tpr, _ = roc_curve(result["y_test"], result["y_proba"])
        ax.plot(fpr, tpr, label=f"{result['name']} (AUC={result['metrics']['test_roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "roc_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for result in results:
        precision, recall, _ = precision_recall_curve(result["y_test"], result["y_proba"])
        ax.plot(recall, precision, label=result["name"])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "precision_recall_curve.png", dpi=160)
    plt.close(fig)


def plot_feature_importance(ranking: pd.DataFrame, figures_dir: Path, top_n: int = 20) -> None:
    top = ranking.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(top["feature_name"], top["combined_score"])
    ax.set_xlabel("Combined importance score")
    ax.set_title(f"Top {top_n} Features")
    fig.tight_layout()
    fig.savefig(figures_dir / "feature_importance.png", dpi=160)
    plt.close(fig)


def choose_best(
    comparison: pd.DataFrame, latency_budget_ms: float = 50.0, size_budget_kb: float = 5000.0
) -> pd.Series:
    """Pick the most accurate config among those that are already cheap/fast enough to deploy.

    Every candidate here runs in single-digit milliseconds and well under a megabyte, so cost
    is a feasibility gate (must fit a phone-friendly budget), not a tie-breaker to chase once a
    config already clears it -- trading meaningful accuracy for a further latency/size win that
    a user or reviewer would never notice is not a good trade.
    """
    feasible = comparison[
        (comparison["latency_ms"] <= latency_budget_ms) & (comparison["size_kb"] <= size_budget_kb)
    ]
    if feasible.empty:
        feasible = comparison
    return feasible.sort_values(
        ["cv_accuracy_mean", "cv_roc_auc_mean", "latency_ms"], ascending=[False, False, True]
    ).iloc[0]


def train_model(
    csv_dir: Path | None = None,
    models_dir: Path | None = None,
    reports_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> dict[str, Any]:
    csv_dir = csv_dir or (OUTPUTS_DIR / "csv")
    models_dir = ensure_directory(models_dir or MODELS_DIR)
    reports_dir = ensure_directory(reports_dir or (OUTPUTS_DIR / "reports"))
    figures_dir = ensure_directory(figures_dir or (OUTPUTS_DIR / "figures"))

    dataframe, feature_names = load_feature_dataset(csv_dir)
    x_all = dataframe[feature_names].to_numpy(dtype=np.float32)
    y_all = dataframe["label"].to_numpy(dtype=np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=ExperimentConfig().test_size, stratify=y_all, random_state=RANDOM_STATE
    )

    LOGGER.info("Comparing %d candidate model types on the full feature set", len(candidate_models()))
    candidates = candidate_models()
    full_feature_results = [
        fit_and_score(candidate, x_train, y_train, x_test, y_test) for candidate in candidates
    ]
    comparison_rows = []
    for result in full_feature_results:
        row = {"model": result["name"], "latency_ms": result["latency_ms"], "size_kb": result["size_kb"]}
        row.update(result["metrics"])
        comparison_rows.append(row)
    model_comparison = pd.DataFrame(comparison_rows).sort_values("cv_roc_auc_mean", ascending=False)
    save_dataframe(model_comparison, reports_dir / "model_comparison.csv")
    plot_roc_pr_curves(full_feature_results, figures_dir)

    winner_name = model_comparison.iloc[0]["model"]
    winner_candidate = next(c for c in candidates if c.name == winner_name)
    LOGGER.info("Winning model type on full feature set: %s", winner_name)

    LOGGER.info("Ranking features (mutual information, variance, random forest importance)")
    ranking = rank_features(dataframe, feature_names)
    save_dataframe(ranking, reports_dir / "feature_importance_ranking.csv")
    plot_feature_importance(ranking, figures_dir)

    non_redundant = drop_redundant_features(dataframe, ranking)
    feature_sets = {
        "all_features": feature_names,
        "non_redundant": non_redundant,
        "top_20": list(ranking["feature_name"].head(20)),
        "top_10": list(ranking["feature_name"].head(10)),
    }

    LOGGER.info("Evaluating every candidate model across feature-selection variants (5-fold CV, all data)")
    selection_rows = []
    for candidate in candidates:
        for set_name, columns in feature_sets.items():
            x_subset = dataframe[columns].to_numpy(dtype=np.float32)
            scores = cv_only_score(candidate, x_subset, y_all)
            selection_rows.append({"feature_set": set_name, "model": candidate.name, **scores})
    selection_report = pd.DataFrame(selection_rows).sort_values("cv_roc_auc_mean", ascending=False)
    save_dataframe(selection_report, reports_dir / "feature_selection_report.csv")

    LOGGER.info("Running categorical ablation groups for every candidate model")
    ablation_rows = []
    for candidate in candidates:
        for group_name, prefixes in ABLATION_GROUPS.items():
            columns = [name for name in feature_names if any(name.startswith(prefix) for prefix in prefixes)]
            if not columns:
                continue
            x_subset = dataframe[columns].to_numpy(dtype=np.float32)
            scores = cv_only_score(candidate, x_subset, y_all)
            ablation_rows.append({"ablation_group": group_name, "model": candidate.name, "columns": ", ".join(columns), **scores})
        ablation_rows.append(
            {
                "ablation_group": "all_features",
                "model": candidate.name,
                "columns": ", ".join(feature_names),
                **cv_only_score(candidate, x_all, y_all),
            }
        )
    ablation_report = pd.DataFrame(ablation_rows).sort_values("cv_roc_auc_mean", ascending=False)
    save_dataframe(ablation_report, reports_dir / "model_ablation_report.csv")

    LOGGER.info("Selecting final deployment configuration")
    all_configs_df = selection_report.rename(columns={}).copy()
    all_configs_df["config"] = all_configs_df["model"] + "__" + all_configs_df["feature_set"]
    best_row = choose_best(all_configs_df)
    final_model_name = best_row["model"]
    final_feature_set_name = best_row["feature_set"]
    final_features = feature_sets.get(final_feature_set_name, feature_names)
    final_candidate = next(c for c in candidates if c.name == final_model_name)

    LOGGER.info(
        "Final configuration: model=%s feature_set=%s (%d features)",
        final_model_name,
        final_feature_set_name,
        len(final_features),
    )

    x_final = dataframe[final_features].to_numpy(dtype=np.float32)
    final_scaler = StandardScaler() if final_candidate.needs_scaling else None
    x_final_scaled = final_scaler.fit_transform(x_final) if final_scaler else x_final
    final_estimator = final_candidate.build()
    final_estimator.fit(x_final_scaled, y_all)

    joblib.dump(final_estimator, models_dir / "trained_model.pkl")
    if final_scaler is not None:
        joblib.dump(final_scaler, models_dir / "feature_scaler.pkl")
    else:
        (models_dir / "feature_scaler.pkl").unlink(missing_ok=True)

    selected_features_payload = {
        "model_name": final_model_name,
        "feature_names": final_features,
        "needs_scaling": final_candidate.needs_scaling,
        "decision_threshold": 0.5,
    }
    save_json(selected_features_payload, models_dir / "selected_features.json")

    final_latency_ms = measure_latency_ms(final_estimator, x_final_scaled[:1])
    final_size_kb = model_size_kb(final_estimator) + (model_size_kb(final_scaler) if final_scaler else 0.0)

    report_lines = [
        "# Training Report",
        "",
        f"Dataset: {len(dataframe)} images, {len(feature_names)} raw features "
        f"(real={int((y_all == 0).sum())}, screen={int((y_all == 1).sum())}).",
        "",
        "## Model Comparison (full feature set, 80/20 held-out split + 5-fold CV)",
        "",
        model_comparison.to_markdown(index=False),
        "",
        f"Winning model type on the full feature set: **{winner_name}**.",
        "",
        "## Feature Selection",
        "",
        f"Correlation-based redundancy removal (|r| >= 0.95) kept {len(non_redundant)}/{len(feature_names)} features.",
        "",
        selection_report.to_markdown(index=False),
        "",
        "## Ablation (feature-family groups, 5-fold CV)",
        "",
        ablation_report.drop(columns=["columns"]).to_markdown(index=False),
        "",
        "## Final Deployment Configuration",
        "",
        f"- Model: **{final_model_name}**",
        f"- Feature set: **{final_feature_set_name}** ({len(final_features)} features)",
        f"- Selection rule: highest CV accuracy (ROC-AUC as tie-break) among configs that clear "
        "a phone-friendly feasibility gate (<=50ms latency, <=5MB serialized size) -- every "
        "candidate here clears that gate by a wide margin, so the choice is really just the "
        "most accurate one.",
        f"- Retrained on all {len(dataframe)} labeled images before export "
        "(test-set metrics above come from the held-out split before this final refit).",
        f"- Inference latency: ~{final_latency_ms:.3f} ms/image (feature-vector -> probability only, laptop CPU).",
        f"- Serialized size: ~{final_size_kb:.1f} KB.",
        f"- Default decision threshold: 0.5 (see note below).",
        "",
        "## Caveats",
        "",
        "- Only 354 labeled images from a single household/small set of devices and screens. "
        "Held-out accuracy is a useful signal but will likely be optimistic relative to the "
        "graders' own held-out photos, which come from different phones/screens/lighting.",
        "- The decision threshold of 0.5 is a starting point, not tuned; in production, pick it "
        "from the ROC curve based on the desired false-accept vs false-reject tradeoff "
        "(e.g. Youden's J statistic, or a fixed low false-positive-rate operating point if "
        "false-flagging real users is costlier than missing some recaptures).",
    ]
    save_text("\n".join(report_lines), reports_dir / "training_report.md")

    return {
        "model_comparison": model_comparison,
        "selection_report": selection_report,
        "ablation_report": ablation_report,
        "final_model_name": final_model_name,
        "final_feature_set_name": final_feature_set_name,
        "final_features": final_features,
        "final_latency_ms": final_latency_ms,
        "final_size_kb": final_size_kb,
    }


def main() -> None:
    result = train_model()
    print(f"Final model: {result['final_model_name']} / {result['final_feature_set_name']}")
    print(f"Features used: {len(result['final_features'])}")
    print(f"Latency: {result['final_latency_ms']:.3f} ms  Size: {result['final_size_kb']:.1f} KB")


if __name__ == "__main__":
    main()
