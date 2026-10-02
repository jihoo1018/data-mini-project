"""Reusable helpers for the DAY2 battery-life modeling notebook.

The notebook keeps experiment choices visible; this module centralizes repeated
metric, persistence, plotting, and submission-audit mechanics.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, pearsonr, spearmanr
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def find_project_root(start: Path) -> Path:
    """Find the repository containing the canonical feature table."""
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "results/battery_eda/cell_features.csv").exists():
            return candidate
    raise FileNotFoundError(
        "results/battery_eda/cell_features.csv를 포함한 프로젝트 루트를 "
        "찾지 못했습니다."
    )


def save_or_validate_json(path: Path, payload: dict, label: str) -> str:
    """Create an immutable JSON record or verify an existing one."""
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            saved = json.load(file)
        assert saved == payload, f"저장된 {label}과 현재 계산이 다릅니다."
        return f"기존 {label}을(를) 재사용합니다."
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return f"{label}을(를) 최초 생성했습니다."


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return the four regression metrics used throughout the project."""
    return {
        "mape_pct": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def prediction_table(frame, prediction, target: str) -> pd.DataFrame:
    """Build the canonical cell-level prediction table."""
    result = frame[["batch", "cell_id", "policy", target]].copy()
    result["prediction"] = prediction
    result["residual_actual_minus_pred"] = result[target] - result["prediction"]
    result["absolute_percentage_error_pct"] = (
        result["residual_actual_minus_pred"].abs() / result[target] * 100
    )
    return result


def evaluate_holdout(
    *, candidate, baseline, train_df, valid_df, selected_features,
    baseline_features, target, cv_summary, model_name, feature_set,
    cv_mape, cv_std,
) -> SimpleNamespace:
    """Fit the locked candidate once and assemble Hold-out evidence."""
    candidate.fit(train_df[selected_features], train_df[target])
    train_metrics = regression_metrics(
        train_df[target], candidate.predict(train_df[selected_features])
    )
    valid_pred = candidate.predict(valid_df[selected_features])
    valid_metrics = regression_metrics(valid_df[target], valid_pred)

    baseline = clone(baseline)
    baseline.fit(train_df[baseline_features], train_df[target])
    baseline_metrics = regression_metrics(
        valid_df[target], baseline.predict(valid_df[baseline_features])
    )

    baseline_cv = cv_summary.loc[cv_summary["model"].eq("Median Baseline")].iloc[0]
    results = pd.DataFrame([
        {
            "role": "Median Baseline", "model": "Median Baseline",
            "feature_set": "A_deltaQ_logvar",
            "cv_mape_mean_pct": float(baseline_cv["cv_mape_mean_pct"]),
            "cv_mape_std_pct": float(baseline_cv["cv_mape_std_pct"]),
            **{f"valid_{key}": value for key, value in baseline_metrics.items()},
        },
        {
            "role": "Locked Candidate", "model": model_name,
            "feature_set": feature_set, "cv_mape_mean_pct": cv_mape,
            "cv_mape_std_pct": cv_std,
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
        },
    ])
    results["gap_cv_valid_pp"] = (
        results["valid_mape_pct"] - results["cv_mape_mean_pct"]
    )
    return SimpleNamespace(
        candidate=candidate,
        train_metrics=train_metrics,
        valid_metrics=valid_metrics,
        baseline_metrics=baseline_metrics,
        gap_cv_valid_pp=valid_metrics["mape_pct"] - cv_mape,
        improvement_vs_baseline_pp=(
            baseline_metrics["mape_pct"] - valid_metrics["mape_pct"]
        ),
        results=results,
        predictions=prediction_table(valid_df, valid_pred, target),
    )


def load_or_evaluate_batch2(
    *, record_path: Path, result_path: Path, prediction_path: Path,
    report_path: Path, model, train_df, test_df, features, target,
    model_name, feature_set, cv_mape, cv_std, valid_mape,
    paper_target_mape, version,
) -> SimpleNamespace:
    """Load the locked Batch 2 evaluation, or create it exactly once."""
    if record_path.exists():
        assert all(path.exists() for path in (result_path, prediction_path, report_path))
        with record_path.open("r", encoding="utf-8") as file:
            record = json.load(file)
        assert record["final_model"] == model_name
        assert record["final_feature_set"] == feature_set
        assert record["final_features"] == features
        assert record["post_test_model_change_allowed"] is False
        result = pd.read_csv(result_path)
        predictions = pd.read_csv(prediction_path)
        report = pd.read_csv(report_path)
        metrics = {
            "mape_pct": float(result.loc[0, "test_mape_pct"]),
            "mae": float(result.loc[0, "test_mae"]),
            "rmse": float(result.loc[0, "test_rmse"]),
            "r2": float(result.loc[0, "test_r2"]),
        }
        return SimpleNamespace(
            model=model, record=record, result=result, predictions=predictions,
            report=report, metrics=metrics,
            gap_valid_test_pp=float(result.loc[0, "gap_valid_test_pp"]),
            gap_target_test_pp=float(result.loc[0, "gap_target_test_pp"]),
            mode="loaded_saved_single_evaluation",
        )

    from datetime import datetime, timezone

    pred = model.predict(test_df[features])
    metrics = regression_metrics(test_df[target], pred)
    gap_valid_test = metrics["mape_pct"] - valid_mape
    gap_target_test = metrics["mape_pct"] - paper_target_mape
    predictions = prediction_table(test_df, pred, target)
    result = pd.DataFrame([{
        "model": model_name, "feature_set": feature_set,
        "n_train_batch1": len(train_df), "n_test_batch2": len(test_df),
        "test_mape_pct": metrics["mape_pct"], "test_mae": metrics["mae"],
        "test_rmse": metrics["rmse"], "test_r2": metrics["r2"],
        "gap_valid_test_pp": gap_valid_test,
        "paper_target_mape_pct": paper_target_mape,
        "gap_target_test_pp": gap_target_test,
    }])
    report = pd.DataFrame([
        {"index": "Train (Batch 1 Group CV)", "mape_pct": cv_mape,
         "note": f"mean ± std = {cv_mape:.2f}% ± {cv_std:.2f}%"},
        {"index": "Valid (Batch 1 Hold-out)", "mape_pct": valid_mape,
         "note": "locked candidate internal generalization"},
        {"index": "Test (Batch 2)", "mape_pct": metrics["mape_pct"],
         "note": "single external batch evaluation"},
        {"index": "Gap (Train-Valid)", "mape_pct": valid_mape - cv_mape,
         "note": "positive means Hold-out degradation"},
        {"index": "Gap (Valid-Test)", "mape_pct": gap_valid_test,
         "note": "positive means batch generalization degradation"},
        {"index": "Gap (Target-Test)", "mape_pct": gap_target_test,
         "note": f"paper target = {paper_target_mape:.1f}%"},
    ])
    result.to_csv(result_path, index=False)
    predictions.to_csv(prediction_path, index=False)
    report.to_csv(report_path, index=False)
    record = {
        "version": version, "evaluation_count": 1,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_data": "all 46 labeled Batch 1 cells",
        "test_data": "all 39 labeled Batch 2 cells",
        "final_model": model_name, "final_feature_set": feature_set,
        "final_features": features,
        "test_mape_pct": round(float(metrics["mape_pct"]), 10),
        "test_mae": round(float(metrics["mae"]), 10),
        "test_rmse": round(float(metrics["rmse"]), 10),
        "test_r2": round(float(metrics["r2"]), 10),
        "gap_valid_test_pp": round(float(gap_valid_test), 10),
        "paper_target_mape_pct": paper_target_mape,
        "gap_target_test_pp": round(float(gap_target_test), 10),
        "post_test_model_change_allowed": False,
    }
    with record_path.open("w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
    return SimpleNamespace(
        model=model, record=record, result=result, predictions=predictions,
        report=report, metrics=metrics, gap_valid_test_pp=gap_valid_test,
        gap_target_test_pp=gap_target_test, mode="new_single_evaluation",
    )


def build_gap_report(cv_mean, cv_std, valid, test, paper_target):
    """Create the professor-format performance table and immutable record."""
    gaps = {
        "train_valid": valid - cv_mean,
        "valid_test": test - valid,
        "target_test": test - paper_target,
    }
    table = pd.DataFrame([
        ("Train (Batch 1 CV)", f"{cv_mean:.2f} ± {cv_std:.2f}", "Group CV"),
        ("Valid (Batch 1 Hold-out)", f"{valid:.2f}", "고정 policy Hold-out"),
        ("Test (Batch 2)", f"{test:.2f}", "최종 평가 1회"),
        ("Gap (Train–Valid)", f"{gaps['train_valid']:+.2f}%p", "Valid − Train; 양수: 과적합 의심"),
        ("Gap (Valid–Test)", f"{gaps['valid_test']:+.2f}%p", "Test − Valid; 양수: 배치 일반화 저하"),
        ("Gap (Target–Test)", f"{gaps['target_test']:+.2f}%p", "Test − 9.1; 양수: 원논문보다 낮은 성능"),
    ], columns=["구분", "MAPE (%)", "비고"])
    record = {
        "version": "v1", "primary_metric": "MAPE", "mape_unit": "%",
        "gap_unit": "%p",
        "positive_gap_meaning": "performance degradation at the later stage",
        "definitions": {
            "gap_train_valid": "valid_mape - train_cv_mape_mean",
            "gap_valid_test": "test_mape - valid_mape",
            "gap_target_test": "test_mape - 9.1",
        },
        "values": {
            "train_cv_mape_mean_pct": round(cv_mean, 10),
            "train_cv_mape_std_pct": round(cv_std, 10),
            "valid_mape_pct": round(valid, 10), "test_mape_pct": round(test, 10),
            "gap_train_valid_pp": round(gaps["train_valid"], 10),
            "gap_valid_test_pp": round(gaps["valid_test"], 10),
            "gap_target_test_pp": round(gaps["target_test"], 10),
        },
        "regression_metrics": ["MAPE", "MAE", "RMSE", "R2"],
        "classification_metrics_not_used": ["F1-score", "Accuracy"],
    }
    return table, record, SimpleNamespace(**gaps)


PLOT_COLORS = {
    "Batch 1 Hold-out": "#0072B2", "Batch 2 Test": "#D55E00",
    "reference": "#333333", "bar": "#4C78A8", "error": "#222222",
    "ape": "#CC79A7",
}
MARKERS = {"Batch 1 Hold-out": "o", "Batch 2 Test": "^"}


def set_plot_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.grid": True,
        "grid.alpha": 0.25, "font.size": 10,
    })


def prediction_plot_frame(valid_predictions, test_predictions, target):
    parts = []
    for frame, name in ((valid_predictions, "Batch 1 Hold-out"),
                        (test_predictions, "Batch 2 Test")):
        part = frame[["cell_id", target, "prediction"]].copy()
        part["dataset"] = name
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    result["residual_actual_minus_pred"] = result[target] - result["prediction"]
    result["absolute_percentage_error_pct"] = (
        result["residual_actual_minus_pred"].abs() / result[target] * 100
    )
    return result


def plot_actual_vs_predicted(frame, target, path):
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for name in MARKERS:
        subset = frame[frame["dataset"].eq(name)]
        ax.scatter(subset[target], subset["prediction"],
                   label=f"{name} (n={len(subset)})", color=PLOT_COLORS[name],
                   marker=MARKERS[name], s=52, alpha=.82, edgecolor="white",
                   linewidth=.6)
    values = pd.concat([frame[target], frame["prediction"]])
    padding = (values.max() - values.min()) * .06
    limits = (float(values.min() - padding), float(values.max() + padding))
    ax.plot(limits, limits, "--", color=PLOT_COLORS["reference"],
            linewidth=1.4, label="Ideal: y = x")
    ax.set(xlim=limits, ylim=limits, title="Actual vs. Predicted Cycle Life",
           xlabel="Actual cycle_life (cycles)",
           ylabel="Predicted cycle_life (cycles)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.show()


def plot_residuals(frame, path):
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    for name in MARKERS:
        subset = frame[frame["dataset"].eq(name)]
        ax.scatter(subset["prediction"], subset["residual_actual_minus_pred"],
                   label=f"{name} (n={len(subset)})", color=PLOT_COLORS[name],
                   marker=MARKERS[name], s=52, alpha=.82, edgecolor="white",
                   linewidth=.6)
    ax.axhline(0, color=PLOT_COLORS["reference"], linestyle="--", linewidth=1.4)
    ax.set(title="Residuals by Predicted Cycle Life",
           xlabel="Predicted cycle_life (cycles)",
           ylabel="Residual: actual − predicted (cycles)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.show()


def best_cv_by_model(cv_summary, order):
    return (cv_summary.sort_values(["model", "cv_mape_mean_pct", "cv_mape_std_pct"])
            .groupby("model", as_index=False, sort=False).head(1)
            .set_index("model").loc[order].reset_index())


def plot_model_comparison(frame, path):
    labels = ["Median\nBaseline", "Linear\nRegression", "Ridge",
              "Elastic\nNet", "Random\nForest", "Gradient\nBoosting"]
    fig, ax = plt.subplots(figsize=(9.2, 5.6)); x = np.arange(len(frame))
    bars = ax.bar(x, frame["cv_mape_mean_pct"], yerr=frame["cv_mape_std_pct"],
                  capsize=5, color=PLOT_COLORS["bar"], edgecolor="none",
                  ecolor=PLOT_COLORS["error"], alpha=.88)
    for bar, mean, std, feature in zip(bars, frame["cv_mape_mean_pct"],
                                       frame["cv_mape_std_pct"], frame["feature_set"]):
        ax.text(bar.get_x() + bar.get_width()/2, mean + std + .45,
                f"{mean:.2f} ± {std:.2f}%\nSet {feature.split('_')[0]}",
                ha="center", va="bottom", fontsize=9)
    ax.set(title="Best Group CV MAPE by Model Family", xlabel="Model",
           ylabel="CV MAPE (%) — mean ± standard deviation",
           xticks=x, xticklabels=labels,
           ylim=(0, float((frame["cv_mape_mean_pct"] + frame["cv_mape_std_pct"]).max())*1.18))
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.show()


def batch2_ape_table(predictions, target):
    result = predictions[["cell_id", "policy", target, "prediction"]].copy()
    result["APE_pct"] = (result[target] - result["prediction"]).abs() / result[target] * 100
    result["residual_actual_minus_pred"] = result[target] - result["prediction"]
    return result.sort_values("APE_pct", ascending=False).reset_index(drop=True)


def plot_batch2_ape(frame, path):
    fig, ax = plt.subplots(figsize=(11.0, 5.8)); x = np.arange(len(frame))
    ax.bar(x, frame["APE_pct"], color=PLOT_COLORS["ape"], edgecolor="none", alpha=.86)
    median = frame["APE_pct"].median()
    ax.axhline(median, color=PLOT_COLORS["reference"], linestyle="--",
               linewidth=1.3, label=f"Median APE = {median:.1f}%")
    ax.set(title="Batch 2 Absolute Percentage Error by Cell",
           xlabel="Batch 2 cell (sorted by APE)",
           ylabel="Absolute percentage error (%)", xticks=x,
           xticklabels=frame["cell_id"])
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.show()


def performance_plot_summary(batch1, valid_plot, test_plot, batch2_ape, target, paths):
    minimum = float(batch1[target].min())
    short = batch2_ape[batch2_ape[target] < minimum].copy()
    over = short[short["residual_actual_minus_pred"] < 0]
    rate = len(over) / len(short) * 100 if len(short) else np.nan
    top = [{"cell_id": str(row.cell_id),
            "actual_cycle_life": round(float(getattr(row, target)), 6),
            "predicted_cycle_life": round(float(row.prediction), 6),
            "ape_pct": round(float(row.APE_pct), 6)}
           for row in batch2_ape.head(5).itertuples(index=False)]
    record = {
        "version": "v1", "batch1_holdout_cells": int(len(valid_plot)),
        "batch2_test_cells": int(len(test_plot)),
        "batch1_min_cycle_life": round(minimum, 10),
        "short_life_definition": "Batch 2 actual cycle_life below Batch 1 minimum",
        "short_life_batch2_cells": int(len(short)),
        "short_life_overpredicted_cells": int(len(over)),
        "short_life_overprediction_rate_pct": round(float(rate), 10),
        "batch2_median_ape_pct": round(float(batch2_ape["APE_pct"].median()), 10),
        "batch2_max_ape_pct": round(float(batch2_ape["APE_pct"].max()), 10),
        "top_5_ape_cells": top, "figure_files": [Path(path).name for path in paths],
    }
    return record, minimum, short, over, rate


def pooled_smd(first, second):
    first, second = pd.Series(first).dropna().astype(float), pd.Series(second).dropna().astype(float)
    variance = (((len(first)-1)*first.var(ddof=1) + (len(second)-1)*second.var(ddof=1))
                / (len(first)+len(second)-2))
    return (second.mean() - first.mean()) / np.sqrt(variance)


def build_domain_evidence(*, df, batch1, batch2, batch3, predictions,
                          cv_summary, final_model, target):
    """Calculate post-test domain evidence without changing the model."""
    dq_pearson = pearsonr(batch1["deltaQ_logvar"], batch1[target])
    dq_spearman = spearmanr(batch1["deltaQ_logvar"], batch1[target])
    slope = float(final_model.named_steps["model"].coef_[0]
                  / final_model.named_steps["scaler"].scale_[0])
    error = batch2_ape_table(predictions, target)
    minimum = float(batch1[target].min())
    short, in_range = error[error[target] < minimum].copy(), error[error[target] >= minimum].copy()
    over_mask = short["residual_actual_minus_pred"] < 0
    life_pearson = pearsonr(error[target], error["APE_pct"])
    life_spearman = spearmanr(error[target], error["APE_pct"])
    target_ks = ks_2samp(batch1[target], batch2[target])
    deltaq_ks = ks_2samp(batch1["deltaQ_logvar"], batch2["deltaQ_logvar"])
    best = (cv_summary.sort_values(["model", "cv_mape_mean_pct"])
            .groupby("model", as_index=False, sort=False).head(1).set_index("model"))
    linear_mean = float(best.loc["Linear Regression", "cv_mape_mean_pct"])
    complex_gaps = {
        name: {"best_feature_set": str(best.loc[name, "feature_set"]),
               "cv_mape_mean_pct": round(float(best.loc[name, "cv_mape_mean_pct"]), 10),
               "cv_mape_std_pct": round(float(best.loc[name, "cv_mape_std_pct"]), 10),
               "gap_vs_linear_pp": round(float(best.loc[name, "cv_mape_mean_pct"]-linear_mean), 10)}
        for name in ("Ridge", "Elastic Net", "Random Forest", "Gradient Boosting")
    }
    unlabeled = df.loc[df[target].isna()].groupby("batch").size().to_dict()
    policy_sizes = batch1.groupby("policy").size()
    record = {
        "version": "v1", "post_test_interpretation_only": True,
        "model_or_feature_changed_after_test": False,
        "deltaq_life_relationship_batch1": {
            "pearson_r": round(float(dq_pearson.statistic), 10),
            "pearson_p": round(float(dq_pearson.pvalue), 18),
            "spearman_rho": round(float(dq_spearman.statistic), 10),
            "spearman_p": round(float(dq_spearman.pvalue), 18),
            "linear_slope_cycles_per_deltaq_logvar_unit": round(slope, 10),
            "linear_slope_cycles_per_0_1_increase": round(slope*.1, 10),
        },
        "batch2_short_life": {
            "batch1_min_cycle_life": minimum, "short_life_cells": int(len(short)),
            "total_labeled_cells": int(len(error)),
            "short_life_share_pct": round(len(short)/len(error)*100, 10),
            "overpredicted_short_life_cells": int(over_mask.sum()),
            "overprediction_rate_pct": round(over_mask.mean()*100, 10),
        },
        "batch2_ape": {
            "short_life_mean_ape_pct": round(float(short["APE_pct"].mean()), 10),
            "short_life_median_ape_pct": round(float(short["APE_pct"].median()), 10),
            "in_range_mean_ape_pct": round(float(in_range["APE_pct"].mean()), 10),
            "in_range_median_ape_pct": round(float(in_range["APE_pct"].median()), 10),
            "life_ape_pearson_r": round(float(life_pearson.statistic), 10),
            "life_ape_spearman_rho": round(float(life_spearman.statistic), 10),
        },
        "distribution_shift": {
            "batch1_target_mean": round(float(batch1[target].mean()), 10),
            "batch2_target_mean": round(float(batch2[target].mean()), 10),
            "batch1_target_median": float(batch1[target].median()),
            "batch2_target_median": float(batch2[target].median()),
            "target_standardized_mean_difference": round(float(pooled_smd(batch1[target], batch2[target])), 10),
            "target_ks_statistic": round(float(target_ks.statistic), 10),
            "target_ks_p": round(float(target_ks.pvalue), 18),
            "batch1_deltaq_logvar_mean": round(float(batch1["deltaQ_logvar"].mean()), 10),
            "batch2_deltaq_logvar_mean": round(float(batch2["deltaQ_logvar"].mean()), 10),
            "deltaq_standardized_mean_difference": round(float(pooled_smd(batch1["deltaQ_logvar"], batch2["deltaQ_logvar"])), 10),
            "deltaq_ks_statistic": round(float(deltaq_ks.statistic), 10),
            "deltaq_ks_p": round(float(deltaq_ks.pvalue), 18),
        },
        "complex_model_comparison": complex_gaps,
        "limitations": {
            "batch1_labeled_cells": int(len(batch1)),
            "batch1_policy_count": int(batch1["policy"].nunique()),
            "cells_per_policy_min": int(policy_sizes.min()),
            "cells_per_policy_max": int(policy_sizes.max()),
            "batch2_unlabeled_cells": int(unlabeled.get("Batch 2", 0)),
            "batch3_labeled_cells": int(len(batch3)),
            "batch3_unlabeled_cells": int(unlabeled.get("Batch 3", 0)),
            "batch3_max_cycle_life": float(batch3[target].max()),
            "newstructure_available_in_feature_table": "newstructure" in df.columns,
        },
    }
    return SimpleNamespace(
        record=record, dq_pearson=dq_pearson, dq_spearman=dq_spearman,
        deltaq_slope_per_01=slope*.1, batch2_error_df=error,
        short_life_error=short, in_range_error=in_range,
        short_life_overprediction_mask=over_mask, life_ape_pearson=life_pearson,
        life_ape_spearman=life_spearman, target_ks=target_ks,
        deltaq_ks=deltaq_ks, unlabeled_by_batch=unlabeled,
        batch1_policy_sizes=policy_sizes,
    )


def build_domain_question_table(e, target_gap, test_mape, target):
    share = len(e.short_life_error) / len(e.batch2_error_df) * 100
    return pd.DataFrame([
        ("초기 ΔQ 변화가 클수록 왜 수명이 짧게 예측되는가?",
         "초기 방전용량 곡선 변화의 변동성이 크다는 것은 초기 열화가 빠르다는 대리신호로 볼 수 있다.",
         f"Batch 1 Pearson r={e.dq_pearson.statistic:.3f}; deltaQ_logvar 0.1 증가 시 선형 예측 {e.deltaq_slope_per_01:.1f} cycles",
         "강한 연관성이며 전기화학적 인과를 직접 증명하지 않는다."),
        ("Batch 2의 짧은 수명이 과대예측되는가?",
         "그렇다. Batch 1 범위 아래의 모든 Batch 2 셀이 과대예측됐다.",
         f"{len(e.short_life_error)}/{len(e.batch2_error_df)} cells ({share:.1f}%); 과대예측 {e.short_life_overprediction_mask.mean()*100:.1f}%",
         "Batch 2 평가 후 모델을 변경하지 않고 한계로 기록한다."),
        ("트리 모델이 534사이클 아래를 예측하지 못했는가?",
         "Random Forest는 leaf의 학습 Target 평균을 출력하므로 학습 최소값 534 아래로 외삽할 수 없다.",
         "Batch 1 Target 범위: 534–1227 cycles",
         "Gradient Boosting은 이론적으로 범위 밖 출력이 가능하지만 구간상수 구조라 안정적인 외삽을 보장하지 않는다."),
        ("MAPE가 짧은 수명 셀에서 커지는가?",
         "그렇다. 짧은 수명 그룹에서 APE가 뚜렷하게 컸다.",
         f"평균 APE {e.short_life_error['APE_pct'].mean():.2f}% vs {e.in_range_error['APE_pct'].mean():.2f}%; Spearman ρ={e.life_ape_spearman.statistic:.3f}",
         "MAPE는 같은 절대오차도 실제값이 작을수록 커지는 특성이 있다."),
        ("Batch 1과 Batch 2의 분포 이동이 얼마나 큰가?",
         "Target과 핵심 Feature 모두 큰 이동이 확인됐다.",
         f"수명 중앙값 858.5→472; Target KS={e.target_ks.statistic:.3f}; deltaQ_logvar KS={e.deltaq_ks.statistic:.3f}; 범위 아래 {share:.1f}%",
         "KS와 Target 비교는 잠긴 Test 평가 이후의 사후 진단이다."),
        ("단순 모델과 복잡한 모델의 차이가 의미 있는가?",
         "복잡한 모델이 개선되지 않았고 변동성도 더 커서 선형회귀가 더 합리적이다.",
         "Linear 7.60±1.57%; Elastic Net 8.80±1.75%; RF 9.66±3.85%; GB 9.42±4.15%",
         "소표본이므로 통계적으로 우월하다고 과장하지 않는다."),
        ("원논문 9.1%보다 나쁜 원인은 무엇인가?",
         "배치 이동, 단수명 외삽, 소표본과 재현 범위 차이가 함께 작용한 것으로 해석한다.",
         f"Test MAPE {test_mape:.2f}%; Target Gap {target_gap:+.2f}%p; 범위 아래 {share:.1f}%",
         "각 요인의 개별 인과 기여도는 현재 실험으로 분리할 수 없다."),
    ], columns=["질문", "결론", "수치 근거", "주의"])
