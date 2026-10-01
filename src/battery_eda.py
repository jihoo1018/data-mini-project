#!/usr/bin/env python3
"""Reproducible EDA for the MIT-Stanford battery Batch 1/2/3 MAT files.

The script reads MATLAB v7.3/HDF5 files without loading full raw traces into
memory. It produces cell-level features, figures, summary tables, and a
standalone HTML report.
"""

from __future__ import annotations

import base64
import io
import math
import re
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results" / "battery_eda"
FIG = OUT / "figures"

FILES = {
    "Batch 1": DATA / "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "Batch 2": DATA / "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "Batch 3": DATA / "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}

BATCH_COLORS = {"Batch 1": "#2563eb", "Batch 2": "#f97316", "Batch 3": "#16a34a"}


def decode_char(obj: h5py.Dataset) -> str:
    values = np.asarray(obj[()]).reshape(-1)
    return "".join(chr(int(x)) for x in values if int(x) > 0)


def deref(file: h5py.File, ref):
    """Dereference without asking HDF5 for the object's expensive full path."""
    return file[ref]


def safe_at(arr: np.ndarray, index: int) -> float:
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if index >= len(arr) or not np.isfinite(arr[index]):
        return np.nan
    return float(arr[index])


def clean_slice(arr: np.ndarray, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(arr, dtype=float).reshape(-1)
    stop = min(stop, len(values))
    x = np.arange(start + 1, stop + 1, dtype=float)
    y = values[start:stop]
    mask = np.isfinite(y)
    return x[mask], y[mask]


def linear_slope(arr: np.ndarray, start: int, stop: int) -> float:
    x, y = clean_slice(arr, start, stop)
    if len(y) < 3 or np.nanstd(y) == 0:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def parse_policy(policy: str) -> tuple[float, float, float, str]:
    core = policy.replace("-", "_")
    match = re.match(r"^(\d+(?:_\d+)?)C_(\d+)PER_(\d+(?:_\d+)?)C", core)
    if not match:
        return np.nan, np.nan, np.nan, "variable/other"
    first = float(match.group(1).replace("_", "."))
    switch = float(match.group(2))
    second = float(match.group(3).replace("_", "."))
    structure = "new" if "NEWSTRUCTURE" in policy else "standard"
    if "SLOWCYCLE" in policy:
        structure = "slow-cycle"
    return first, switch, second, structure


def knee_proxy(cycles: np.ndarray, qd: np.ndarray) -> tuple[float, float]:
    """Return a smoothed maximum-negative-curvature knee proxy.

    This is deliberately reported as an exploratory proxy rather than a
    validated knee detector.
    """
    cycles = np.asarray(cycles, dtype=float).reshape(-1)
    qd = np.asarray(qd, dtype=float).reshape(-1)
    mask = np.isfinite(cycles) & np.isfinite(qd) & (cycles >= 2) & (qd > 0.5) & (qd < 1.5)
    x, y = cycles[mask], qd[mask]
    if len(y) < 80:
        return np.nan, np.nan
    order = np.argsort(x)
    x, y = x[order], y[order]
    window = max(11, int(len(y) * 0.05))
    window = min(window if window % 2 else window + 1, 101)
    if window >= len(y):
        window = len(y) - 1 if len(y) % 2 == 0 else len(y)
    if window < 7:
        return np.nan, np.nan
    smooth = savgol_filter(y, window_length=window, polyorder=2, mode="interp")
    xn = (x - x.min()) / max(x.max() - x.min(), 1.0)
    dy = np.gradient(smooth, xn)
    d2 = np.gradient(dy, xn)
    valid = (xn >= 0.20) & (xn <= 0.90)
    if not valid.any():
        return np.nan, np.nan
    candidates = np.where(valid)[0]
    idx = candidates[np.argmin(d2[valid])]
    return float(x[idx]), float(xn[idx])


def extract_batch(batch: str, path: Path):
    rows: list[dict] = []
    degradation: list[dict] = []
    delta_curves: list[dict] = []

    with h5py.File(path, "r") as file:
        batch_group = file["batch"]
        n_cells = batch_group["cycle_life"].shape[0]

        for i in range(n_cells):
            cell_id = f"b{batch[-1]}c{i}"
            cycle_life = float(np.asarray(deref(file, batch_group["cycle_life"][i, 0]))[0, 0])
            policy = decode_char(deref(file, batch_group["policy"][i, 0]))
            policy_readable = decode_char(deref(file, batch_group["policy_readable"][i, 0]))
            c_rate_1, switch_soc, c_rate_2, protocol_type = parse_policy(policy)

            summary = deref(file, batch_group["summary"][i, 0])
            s = {name: np.asarray(summary[name][()]).reshape(-1).astype(float) for name in summary.keys()}
            cycles = s["cycle"]
            qd = s["QDischarge"]
            ir = s["IR"]
            tavg = s["Tavg"]
            tmax = s["Tmax"]
            chargetime = s["chargetime"]

            knee_cycle, knee_fraction = knee_proxy(cycles, qd)

            cycle_group = deref(file, batch_group["cycles"][i, 0])
            qdlin_refs = cycle_group["Qdlin"]
            delta_q = np.full(1000, np.nan)
            voltage = np.asarray(deref(file, batch_group["Vdlin"][i, 0])[()]).reshape(-1).astype(float)
            if qdlin_refs.shape[0] > 99:
                q10 = np.asarray(deref(file, qdlin_refs[9, 0])[()]).reshape(-1).astype(float)
                q100 = np.asarray(deref(file, qdlin_refs[99, 0])[()]).reshape(-1).astype(float)
                m = min(len(q10), len(q100), len(voltage))
                delta_q = q100[:m] - q10[:m]
                voltage = voltage[:m]

            finite_dq = delta_q[np.isfinite(delta_q)]
            if len(finite_dq):
                dq_var = float(np.var(finite_dq, ddof=1)) if len(finite_dq) > 1 else np.nan
                dq_stats = {
                    "deltaQ_min": float(np.min(finite_dq)),
                    "deltaQ_mean": float(np.mean(finite_dq)),
                    "deltaQ_abs_mean": float(np.mean(np.abs(finite_dq))),
                    "deltaQ_range": float(np.ptp(finite_dq)),
                    "deltaQ_var": dq_var,
                    "deltaQ_logvar": float(np.log10(dq_var)) if dq_var > 0 else np.nan,
                    "deltaQ_skew": float(stats.skew(finite_dq, bias=False)),
                    "deltaQ_kurtosis": float(stats.kurtosis(finite_dq, bias=False)),
                }
            else:
                dq_stats = {k: np.nan for k in [
                    "deltaQ_min", "deltaQ_mean", "deltaQ_abs_mean", "deltaQ_range",
                    "deltaQ_var", "deltaQ_logvar", "deltaQ_skew", "deltaQ_kurtosis",
                ]}

            def segment_mean(arr, start, stop):
                _, values = clean_slice(arr, start, stop)
                return float(np.nanmean(values)) if len(values) else np.nan

            def segment_std(arr, start, stop):
                _, values = clean_slice(arr, start, stop)
                return float(np.nanstd(values, ddof=1)) if len(values) > 1 else np.nan

            has_label = bool(np.isfinite(cycle_life))
            row = {
                "batch": batch,
                "cell_id": cell_id,
                "cycle_life": cycle_life,
                "life_group": ("Short (<500)" if cycle_life < 500 else ("Long (>1000)" if cycle_life > 1000 else "Mid (500-1000)")) if has_label else np.nan,
                "label_550": float(cycle_life >= 550) if has_label else np.nan,
                "policy": policy,
                "policy_readable": policy_readable,
                "protocol_type": protocol_type,
                "c_rate_1": c_rate_1,
                "switch_soc": switch_soc,
                "c_rate_2": c_rate_2,
                "summary_cycles": int(len(cycles)),
                "Qd_mean_2_10": segment_mean(qd, 1, 10),
                "Qd_10": safe_at(qd, 9),
                "Qd_100": safe_at(qd, 99),
                "Qd_change_10_100": safe_at(qd, 99) - safe_at(qd, 9),
                "Qd_slope_10_100": linear_slope(qd, 9, 100),
                "IR_mean_2_100": segment_mean(ir, 1, 100),
                "IR_100": safe_at(ir, 99),
                "IR_slope_10_100": linear_slope(ir, 9, 100),
                "Tavg_mean_2_100": segment_mean(tavg, 1, 100),
                "Tmax_max_2_100": float(np.nanmax(tmax[1:min(100, len(tmax))])),
                "charge_time_mean_2_100": segment_mean(chargetime, 1, 100),
                "charge_time_std_2_100": segment_std(chargetime, 1, 100),
                "knee_cycle_proxy": knee_cycle,
                "knee_fraction_proxy": knee_fraction,
                **dq_stats,
            }
            rows.append(row)

            valid = np.isfinite(cycles) & np.isfinite(qd) & (cycles >= 2) & (qd > 0.5) & (qd < 1.5)
            degradation.append({
                "batch": batch,
                "cell_id": cell_id,
                "cycle_life": cycle_life,
                "cycle": cycles[valid],
                "qd": qd[valid],
            })
            delta_curves.append({
                "batch": batch,
                "cell_id": cell_id,
                "cycle_life": cycle_life,
                "voltage": voltage,
                "delta_q": delta_q[: len(voltage)],
            })

    return pd.DataFrame(rows), degradation, delta_curves


def save_figure(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_life_distribution(df: pd.DataFrame) -> Path:
    df = df.dropna(subset=["cycle_life"]).copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    bins = np.arange(100, math.ceil(df.cycle_life.max() / 100) * 100 + 101, 100)
    for batch in FILES:
        sub = df[df.batch == batch]
        axes[0, 0].hist(sub.cycle_life, bins=bins, alpha=0.38, label=batch, color=BATCH_COLORS[batch])
    axes[0, 0].set(title="Cycle-life distribution", xlabel="Cycle life", ylabel="Cells")
    axes[0, 0].legend()

    sns.boxplot(data=df, x="batch", y="cycle_life", hue="batch", palette=BATCH_COLORS, legend=False, ax=axes[0, 1])
    sns.stripplot(data=df, x="batch", y="cycle_life", color="#111827", alpha=0.55, size=3, ax=axes[0, 1])
    axes[0, 1].set(title="Batch shift in cycle life", xlabel="", ylabel="Cycle life")

    order = ["Short (<500)", "Mid (500-1000)", "Long (>1000)"]
    comp = pd.crosstab(df.batch, df.life_group, normalize="index").reindex(columns=order, fill_value=0)
    comp.plot(kind="bar", stacked=True, color=["#dc2626", "#f59e0b", "#16a34a"], ax=axes[1, 0])
    axes[1, 0].set(title="Life-group composition", xlabel="", ylabel="Share")
    axes[1, 0].legend(title="", fontsize=8)
    axes[1, 0].tick_params(axis="x", rotation=0)

    cls = pd.crosstab(df.batch, df.label_550, normalize="index").reindex(columns=[0, 1], fill_value=0)
    cls.columns = ["Short label (<550)", "Long label (>=550)"]
    cls.plot(kind="bar", stacked=True, color=["#ef4444", "#3b82f6"], ax=axes[1, 1])
    axes[1, 1].set(title="Classification balance at 550 cycles", xlabel="", ylabel="Share")
    axes[1, 1].legend(title="", fontsize=8)
    axes[1, 1].tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return save_figure(fig, "01_cycle_life_distribution.png")


def plot_degradation(degradation: list[dict]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    max_life = max(d["cycle_life"] for d in degradation if np.isfinite(d["cycle_life"]))
    norm = plt.Normalize(0, max_life)
    for ax, batch in zip(axes, FILES):
        for item in [d for d in degradation if d["batch"] == batch]:
            color = plt.cm.viridis(norm(item["cycle_life"])) if np.isfinite(item["cycle_life"]) else "#cbd5e1"
            alpha = 0.45 if np.isfinite(item["cycle_life"]) else 0.25
            ax.plot(item["cycle"], item["qd"], lw=0.65, alpha=alpha, color=color)
        ax.set(title=batch, xlabel="Cycle", xlim=(0, max_life * 1.03), ylim=(0.75, 1.16))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Discharge capacity Qd (Ah)")
    sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    fig.colorbar(sm, ax=axes, label="Cycle life", fraction=0.025, pad=0.02)
    fig.suptitle("Cell-level degradation curves", y=1.02, fontsize=15)
    return save_figure(fig, "02_degradation_curves.png")


def plot_normalized_degradation(degradation: list[dict], df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    # Summary traces usually stop one cycle before the reported cycle life.
    # Ending at 0.99 avoids an artificial all-NaN column at exactly 1.00.
    grid = np.linspace(0.01, 0.99, 197)
    for batch in FILES:
        curves = []
        for item in [d for d in degradation if d["batch"] == batch and np.isfinite(d["cycle_life"])]:
            x = item["cycle"] / item["cycle_life"]
            y = item["qd"]
            baseline = np.nanmedian(y[(item["cycle"] >= 2) & (item["cycle"] <= 10)])
            if np.isfinite(baseline) and baseline > 0:
                order = np.argsort(x)
                curves.append(np.interp(grid, x[order], (y / baseline)[order], left=np.nan, right=np.nan))
        arr = np.asarray(curves)
        median = np.nanmedian(arr, axis=0)
        lo = np.nanpercentile(arr, 25, axis=0)
        hi = np.nanpercentile(arr, 75, axis=0)
        axes[0].plot(grid, median, lw=2.2, color=BATCH_COLORS[batch], label=batch)
        axes[0].fill_between(grid, lo, hi, color=BATCH_COLORS[batch], alpha=0.13)
    axes[0].axhline(0.8, color="#111827", ls="--", lw=1, label="80% SOH")
    axes[0].set(title="Median normalized degradation (IQR band)", xlabel="Fraction of cell life", ylabel="Qd / initial Qd", ylim=(0.75, 1.04))
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    labeled = df.dropna(subset=["cycle_life"])
    sns.boxplot(data=labeled, x="batch", y="knee_fraction_proxy", hue="batch", palette=BATCH_COLORS, legend=False, ax=axes[1])
    sns.stripplot(data=labeled, x="batch", y="knee_fraction_proxy", color="#111827", alpha=0.45, size=3, ax=axes[1])
    axes[1].set(title="Exploratory curvature-knee proxy", xlabel="", ylabel="Knee / cycle life", ylim=(0.15, 0.95))
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return save_figure(fig, "03_normalized_degradation.png")


def plot_deltaq(delta_curves: list[dict], df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for col, batch in enumerate(FILES):
        items = [d for d in delta_curves if d["batch"] == batch and np.isfinite(d["cycle_life"])]
        groups = {"<550": [], ">=550": []}
        for item in items:
            key = ">=550" if item["cycle_life"] >= 550 else "<550"
            groups[key].append(item["delta_q"])
            axes[0, col].plot(item["voltage"], item["delta_q"], color="#94a3b8", alpha=0.16, lw=0.6)
        for label, color in [("<550", "#dc2626"), (">=550", "#2563eb")]:
            if groups[label]:
                m = min(len(x) for x in groups[label])
                med = np.nanmedian(np.vstack([x[:m] for x in groups[label]]), axis=0)
                voltage = items[0]["voltage"][:m]
                axes[0, col].plot(voltage, med, color=color, lw=2.2, label=f"Median {label}")
        axes[0, col].set(title=batch, xlabel="Voltage (V)", xlim=(3.5, 2.0))
        axes[0, col].grid(alpha=0.2)
        axes[0, col].legend(fontsize=8)
    axes[0, 0].set_ylabel("ΔQ(V): cycle 100 - cycle 10 (Ah)")

    plot_df = df.dropna(subset=["cycle_life"]).copy()
    plot_df["label"] = plot_df.label_550.map({0: "<550", 1: ">=550"})
    sns.boxplot(data=plot_df, x="batch", y="deltaQ_logvar", hue="label", palette={"<550": "#dc2626", ">=550": "#2563eb"}, ax=axes[1, 0])
    axes[1, 0].set(title="log10 variance of ΔQ(V)", xlabel="", ylabel="log10(var ΔQ)")
    axes[1, 0].tick_params(axis="x", rotation=0)

    sns.scatterplot(data=plot_df, x="deltaQ_logvar", y="cycle_life", hue="batch", palette=BATCH_COLORS, style="label", s=55, ax=axes[1, 1])
    axes[1, 1].set(title="ΔQ variance vs cycle life", xlabel="log10(var ΔQ)", ylabel="Cycle life")
    axes[1, 1].grid(alpha=0.2)

    sns.scatterplot(data=plot_df, x="deltaQ_min", y="cycle_life", hue="batch", palette=BATCH_COLORS, s=55, ax=axes[1, 2])
    axes[1, 2].set(title="Minimum ΔQ vs cycle life", xlabel="min ΔQ(V)", ylabel="Cycle life")
    axes[1, 2].grid(alpha=0.2)
    fig.suptitle("Early-cycle ΔQ(V) signal", y=1.01, fontsize=15)
    fig.tight_layout()
    return save_figure(fig, "04_deltaq_analysis.png")


def plot_policy(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    parsed = df.dropna(subset=["cycle_life", "c_rate_1", "c_rate_2", "switch_soc"]).copy()
    sns.scatterplot(
        data=parsed, x="c_rate_1", y="cycle_life", hue="batch", palette=BATCH_COLORS,
        size="switch_soc", sizes=(30, 180), alpha=0.8, ax=axes[0]
    )
    axes[0].set(title="First-step C-rate vs cycle life", xlabel="First-step C-rate", ylabel="Cycle life")
    axes[0].grid(alpha=0.2)

    policy_stats = (
        df.dropna(subset=["cycle_life"]).groupby(["batch", "policy"], as_index=False)
        .agg(mean_life=("cycle_life", "mean"), n=("cycle_life", "size"))
        .sort_values("mean_life", ascending=False)
    )
    top = policy_stats.groupby("batch", group_keys=False).head(5).copy()
    top["label"] = top.batch.str.replace("Batch ", "B") + " | " + top.policy
    top = top.sort_values("mean_life")
    axes[1].barh(top.label, top.mean_life, color=[BATCH_COLORS[b] for b in top.batch])
    for y, (_, row) in enumerate(top.iterrows()):
        axes[1].text(row.mean_life + 12, y, f"n={int(row.n)}", va="center", fontsize=8)
    axes[1].set(title="Top 5 policies by mean life within each batch", xlabel="Mean cycle life", ylabel="")
    axes[1].grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return save_figure(fig, "05_policy_and_crate.png")


FEATURES = [
    "Qd_mean_2_10", "Qd_100", "Qd_change_10_100", "Qd_slope_10_100",
    "IR_mean_2_100", "IR_100", "IR_slope_10_100", "Tavg_mean_2_100",
    "Tmax_max_2_100", "charge_time_mean_2_100", "charge_time_std_2_100",
    "deltaQ_min", "deltaQ_mean", "deltaQ_abs_mean", "deltaQ_range",
    "deltaQ_logvar", "deltaQ_skew", "deltaQ_kurtosis", "c_rate_1",
    "switch_soc", "c_rate_2",
]


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for batch in ["All", *FILES.keys()]:
        sub = df if batch == "All" else df[df.batch == batch]
        for feature in FEATURES:
            pair = sub[[feature, "cycle_life"]].dropna()
            if len(pair) >= 5 and pair[feature].nunique() > 1:
                rho, p = stats.spearmanr(pair[feature], pair.cycle_life)
                r, p_r = stats.pearsonr(pair[feature], pair.cycle_life)
            else:
                rho = p = r = p_r = np.nan
            rows.append({"scope": batch, "feature": feature, "spearman_rho": rho, "spearman_p": p, "pearson_r": r, "pearson_p": p_r, "n": len(pair)})
    return pd.DataFrame(rows)


def plot_correlations(df: pd.DataFrame, corr_long: pd.DataFrame) -> Path:
    pivot = corr_long.pivot(index="feature", columns="scope", values="spearman_rho")[["All", "Batch 1", "Batch 2", "Batch 3"]]
    order = pivot["All"].abs().sort_values(ascending=False).index
    fig, axes = plt.subplots(1, 2, figsize=(17, 9), gridspec_kw={"width_ratios": [0.8, 1.35]})
    sns.heatmap(pivot.loc[order], cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", ax=axes[0], cbar_kws={"label": "Spearman rho"})
    axes[0].set(title="Feature association with cycle life", xlabel="", ylabel="")

    selected = list(order[:10])
    matrix = df[selected].corr(method="spearman")
    sns.heatmap(matrix, cmap="vlag", center=0, vmin=-1, vmax=1, square=True, ax=axes[1], cbar_kws={"label": "Spearman rho"})
    axes[1].set(title="Multicollinearity among top target-associated features", xlabel="", ylabel="")
    fig.tight_layout()
    return save_figure(fig, "06_correlations_and_multicollinearity.png")


def compute_vif(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    x = df[columns].copy()
    x = x.fillna(x.median(numeric_only=True))
    x = x.loc[:, x.nunique() > 1]
    z = (x - x.mean()) / x.std(ddof=0)
    rows = []
    for col in z.columns:
        y = z[col].to_numpy(float)
        others = z.drop(columns=col).to_numpy(float)
        design = np.column_stack([np.ones(len(others)), others])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ beta
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        vif = np.inf if r2 >= 0.999999 else 1 / (1 - r2)
        rows.append({"feature": col, "VIF": vif})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def batch_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("batch").agg(
        raw_cells=("cell_id", "size"),
        labeled_cells=("cycle_life", "count"),
        min_life=("cycle_life", "min"),
        median_life=("cycle_life", "median"),
        mean_life=("cycle_life", "mean"),
        max_life=("cycle_life", "max"),
        std_life=("cycle_life", "std"),
        short_lt500=("cycle_life", lambda x: int((x < 500).sum())),
        long_gt1000=("cycle_life", lambda x: int((x > 1000).sum())),
        class1_ge550=("label_550", "sum"),
        policies=("policy", "nunique"),
        knee_fraction_median=("knee_fraction_proxy", "median"),
    ).reset_index()


def outlier_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for batch, sub in df.groupby("batch"):
        q1, q3 = sub.cycle_life.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        for _, row in sub[(sub.cycle_life < low) | (sub.cycle_life > high)].iterrows():
            rows.append({"batch": batch, "cell_id": row.cell_id, "cycle_life": row.cycle_life, "policy": row.policy, "IQR_rule": "low" if row.cycle_life < low else "high"})
    return pd.DataFrame(rows, columns=["batch", "cell_id", "cycle_life", "policy", "IQR_rule"])


def ks_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    batches = list(FILES)
    for i, a in enumerate(batches):
        for b in batches[i + 1 :]:
            x = df.loc[df.batch == a, "cycle_life"].dropna()
            y = df.loc[df.batch == b, "cycle_life"].dropna()
            ks = stats.ks_2samp(x, y)
            mw = stats.mannwhitneyu(x, y, alternative="two-sided")
            rows.append({"comparison": f"{a} vs {b}", "median_gap": float(y.median() - x.median()), "KS_stat": ks.statistic, "KS_p": ks.pvalue, "Mann_Whitney_p": mw.pvalue})
    return pd.DataFrame(rows)


def strongest_text(corr_long: pd.DataFrame, scope: str, n: int = 5) -> str:
    top = corr_long[corr_long.scope == scope].dropna().copy()
    top["abs_rho"] = top.spearman_rho.abs()
    top = top.nlargest(n, "abs_rho")
    return ", ".join(f"{r.feature} ({r.spearman_rho:+.2f})" for r in top.itertuples())


def image_data_uri(path: Path) -> str:
    mime = "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def html_table(df: pd.DataFrame, digits: int = 3) -> str:
    out = df.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    return out.to_html(index=False, classes="data-table", border=0, escape=True)


def write_report(df, summary, unlabeled, outliers, ks, corr_long, vif, figures):
    labeled = df.dropna(subset=["cycle_life"]).copy()
    medians = summary.set_index("batch")["median_life"].to_dict()
    label_rates = labeled.groupby("batch").label_550.mean().to_dict()
    knee_medians = labeled.groupby("batch").knee_fraction_proxy.median().to_dict()
    policy_numeric = labeled.dropna(subset=["c_rate_1", "switch_soc", "c_rate_2"])
    policy_corr = []
    for feature in ["c_rate_1", "switch_soc", "c_rate_2"]:
        rho, p = stats.spearmanr(policy_numeric[feature], policy_numeric.cycle_life)
        policy_corr.append({"condition": feature, "spearman_rho": rho, "p_value": p, "n": len(policy_numeric)})
    policy_corr = pd.DataFrame(policy_corr)

    top_policies = (
        labeled.groupby(["batch", "policy"], as_index=False)
        .agg(n=("cycle_life", "size"), mean_life=("cycle_life", "mean"), median_life=("cycle_life", "median"))
        .sort_values(["batch", "mean_life"], ascending=[True, False])
        .groupby("batch", group_keys=False).head(5)
    )

    top_all = strongest_text(corr_long, "All")
    top_by_batch = {b: strongest_text(corr_long, b, 3) for b in FILES}
    vif_warn = vif[np.isfinite(vif.VIF) & (vif.VIF >= 10)]

    cards = "".join(
        f"""<div class='card'><h3>{b}</h3><div class='metric'>{int(summary.set_index('batch').loc[b, 'labeled_cells'])} labeled</div>
        <p>원본 {int(summary.set_index('batch').loc[b, 'raw_cells'])}개 셀<br>수명 중앙값 <b>{medians[b]:.0f}</b> cycles<br>550-cycle 장수명 비율 <b>{label_rates[b]*100:.1f}%</b><br>
        knee proxy 중앙값 <b>{knee_medians[b]:.2f}</b> of life</p></div>"""
        for b in FILES
    )

    image_blocks = {name: image_data_uri(path) for name, path in figures.items()}
    source_rows = "".join(
        f"<tr><td>{batch}</td><td>{path.name}</td><td>{path.stat().st_size / (1024**3):.2f} GB</td></tr>"
        for batch, path in FILES.items()
    )

    html = f"""<!doctype html>
<html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Battery Batch 1–3 EDA Report</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Apple SD Gothic Neo','Noto Sans KR',Arial,sans-serif;margin:0;background:#f8fafc;color:#172033;line-height:1.65}}
.wrap{{max-width:1180px;margin:auto;padding:38px 28px 80px}} h1{{font-size:34px;line-height:1.2;margin-bottom:8px}} h2{{margin-top:48px;border-bottom:2px solid #dbe4f0;padding-bottom:8px}} h3{{margin-bottom:6px}}
.subtitle{{color:#526176;margin-top:0}} .cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:25px 0}} .card{{background:#fff;border:1px solid #dce5ef;border-radius:14px;padding:18px;box-shadow:0 2px 10px #0f172a0a}} .metric{{font-size:26px;font-weight:750;color:#1d4ed8}}
.callout{{background:#eef6ff;border-left:5px solid #2563eb;padding:16px 19px;border-radius:8px;margin:18px 0}} .warn{{background:#fff7ed;border-left-color:#f97316}}
img{{width:100%;height:auto;background:white;border:1px solid #dce5ef;border-radius:12px;margin:14px 0}} .data-table{{width:100%;border-collapse:collapse;background:#fff;font-size:13px;overflow:auto;display:block}}
.data-table th,.data-table td{{padding:8px 10px;border:1px solid #dce5ef;white-space:nowrap;text-align:right}} .data-table th{{background:#edf2f7}} .data-table td:first-child,.data-table th:first-child{{text-align:left}}
code{{background:#e9eef5;padding:2px 5px;border-radius:4px}} li{{margin:6px 0}} .small{{font-size:13px;color:#5d6b7d}} @media(max-width:800px){{.cards{{grid-template-columns:1fr}}.wrap{{padding:20px 14px}}}}
</style></head><body><main class='wrap'>
<h1>MIT–Stanford Battery Dataset<br>Batch 1·2·3 EDA</h1>
<p class='subtitle'>가이드의 5개 질문을 셀 단위로 분석한 재현 가능한 탐색 보고서</p>
<div class='callout'><b>핵심 결론:</b> 세 배치는 수명 분포와 충전 프로토콜 구성이 다르므로 무작위 통합 분할은 피해야 합니다. 회귀 모델은 Batch 1로 학습·검증하고 Batch 2를 최종 테스트, Batch 3를 추가 외부 검증으로 유지하는 것이 타당합니다. ΔQ(V) 특징은 수명과 강한 관계를 보이지만 배치별 관계 크기가 달라 batch shift를 함께 보고해야 합니다.</div>
<div class='cards'>{cards}</div>

<h2>0. 데이터와 분석 범위</h2>
<table class='data-table'><thead><tr><th>Batch</th><th>File</th><th>Size</th></tr></thead><tbody>{source_rows}</tbody></table>
<ul><li>분석 단위: 개별 셀. 원본 <b>{len(df)}</b>개 중 <code>cycle_life</code>가 있는 <b>{len(labeled)}</b>개를 수명 관련 EDA에 사용했습니다.</li><li>초기 특징은 누수 방지를 위해 100사이클 이내 데이터만 사용했습니다.</li><li>ΔQ(V)는 <code>Qdlin(cycle 100) − Qdlin(cycle 10)</code>으로 계산했습니다.</li><li>수명 구간은 가이드의 단수명 &lt;500, 장수명 &gt;1000을 사용하고, 분류용 라벨은 550사이클을 사용했습니다.</li></ul>
<div class='callout warn'><b>데이터 품질:</b> Batch 2의 8개와 Batch 3의 2개 셀은 원본 <code>cycle_life</code>가 NaN입니다. 정답 라벨이 필요한 통계·상관·분류 비율에서 제외했으며 임의 보간하지 않았습니다.</div>
{html_table(unlabeled[['batch','cell_id','policy','summary_cycles','Qd_100']], 2)}

<h2>1. Cycle Life 분포</h2>
<img src='{image_blocks['life']}' alt='Cycle-life distribution'>
{html_table(summary, 2)}
<h3>해석</h3><ul>
<li>수명 중앙값은 Batch 1 <b>{medians['Batch 1']:.0f}</b>, Batch 2 <b>{medians['Batch 2']:.0f}</b>, Batch 3 <b>{medians['Batch 3']:.0f}</b>사이클입니다.</li>
<li>550사이클 기준 장수명 비율은 각각 <b>{label_rates['Batch 1']*100:.1f}%</b>, <b>{label_rates['Batch 2']*100:.1f}%</b>, <b>{label_rates['Batch 3']*100:.1f}%</b>입니다. 분류 선택 시 배치별 class balance 차이를 반드시 보고해야 합니다.</li>
<li>두 표본 분포의 차이 검정은 아래와 같습니다. p-value는 표본 구성과 프로토콜 차이의 신호이지 인과관계가 아닙니다.</li></ul>
{html_table(ks, 4)}
<h3>IQR 기준 이상치 후보</h3>{html_table(outliers, 1) if len(outliers) else '<p>배치 내부 IQR 기준 이상치가 없습니다.</p>'}
<div class='callout'><b>모델링 시사점:</b> Batch를 합친 뒤 random split하면 쉬운 배치 패턴을 학습해 성능이 부풀 수 있습니다. Batch 1 내부 CV/hold-out과 Batch 2·3 외부 테스트를 분리하십시오.</div>

<h2>2. 방전 용량 열화와 knee 탐색</h2>
<img src='{image_blocks['degradation']}' alt='Degradation curves'>
<img src='{image_blocks['normalized']}' alt='Normalized degradation'>
<h3>해석</h3><ul>
<li>동일 배치 안에서도 EOL 도달 시점의 분산이 크고, 용량 저하는 완전한 직선형보다 후반 가속형 패턴을 보입니다.</li>
<li>곡률 기반 탐색적 knee proxy 중앙값은 Batch 1 <b>{knee_medians['Batch 1']:.2f}</b>, Batch 2 <b>{knee_medians['Batch 2']:.2f}</b>, Batch 3 <b>{knee_medians['Batch 3']:.2f}</b>로 전체 수명 대비 위치가 다릅니다.</li>
<li>이 knee 값은 EDA용 proxy입니다. 운영용 knee 검출에는 Savitzky–Golay 설정과 지속성 조건을 별도 검증해야 합니다.</li></ul>

<h2>3. 초기 ΔQ(V): 100사이클 − 10사이클</h2>
<img src='{image_blocks['deltaq']}' alt='Delta Q analysis'>
<h3>해석</h3><ul>
<li>장·단수명 그룹의 중앙 ΔQ(V) 형상과 분산이 달라 초기 열화 신호로 사용할 수 있습니다.</li>
<li>전체 데이터에서 수명과 연관이 큰 특징 상위 항목은 <b>{top_all}</b>입니다.</li>
<li>단, 배치별 상위 특징은 Batch 1: {top_by_batch['Batch 1']}; Batch 2: {top_by_batch['Batch 2']}; Batch 3: {top_by_batch['Batch 3']}로 달라집니다.</li></ul>
<div class='callout'><b>모델링 시사점:</b> 논문 재현형 회귀라면 <code>deltaQ_logvar</code>, <code>deltaQ_min</code>, 초기 Qd·IR·온도·충전시간 통계를 후보로 두되, Batch 1에서만 전처리와 특징 선택을 학습해야 합니다.</div>

<h2>4. 충전 조건(C-rate)과 수명</h2>
<img src='{image_blocks['policy']}' alt='Policy and C-rate'>
<h3>수치형 충전조건과 수명의 단변량 관계</h3>{html_table(policy_corr, 4)}
<h3>배치별 평균수명 상위 프로토콜</h3>{html_table(top_policies, 1)}
<h3>해석</h3><ul><li>C-rate와 수명의 관계는 단순 단조 관계만으로 설명되지 않습니다. 1단계·2단계 C-rate, 전환 SOC, 구조 변경, batch가 함께 얽혀 있습니다.</li><li>프로토콜당 셀 수가 작은 경우가 많으므로 평균 순위만으로 최적 충전정책이라고 결론 내리면 안 됩니다.</li><li><code>VARCHARGE</code> 정책은 고정 2-step C-rate 파싱에서 제외했지만 셀 자체는 모든 다른 EDA에 포함했습니다.</li></ul>

<h2>5. 상관관계와 다중공선성</h2>
<img src='{image_blocks['correlation']}' alt='Correlation and multicollinearity'>
<h3>VIF 진단</h3>{html_table(vif.head(15), 2)}
<p>{'VIF 10 이상 특징이 있어 선형모델에서는 중복 특징 제거 또는 정규화가 필요합니다: ' + ', '.join(vif_warn.feature.tolist()) if len(vif_warn) else '선택 특징에서 VIF 10 이상 항목은 확인되지 않았습니다.'}</p>
<div class='callout warn'><b>주의:</b> pooled 상관은 batch 차이를 섞은 값입니다. 전체 상관의 부호·크기와 배치별 상관이 다르면 Simpson's paradox 또는 domain shift 가능성이 있으므로, EDA 표에는 반드시 pooled와 batch별 값을 함께 제시하십시오.</div>

<h2>6. 권장 모델 설계 전략</h2>
<ol><li><b>문제:</b> 초기 100사이클로 <code>cycle_life</code>를 예측하는 회귀.</li><li><b>분할:</b> Batch 1 내부에서 셀/프로토콜 단위 hold-out과 CV → 튜닝 종료 후 Batch 2 한 번 평가 → Batch 3 추가 외부 평가.</li><li><b>전처리:</b> 결측 대치·스케일링·특징 선택은 Batch 1 train에서만 fit.</li><li><b>후보 모델:</b> Elastic Net(해석·다중공선성 대응), Random Forest/Gradient Boosting(비선형), 단순 기준모델(중앙값 예측).</li><li><b>지표:</b> 프로젝트 필수 MAPE와 함께 MAE·RMSE·R²를 보조 지표로 보고. Train–Valid, Valid–Test, Target–Test gap을 모두 계산.</li><li><b>분류를 선택한다면:</b> 550사이클 threshold, Accuracy와 F1을 동시에 보고하고 Batch별 class balance 및 confusion matrix를 제시.</li></ol>

<h2>7. 재현 파일</h2>
<ul><li><code>cell_features.csv</code>: 셀 단위 메타데이터와 초기 100사이클 특징</li><li><code>batch_summary.csv</code>: 배치별 분포 요약</li><li><code>feature_correlations.csv</code>: pooled·배치별 Pearson/Spearman 상관</li><li><code>vif.csv</code>: 다중공선성 진단</li><li><code>battery_eda.py</code>: 원본 MAT에서 보고서를 다시 만드는 전체 코드</li></ul>
<p class='small'>생성 기준: 로컬 원본 MAT 파일. 보고서의 knee는 탐색적 곡률 proxy이며 검증된 운영 임계값이 아닙니다.</p>
</main></body></html>"""
    (OUT / "EDA_report.html").write_text(html, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    frames = []
    degradation = []
    delta_curves = []
    for batch, path in FILES.items():
        frame, deg, delta = extract_batch(batch, path)
        frames.append(frame)
        degradation.extend(deg)
        delta_curves.extend(delta)
        print(f"Extracted {batch}: {len(frame)} cells")

    df = pd.concat(frames, ignore_index=True)
    summary = batch_summary(df)
    unlabeled = df[df.cycle_life.isna()].copy()
    outliers = outlier_table(df)
    ks = ks_table(df)
    corr_long = correlation_table(df)

    vif_columns = [
        "Qd_mean_2_10", "Qd_change_10_100", "IR_mean_2_100", "IR_slope_10_100",
        "Tavg_mean_2_100", "charge_time_mean_2_100", "deltaQ_min",
        "deltaQ_logvar", "deltaQ_skew", "c_rate_1", "switch_soc", "c_rate_2",
    ]
    vif = compute_vif(df.dropna(subset=["cycle_life"]), vif_columns)

    df.to_csv(OUT / "cell_features.csv", index=False)
    summary.to_csv(OUT / "batch_summary.csv", index=False)
    unlabeled.to_csv(OUT / "unlabeled_cells.csv", index=False)
    outliers.to_csv(OUT / "cycle_life_outliers.csv", index=False)
    ks.to_csv(OUT / "batch_distribution_tests.csv", index=False)
    corr_long.to_csv(OUT / "feature_correlations.csv", index=False)
    vif.to_csv(OUT / "vif.csv", index=False)

    figures = {
        "life": plot_life_distribution(df),
        "degradation": plot_degradation(degradation),
        "normalized": plot_normalized_degradation(degradation, df),
        "deltaq": plot_deltaq(delta_curves, df),
        "policy": plot_policy(df),
        "correlation": plot_correlations(df, corr_long),
    }
    write_report(df, summary, unlabeled, outliers, ks, corr_long, vif, figures)
    print(summary.to_string(index=False))
    print(f"Report: {OUT / 'EDA_report.html'}")


if __name__ == "__main__":
    main()
