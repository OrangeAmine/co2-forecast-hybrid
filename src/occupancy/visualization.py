"""Visualization functions for occupancy inference results.

Provides 9 plot functions for comparing detector outputs, evaluating
inter-detector agreement, and visualizing occupancy patterns over time.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from src.occupancy.evaluation import ConsensusResult

logger = logging.getLogger(__name__)

# Consistent color palette for detectors
DETECTOR_COLORS = {
    "absolute_threshold": "#1f77b4",
    "rate_of_change": "#ff7f0e",
    "adaptive_threshold": "#2ca02c",
    "hybrid": "#d62728",
    "state_machine": "#9467bd",
    "diarra": "#8c564b",
}


def _get_color(name: str) -> str:
    """Get detector color, falling back to gray."""
    return DETECTOR_COLORS.get(name, "#7f7f7f")


def _save_fig(fig: plt.Figure, path: Path) -> None:
    """Save figure and close."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 1. Detector comparison timeline (binary heatmap + CO2 overlay)
# ---------------------------------------------------------------------------

def plot_detector_comparison_timeline(
    detections: dict[str, np.ndarray],
    co2: np.ndarray,
    timestamps: np.ndarray | None = None,
    title: str = "Occupancy Detector Comparison",
    max_samples: int = 168 * 4,
    output_path: Path | None = None,
) -> plt.Figure:
    """Stacked binary heatmap showing all detectors over a time window.

    Each row is a detector (yellow=occupied, blue=unoccupied).
    CO2 is overlaid as a line on a secondary y-axis.

    Args:
        detections: Dict of {detector_name: binary_array}.
        co2: CO2 concentration array (same length).
        timestamps: Optional datetime array for x-axis.
        title: Plot title.
        max_samples: Maximum samples to display (default ~1 week at 1h).
        output_path: Path to save figure.
    """
    names = list(detections.keys())
    n_detectors = len(names)

    # Limit to max_samples for readability
    n = min(len(co2), max_samples)
    co2_plot = co2[:n]
    x = timestamps[:n] if timestamps is not None else np.arange(n)

    fig, (ax_heat, ax_co2) = plt.subplots(
        2, 1, figsize=(16, 4 + n_detectors * 0.6),
        height_ratios=[n_detectors, 2],
        sharex=True,
    )

    # Binary heatmap
    matrix = np.array([detections[name][:n] for name in names])
    cmap = ListedColormap(["#3498db", "#f39c12"])  # blue=0, yellow=1
    ax_heat.imshow(
        matrix, aspect="auto", cmap=cmap, interpolation="nearest",
        extent=[0, n, n_detectors, 0],
    )
    ax_heat.set_yticks(np.arange(n_detectors) + 0.5)
    ax_heat.set_yticklabels(names, fontsize=9)
    ax_heat.set_title(title, fontsize=13, fontweight="bold")

    # CO2 overlay
    ax_co2.plot(np.arange(n), co2_plot, color="#2c3e50", linewidth=0.8)
    ax_co2.set_ylabel("CO2 (ppm)", fontsize=10)
    ax_co2.set_xlabel("Time step (hours)" if timestamps is None else "Time")
    ax_co2.grid(True, alpha=0.3)

    fig.tight_layout()
    if output_path:
        _save_fig(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# 2. Occupancy rates bar chart
# ---------------------------------------------------------------------------

def plot_occupancy_rates_bar(
    occupancy_rates: dict[str, float],
    title: str = "Occupancy Rate by Detector",
    output_path: Path | None = None,
) -> plt.Figure:
    """Bar chart showing fraction of time classified as occupied.

    Args:
        occupancy_rates: Dict of {detector_name: fraction_occupied}.
        title: Plot title.
        output_path: Path to save figure.
    """
    names = list(occupancy_rates.keys())
    rates = [occupancy_rates[n] for n in names]
    colors = [_get_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, rates, color=colors, edgecolor="white", linewidth=0.5)

    # Add percentage labels on bars
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{rate:.1%}", ha="center", va="bottom", fontsize=10,
        )

    ax.set_ylabel("Occupancy Rate", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(rates) * 1.15 if rates else 1.0)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    if output_path:
        _save_fig(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# 3. Agreement heatmap
# ---------------------------------------------------------------------------

def plot_agreement_heatmap(
    matrix: np.ndarray,
    detector_names: list[str],
    metric_name: str = "Cohen's Kappa",
    title: str | None = None,
    output_path: Path | None = None,
) -> plt.Figure:
    """Annotated heatmap of pairwise agreement metrics.

    Args:
        matrix: Square matrix (n_detectors, n_detectors).
        detector_names: Detector names for axis labels.
        metric_name: Name of the metric (for title/colorbar).
        title: Plot title (auto-generated if None).
        output_path: Path to save figure.
    """
    if title is None:
        title = f"Pairwise {metric_name}"

    n = len(detector_names)
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1 if "Kappa" in metric_name else 0, vmax=1)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            color = "white" if abs(matrix[i, j]) > 0.7 else "black"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(detector_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(detector_names, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(metric_name, fontsize=10)

    fig.tight_layout()
    if output_path:
        _save_fig(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# 4. Consistency over time
# ---------------------------------------------------------------------------

def plot_consistency_over_time(
    consistency_scores: np.ndarray,
    timestamps: np.ndarray | None = None,
    title: str = "Detector Consistency Over Time",
    output_path: Path | None = None,
) -> plt.Figure:
    """Line plot of per-timestep consistency score.

    Consistency = fraction of detectors agreeing with the majority.
    Score of 1.0 = all detectors agree, 0.5 = maximum disagreement.

    Args:
        consistency_scores: Array of consistency values (n_samples,).
        timestamps: Optional datetime array for x-axis.
        title: Plot title.
        output_path: Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(14, 4))

    x = timestamps if timestamps is not None else np.arange(len(consistency_scores))

    # Rolling average for readability
    window = min(24, len(consistency_scores) // 10)
    if window > 1:
        rolling = pd.Series(consistency_scores).rolling(
            window=window, min_periods=1, center=True
        ).mean().values
        ax.fill_between(x, 0.5, rolling, alpha=0.3, color="#2ecc71")
        ax.plot(x, rolling, color="#27ae60", linewidth=1.0,
                label=f"Rolling mean ({window}h)")
    else:
        ax.plot(x, consistency_scores, color="#27ae60", linewidth=0.5)

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect agreement")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random agreement")

    ax.set_ylabel("Consistency Score", fontsize=11)
    ax.set_xlabel("Time step" if timestamps is None else "Time")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0.45, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if output_path:
        _save_fig(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# 5. Actual vs predicted occupancy comparison
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted_occupancy(
    detections_actual: dict[str, np.ndarray],
    detections_predicted: dict[str, np.ndarray],
    co2_actual: np.ndarray,
    co2_predicted: np.ndarray,
    max_samples: int = 168,
    title: str = "Actual vs Predicted CO2: Occupancy Detection",
    output_path: Path | None = None,
) -> plt.Figure:
    """Side-by-side timeline comparing detector outputs on actual vs predicted CO2.

    Args:
        detections_actual: Detector outputs on actual CO2.
        detections_predicted: Detector outputs on predicted CO2.
        co2_actual: Actual CO2 values.
        co2_predicted: Predicted CO2 values.
        max_samples: Max samples to display.
        title: Plot title.
        output_path: Path to save figure.
    """
    names = list(detections_actual.keys())
    n_det = len(names)
    n = min(len(co2_actual), max_samples)

    fig, axes = plt.subplots(
        2, 1, figsize=(16, 4 + n_det * 0.5),
        sharex=True, sharey=True,
    )

    cmap = ListedColormap(["#3498db", "#f39c12"])

    for ax_idx, (ax, detections, co2, subtitle) in enumerate(zip(
        axes,
        [detections_actual, detections_predicted],
        [co2_actual, co2_predicted],
        ["Actual CO2", "Predicted CO2"],
    )):
        matrix = np.array([detections[name][:n] for name in names])
        ax.imshow(
            matrix, aspect="auto", cmap=cmap, interpolation="nearest",
            extent=[0, n, n_det, 0],
        )
        ax.set_yticks(np.arange(n_det) + 0.5)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title(subtitle, fontsize=11)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    axes[-1].set_xlabel("Time step (hours)")

    fig.tight_layout()
    if output_path:
        _save_fig(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# 6. Daily occupancy profile
# ---------------------------------------------------------------------------

def plot_daily_occupancy_profile(
    detections: dict[str, np.ndarray],
    hours: np.ndarray,
    title: str = "Average Occupancy by Hour of Day",
    output_path: Path | None = None,
) -> plt.Figure:
    """Line plot of average occupancy probability per hour, per detector.

    Args:
        detections: Dict of {detector_name: binary_array}.
        hours: Hour-of-day array (0-23) for each sample.
        title: Plot title.
        output_path: Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    for name, binary in detections.items():
        # Group by hour and compute mean occupancy
        df_temp = pd.DataFrame({"hour": hours, "occ": binary})
        hourly_mean = df_temp.groupby("hour")["occ"].mean()
        ax.plot(
            hourly_mean.index, hourly_mean.values,
            marker="o", markersize=4, linewidth=1.5,
            color=_get_color(name), label=name,
        )

    ax.set_xlabel("Hour of Day", fontsize=11)
    ax.set_ylabel("Occupancy Probability", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(range(0, 24))
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if output_path:
        _save_fig(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# 7. Confusion matrix grid (actual vs predicted CO2 per detector)
# ---------------------------------------------------------------------------

def plot_confusion_matrix_grid(
    detections_actual: dict[str, np.ndarray],
    detections_predicted: dict[str, np.ndarray],
    title: str = "Actual vs Predicted CO2: Per-Detector Agreement",
    output_path: Path | None = None,
) -> plt.Figure:
    """Grid of 2x2 confusion matrices comparing each detector's output
    on actual CO2 vs predicted CO2.

    Args:
        detections_actual: Detector outputs on actual CO2.
        detections_predicted: Detector outputs on predicted CO2.
        title: Plot title.
        output_path: Path to save figure.
    """
    # Use only detectors present in both
    common_names = [n for n in detections_actual if n in detections_predicted]
    n_det = len(common_names)
    ncols = min(3, n_det)
    nrows = (n_det + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    if n_det == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, name in enumerate(common_names):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        actual = detections_actual[name]
        predicted = detections_predicted[name]
        n = min(len(actual), len(predicted))
        actual, predicted = actual[:n], predicted[:n]

        # Build 2x2 confusion matrix
        # Rows = actual CO2 detector, Cols = predicted CO2 detector
        cm = np.array([
            [(actual == 0) & (predicted == 0),
             (actual == 0) & (predicted == 1)],
            [(actual == 1) & (predicted == 0),
             (actual == 1) & (predicted == 1)],
        ])
        cm_counts = np.array([[c.sum() for c in row] for row in cm])

        im = ax.imshow(cm_counts, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm_counts[i, j]}", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if cm_counts[i, j] > cm_counts.max() * 0.5 else "black")

        agreement = float((actual == predicted).mean())
        ax.set_title(f"{name}\n(agree={agreement:.1%})", fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Unocc", "Occ"], fontsize=8)
        ax.set_yticklabels(["Unocc", "Occ"], fontsize=8)
        ax.set_xlabel("Predicted CO2", fontsize=8)
        ax.set_ylabel("Actual CO2", fontsize=8)

    # Hide unused axes
    for idx in range(n_det, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    if output_path:
        _save_fig(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# 8. CO2 with consensus shading
# ---------------------------------------------------------------------------

def plot_co2_with_consensus(
    co2: np.ndarray,
    consensus: np.ndarray,
    timestamps: np.ndarray | None = None,
    max_samples: int = 168 * 2,
    title: str = "CO2 with Majority-Vote Occupancy",
    output_path: Path | None = None,
) -> plt.Figure:
    """CO2 time series with background shading for occupied/unoccupied.

    Green shading = majority of detectors say occupied.
    No shading = majority say unoccupied.

    Args:
        co2: CO2 concentration array.
        consensus: Binary majority vote array.
        timestamps: Optional datetime array.
        max_samples: Maximum samples to display.
        title: Plot title.
        output_path: Path to save figure.
    """
    n = min(len(co2), max_samples)
    co2_plot = co2[:n]
    consensus_plot = consensus[:n]
    x = timestamps[:n] if timestamps is not None else np.arange(n)

    fig, ax = plt.subplots(figsize=(16, 4))

    # Background shading for occupied regions
    ax.fill_between(
        x, co2_plot.min() * 0.95, co2_plot.max() * 1.05,
        where=consensus_plot.astype(bool),
        alpha=0.15, color="#27ae60", label="Occupied (consensus)",
    )

    # CO2 line
    ax.plot(x, co2_plot, color="#2c3e50", linewidth=0.8, label="CO2")

    ax.set_ylabel("CO2 (ppm)", fontsize=11)
    ax.set_xlabel("Time step" if timestamps is None else "Time")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if output_path:
        _save_fig(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# 9. Summary statistics table (as a figure)
# ---------------------------------------------------------------------------

def plot_summary_table(
    consensus_actual: ConsensusResult,
    consensus_predicted: ConsensusResult | None = None,
    comparison_kappas: dict[str, float] | None = None,
    title: str = "Occupancy Detection Summary",
    output_path: Path | None = None,
) -> plt.Figure:
    """Render a summary statistics table as a matplotlib figure.

    Args:
        consensus_actual: ConsensusResult from actual CO2 detections.
        consensus_predicted: Optional ConsensusResult from predicted CO2.
        comparison_kappas: Optional dict of per-detector actual-vs-predicted kappa.
        title: Plot title.
        output_path: Path to save figure.
    """
    rows = []
    for name in consensus_actual.detector_names:
        row = [
            name,
            f"{consensus_actual.occupancy_rates[name]:.1%}",
        ]
        if consensus_predicted is not None:
            row.append(
                f"{consensus_predicted.occupancy_rates.get(name, 0):.1%}"
            )
        if comparison_kappas is not None:
            row.append(f"{comparison_kappas.get(name, 0):.3f}")
        rows.append(row)

    # Add consensus row
    consensus_row = [
        "CONSENSUS (majority)",
        f"{consensus_actual.majority_vote.mean():.1%}",
    ]
    if consensus_predicted is not None:
        consensus_row.append(
            f"{consensus_predicted.majority_vote.mean():.1%}"
        )
    if comparison_kappas is not None:
        consensus_row.append("—")
    rows.append(consensus_row)

    # Column headers
    headers = ["Detector", "Occ. Rate\n(Actual)"]
    if consensus_predicted is not None:
        headers.append("Occ. Rate\n(Predicted)")
    if comparison_kappas is not None:
        headers.append("Kappa\n(Act. vs Pred.)")

    n_cols = len(headers)
    fig, ax = plt.subplots(figsize=(3 * n_cols, 0.5 * (len(rows) + 1) + 1))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    # Style header row
    for j in range(n_cols):
        table[(0, j)].set_facecolor("#34495e")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Highlight consensus row
    last_row = len(rows)
    for j in range(n_cols):
        table[(last_row, j)].set_facecolor("#ecf0f1")
        table[(last_row, j)].set_text_props(fontweight="bold")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)

    fig.tight_layout()
    if output_path:
        _save_fig(fig, output_path)
    return fig
