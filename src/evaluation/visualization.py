"""Visualization utilities for model evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Use a clean style
plt.style.use("seaborn-v0_8-whitegrid")


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Path,
    dates: pd.Series | None = None,
    n_points: int = 500,
) -> None:
    """Plot predicted vs actual values over time.

    Args:
        y_true: Ground truth values (1D).
        y_pred: Predicted values (1D).
        model_name: Model name for the title.
        output_path: Path to save the plot.
        dates: Optional datetime series for x-axis.
        n_points: Max number of points to display.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = y_true.ravel()[:n_points]
    y_pred = y_pred.ravel()[:n_points]

    fig, ax = plt.subplots(figsize=(14, 5))

    if dates is not None and len(dates) >= n_points:
        x = dates.iloc[:n_points]
        ax.set_xlabel("Time")
    else:
        x = np.arange(len(y_true))
        ax.set_xlabel("Sample Index")

    ax.plot(x, y_true, label="Actual", color="#2196F3", linewidth=1.0, alpha=0.8)
    ax.plot(x, y_pred, label="Predicted", color="#FF5722", linewidth=1.0, alpha=0.8)
    ax.set_ylabel("CO2 (ppm)")
    ax.set_title(f"{model_name} - Predictions vs Actual")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """Scatter plot of predicted vs actual with perfect prediction line.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        model_name: Model name for the title.
        output_path: Path to save the plot.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.3, s=5, color="#2196F3")

    # Perfect prediction line
    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())
    ax.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=1.5, label="Perfect prediction")

    ax.set_xlabel("Actual CO2 (ppm)")
    ax.set_ylabel("Predicted CO2 (ppm)")
    ax.set_title(f"{model_name} - Scatter Plot")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """Plot residual distribution and residuals over time.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        model_name: Model name for the title.
        output_path: Path to save the plot.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    residuals = y_true.ravel() - y_pred.ravel()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(residuals, bins=50, color="#2196F3", alpha=0.7, edgecolor="white")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Residual (ppm)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"{model_name} - Residual Distribution")

    # Residuals over time
    axes[1].scatter(np.arange(len(residuals)), residuals, alpha=0.3, s=3, color="#FF5722")
    axes[1].axhline(0, color="black", linestyle="-", linewidth=0.8)
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Residual (ppm)")
    axes[1].set_title(f"{model_name} - Residuals Over Time")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    log_dir: Path,
    model_name: str,
    output_path: Path,
) -> None:
    """Plot training and validation loss curves from TensorBoard logs.

    Attempts to read TensorBoard event files. Falls back to a placeholder
    if the TensorBoard data is not available.

    Args:
        log_dir: Directory containing TensorBoard logs.
        model_name: Model name for the title.
        output_path: Path to save the plot.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        # Find the event file
        event_files = list(log_dir.rglob("events.out.tfevents.*"))
        if not event_files:
            print(f"  No TensorBoard event files found in {log_dir}")
            return

        ea = EventAccumulator(str(event_files[0].parent))
        ea.Reload()

        train_loss = [(s.step, s.value) for s in ea.Scalars("train_loss")]
        val_loss = [(s.step, s.value) for s in ea.Scalars("val_loss")]

        fig, ax = plt.subplots(figsize=(10, 5))
        if train_loss:
            steps, values = zip(*train_loss)
            ax.plot(steps, values, label="Train Loss", color="#2196F3")
        if val_loss:
            steps, values = zip(*val_loss)
            ax.plot(steps, values, label="Val Loss", color="#FF5722")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.set_title(f"{model_name} - Training Curves")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    except Exception as e:
        print(f"  Could not plot training curves: {e}")


def plot_predictions_with_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    model_name: str,
    output_path: Path,
    coverage: float = 0.9,
    n_points: int = 200,
) -> None:
    """Plot predictions with conformal prediction intervals.

    Shows point predictions with shaded confidence bands and ground truth.
    Uses the first forecast step (h=1) for the time-series view.

    Args:
        y_true: (n_samples, horizon) ground truth.
        y_pred: (n_samples, horizon) point predictions.
        lower: (n_samples, horizon) lower bounds.
        upper: (n_samples, horizon) upper bounds.
        model_name: Model name for the title.
        output_path: Path to save the plot.
        coverage: Target coverage level for the title.
        n_points: Max number of points to display.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use first forecast step for visualization
    y_t = y_true[:n_points, 0]
    y_p = y_pred[:n_points, 0]
    lo = lower[:n_points, 0]
    hi = upper[:n_points, 0]
    x = np.arange(len(y_t))

    fig, ax = plt.subplots(figsize=(14, 5))

    # Confidence band
    ax.fill_between(x, lo, hi, alpha=0.25, color="#FF9800",
                    label=f"{coverage*100:.0f}% prediction interval")

    ax.plot(x, y_t, label="Actual", color="#2196F3", linewidth=1.0, alpha=0.8)
    ax.plot(x, y_p, label="Predicted", color="#FF5722", linewidth=1.0, alpha=0.8)

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("CO2 (ppm)")
    ax.set_title(f"{model_name} - Predictions with {coverage*100:.0f}% Conformal Intervals")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
