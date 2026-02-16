"""Cross-model and cross-experiment comparison utilities."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_model_results(results_dir: Path) -> dict[str, dict]:
    """Load all model results from the results directory.

    Scans for metrics.json files in subdirectories.

    Args:
        results_dir: Path to the results directory.

    Returns:
        Dict mapping model_name -> {"metrics": {...}, "run_dir": Path}.
    """
    results = {}
    for metrics_file in sorted(results_dir.rglob("metrics.json")):
        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        model_name = data["model"]
        results[model_name] = {
            "metrics": data["metrics"],
            "run_dir": metrics_file.parent,
        }
    return results


def parse_model_name(model_name: str) -> tuple[str, str, str]:
    """Parse experiment tag, base model name, and horizon from model name.

    Handles both legacy names (``LSTM_h1``) and experiment-tagged names
    (``exp1_LSTM_h1``, ``exp2_CNN-LSTM_h24``).

    Args:
        model_name: Full model name string.

    Returns:
        Tuple of (experiment_tag, base_model, horizon_key).
    """
    parts = model_name.split("_")
    exp_tag = ""
    horizon_key = "unknown"

    for part in parts:
        if part.startswith("exp"):
            exp_tag = part
        if part.startswith("h") and part[1:].isdigit():
            horizon_key = part

    # Extract base model name (remove exp prefix and horizon suffix)
    base_parts = [p for p in parts if p != exp_tag and p != horizon_key]
    base_model = "_".join(base_parts) if base_parts else model_name

    # Default experiment tag for legacy results
    if not exp_tag:
        exp_tag = "exp1"

    return exp_tag, base_model, horizon_key


def compare_models(
    results: dict[str, dict[str, float]],
    output_dir: Path,
) -> pd.DataFrame:
    """Generate comparison table and charts across all models.

    Args:
        results: Dict mapping model_name -> metrics dict with keys
            "mse", "rmse", "mae", "mape", "r2".
        output_dir: Directory to save comparison outputs.

    Returns:
        Comparison DataFrame.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build comparison table
    rows: list[dict[str, str | float]] = []
    for model_name, metrics in results.items():
        row: dict[str, str | float] = {"Model": model_name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index("Model")

    # Save as CSV
    csv_path = output_dir / "comparison_table.csv"
    df.to_csv(csv_path)
    print(f"\nComparison table saved to: {csv_path}")

    # Print table
    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"{'='*70}")

    # Bar chart for each metric
    metrics_to_plot = ["rmse", "mae", "mape", "r2"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#FFC107", "#9C27B0", "#00BCD4",
              "#795548", "#607D8B"]

    for i, metric in enumerate(metrics_to_plot):
        if metric not in df.columns:
            continue
        ax = axes[i]
        values = df[metric].values
        models = df.index.tolist()
        bars = ax.bar(models, values, color=colors[: len(models)], alpha=0.8)
        ax.set_title(metric.upper(), fontsize=14, fontweight="bold")
        ax.set_ylabel(metric.upper())

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Rotate x labels if needed
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle("Model Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_bar_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Rankings summary
    summary_path = output_dir / "comparison_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("MODEL COMPARISON SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        for metric in ["rmse", "mae", "mape"]:
            if metric in df.columns:
                best = df[metric].idxmin()
                f.write(f"Best {metric.upper()}: {best} ({df.loc[best, metric]:.4f})\n")

        if "r2" in df.columns:
            best = df["r2"].idxmax()
            f.write(f"Best R2: {best} ({df.loc[best, 'r2']:.4f})\n")

        f.write(f"\nFull table:\n{df.to_string(float_format=lambda x: f'{x:.4f}')}\n")

    print(f"Summary saved to: {summary_path}")

    return df


def compare_experiments(
    all_results: dict[str, dict[str, dict[str, float]]],
    output_dir: Path,
) -> pd.DataFrame:
    """Generate cross-experiment comparison tables and visualizations.

    Creates grouped bar charts showing each model's performance across
    experiments, and a heatmap table (model x experiment) per metric.

    Args:
        all_results: Nested dict of {experiment: {model_name: metrics}}.
        output_dir: Directory to save cross-experiment outputs.

    Returns:
        Wide-format DataFrame (rows=base_model, columns=metric_per_experiment).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a table: rows = (experiment, base_model), cols = metrics
    rows: list[dict[str, str | float]] = []
    for exp_name, models in all_results.items():
        for model_name, metrics in models.items():
            _, base_model, _ = parse_model_name(model_name)
            row: dict[str, str | float] = {
                "Experiment": exp_name,
                "Model": base_model,
                "FullName": model_name,
            }
            row.update(metrics)
            rows.append(row)

    if not rows:
        print("No results to compare across experiments.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Save full table
    csv_path = output_dir / "cross_experiment_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCross-experiment table saved to: {csv_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("CROSS-EXPERIMENT COMPARISON")
    print(f"{'='*70}")
    pivot = df.pivot_table(index="Model", columns="Experiment", values="rmse")
    print("\nRMSE by Model x Experiment:")
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    # Grouped bar chart for RMSE and R2
    experiments = sorted(df["Experiment"].unique())
    base_models = sorted(df["Model"].unique())

    for metric, title in [("rmse", "RMSE"), ("mae", "MAE"), ("r2", "R²")]:
        if metric not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(base_models))
        width = 0.8 / max(len(experiments), 1)
        exp_colors = ["#2196F3", "#FF5722", "#4CAF50", "#FFC107", "#9C27B0"]

        for i, exp in enumerate(experiments):
            exp_data = df[df["Experiment"] == exp]
            values = []
            for model in base_models:
                match = exp_data[exp_data["Model"] == model]
                # Use NaN for missing model-experiment combinations so
                # matplotlib skips them instead of plotting a misleading zero bar.
                values.append(float(match[metric].iloc[0]) if len(match) > 0 else np.nan)  # type: ignore[union-attr]

            offset = (i - len(experiments) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width,
                          label=exp, color=exp_colors[i % len(exp_colors)], alpha=0.85)

            # Value labels (skip NaN entries for missing model-experiment combos)
            for bar, val in zip(bars, values):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f"{title} - Cross Experiment Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(base_models, rotation=15)
        ax.legend(title="Experiment")
        plt.tight_layout()
        plt.savefig(output_dir / f"cross_experiment_{metric}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Summary: best experiment per model
    summary_path = output_dir / "cross_experiment_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("CROSS-EXPERIMENT COMPARISON SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        for model in base_models:
            model_rows = df[df["Model"] == model]
            if model_rows.empty:
                continue
            f.write(f"\n{model}:\n")
            for metric in ["rmse", "mae", "r2"]:
                if metric not in model_rows.columns:
                    continue
                if metric == "r2":
                    best_idx = model_rows[metric].idxmax()  # type: ignore[union-attr]
                else:
                    best_idx = model_rows[metric].idxmin()  # type: ignore[union-attr]
                best_row = model_rows.loc[best_idx]
                f.write(f"  Best {metric.upper()}: {best_row['Experiment']} "
                        f"({best_row[metric]:.4f})\n")

    print(f"Summary saved to: {summary_path}")

    return df
