"""Load all model results and produce comparison report.

Scans the results/ directory for metrics.json files from all trained
models and generates:
1. Per-horizon model comparison tables and bar charts
2. Cross-experiment comparison (grouped bar charts and summary)

Automatically parses experiment tags from model names (e.g.,
``exp1_LSTM_h1`` → experiment="exp1", model="LSTM", horizon="h1").
Legacy names without experiment prefix default to ``exp1``.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --results-dir results
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.comparison import (
    compare_experiments,
    compare_models,
    parse_model_name,
)
from src.evaluation.visualization import plot_predictions_vs_actual


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare all trained models")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Results directory",
    )
    args = parser.parse_args()

    results_dir = args.results_dir

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    #  Collect all metrics files, grouped by horizon AND experiment       #
    # ------------------------------------------------------------------ #
    # horizon_results: {horizon_key: {model_name: metrics}}
    horizon_results: dict[str, dict[str, dict]] = {}
    # experiment_results: {horizon_key: {experiment: {model_name: metrics}}}
    experiment_results: dict[str, dict[str, dict[str, dict]]] = {}

    for metrics_file in sorted(results_dir.rglob("metrics.json")):
        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        model_name = data["model"]
        exp_tag, base_model, horizon_key = parse_model_name(model_name)

        # Per-horizon flat grouping (for compare_models)
        if horizon_key not in horizon_results:
            horizon_results[horizon_key] = {}
        horizon_results[horizon_key][model_name] = data["metrics"]

        # Per-horizon + per-experiment grouping (for compare_experiments)
        if horizon_key not in experiment_results:
            experiment_results[horizon_key] = {}

        # Use experiment label from JSON if available, else tag from name
        exp_label = exp_tag
        if "experiment" in data and "name" in data["experiment"]:
            exp_label = data["experiment"]["name"]

        if exp_label not in experiment_results[horizon_key]:
            experiment_results[horizon_key][exp_label] = {}
        experiment_results[horizon_key][exp_label][model_name] = data["metrics"]

    if not horizon_results:
        print("No model results found. Train models first using train_all.py")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    #  Per-horizon model comparison                                       #
    # ------------------------------------------------------------------ #
    for horizon_key, results in sorted(horizon_results.items()):
        print(f"\n{'#'*70}")
        print(f"  COMPARISON: Forecast Horizon {horizon_key}")
        print(f"{'#'*70}")

        comparison_dir = results_dir / "comparison" / horizon_key
        compare_models(results, comparison_dir)

    # ------------------------------------------------------------------ #
    #  Cross-experiment comparison (per horizon)                          #
    # ------------------------------------------------------------------ #
    for horizon_key, exp_data in sorted(experiment_results.items()):
        if len(exp_data) < 2:
            print(f"\n  Skipping cross-experiment comparison for {horizon_key} "
                  f"(only {len(exp_data)} experiment(s) found)")
            continue

        print(f"\n{'#'*70}")
        print(f"  CROSS-EXPERIMENT COMPARISON: Horizon {horizon_key}")
        print(f"{'#'*70}")

        cross_exp_dir = results_dir / "comparison" / horizon_key / "cross_experiment"
        compare_experiments(exp_data, cross_exp_dir)

    # ------------------------------------------------------------------ #
    #  Overlay predictions plot (if predictions.npz files exist)          #
    # ------------------------------------------------------------------ #
    for horizon_key in horizon_results:
        all_preds: dict[str, dict] = {}
        for pred_file in results_dir.rglob("predictions.npz"):
            run_dir = pred_file.parent
            metrics_file = run_dir / "metrics.json"
            if not metrics_file.exists():
                continue
            with open(metrics_file) as f:
                model_name = json.load(f)["model"]

            _, _, pred_horizon = parse_model_name(model_name)
            if pred_horizon != horizon_key:
                continue

            data = np.load(pred_file)
            all_preds[model_name] = {
                "y_true": data["y_true"],
                "y_pred": data["y_pred"],
            }

        if len(all_preds) > 1:
            comparison_dir = results_dir / "comparison" / horizon_key

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(16, 6))
            n_points = 300
            colors = [
                "#2196F3", "#FF5722", "#4CAF50", "#FFC107", "#9C27B0",
                "#00BCD4", "#795548", "#607D8B",
            ]

            # Plot actual from first model
            first_model = list(all_preds.keys())[0]
            y_true = all_preds[first_model]["y_true"].ravel()[:n_points]
            ax.plot(y_true, label="Actual", color="black", linewidth=1.5, alpha=0.7)

            for i, (name, preds) in enumerate(all_preds.items()):
                y_pred = preds["y_pred"].ravel()[:n_points]
                ax.plot(
                    y_pred,
                    label=name,
                    color=colors[i % len(colors)],
                    linewidth=1.0,
                    alpha=0.8,
                )

            ax.set_xlabel("Sample Index")
            ax.set_ylabel("CO2 (ppm)")
            ax.set_title(
                f"All Models - Predictions vs Actual ({horizon_key})"
            )
            ax.legend(fontsize=8, ncol=2)
            plt.tight_layout()
            plt.savefig(
                comparison_dir / "overlay_predictions.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
            print(
                f"\nOverlay plot saved to: "
                f"{comparison_dir / 'overlay_predictions.png'}"
            )


if __name__ == "__main__":
    main()
