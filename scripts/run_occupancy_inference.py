"""Run occupancy inference using 6 detection methods on actual and predicted CO2.

Loads processed data via the preprocessing pipeline, loads predicted CO2
from a trained model's predictions.npz, then runs all 6 occupancy detectors
on both actual and predicted CO2 signals. Evaluates cross-method consensus
and generates comparison visualizations.

Usage:
    python scripts/run_occupancy_inference.py \
        --data-config configs/experiments/preproc_E_occupancy_1h.yaml \
        --occupancy-config configs/occupancy.yaml \
        --predictions-dir results/preproc_E_LSTM_h1_20260221_232331 \
        --output-dir results/occupancy

    # Without predictions (actual CO2 only):
    python scripts/run_occupancy_inference.py \
        --data-config configs/experiments/preproc_E_occupancy_1h.yaml \
        --occupancy-config configs/occupancy.yaml \
        --output-dir results/occupancy_actual_only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import cohen_kappa_score

from src.data.pipeline import run_preprocessing_pipeline
from src.occupancy.detectors import run_all_detectors
from src.occupancy.evaluation import ConsensusEvaluator
from src.occupancy.visualization import (
    plot_actual_vs_predicted_occupancy,
    plot_agreement_heatmap,
    plot_co2_with_consensus,
    plot_confusion_matrix_grid,
    plot_consistency_over_time,
    plot_daily_occupancy_profile,
    plot_detector_comparison_timeline,
    plot_occupancy_rates_bar,
    plot_summary_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    """Load a YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(*configs: dict) -> dict:
    """Deep-merge multiple config dicts (later ones override earlier)."""
    result: dict = {}
    for cfg in configs:
        for key, value in cfg.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
    return result


def align_predictions_to_test_df(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    lookback: int,
    horizon: int = 1,
) -> np.ndarray:
    """Align model predictions to the test DataFrame indices.

    The sliding window ``create_sequences()`` produces sequences starting
    at index 0 of the test data. Prediction i corresponds to the target
    at test_df index ``i + lookback`` (for h=1, step 0).

    For h=1, we take the first forecast step from each prediction window,
    giving one prediction per timestep from index ``lookback`` to
    ``lookback + n_predictions - 1``.

    Args:
        test_df: Test DataFrame from the pipeline.
        y_pred: Predictions array of shape (n_predictions, horizon).
        lookback: Lookback window size in steps.
        horizon: Forecast horizon in steps.

    Returns:
        1D array of CO2 predictions aligned to test_df, with NaN for
        positions without predictions. Length = len(test_df).
    """
    n_test = len(test_df)
    n_pred = y_pred.shape[0]

    # Take only the first forecast step (h=1 slice)
    if y_pred.ndim == 2:
        y_pred_flat = y_pred[:, 0]
    else:
        y_pred_flat = y_pred

    aligned = np.full(n_test, np.nan, dtype=np.float64)

    # Prediction i maps to test_df row (i + lookback)
    start_idx = lookback
    end_idx = start_idx + n_pred

    # Clamp to valid range
    valid_end = min(end_idx, n_test)
    valid_n = valid_end - start_idx

    if valid_n > 0:
        aligned[start_idx:valid_end] = y_pred_flat[:valid_n]

    n_valid = int(np.sum(~np.isnan(aligned)))
    logger.info(
        f"  Aligned {n_valid}/{n_test} predictions "
        f"(offset={start_idx}, coverage={n_valid/n_test:.1%})"
    )
    return aligned


def build_detector_dataframe(
    test_df: pd.DataFrame,
    co2_values: np.ndarray,
    valid_mask: np.ndarray,
) -> pd.DataFrame:
    """Build a DataFrame for detector input with specified CO2 values.

    Replaces the CO2 column with provided values and filters to valid rows.

    Args:
        test_df: Original test DataFrame.
        co2_values: CO2 values to use (actual or predicted).
        valid_mask: Boolean mask indicating rows with valid CO2 values.

    Returns:
        DataFrame subset with valid rows and replaced CO2 column.
    """
    df = test_df.copy()
    df["CO2"] = co2_values
    df = df[valid_mask].reset_index(drop=True)

    # Recompute dCO2 for the (potentially modified) CO2 column
    df["dCO2"] = df["CO2"].diff().fillna(0.0)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run occupancy inference on actual and predicted CO2"
    )
    parser.add_argument(
        "--data-config", type=str, required=True,
        help="Path to experiment data config YAML"
    )
    parser.add_argument(
        "--occupancy-config", type=str, default="configs/occupancy.yaml",
        help="Path to occupancy detector config YAML"
    )
    parser.add_argument(
        "--predictions-dir", type=str, default=None,
        help="Path to model results dir with predictions.npz (optional)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/occupancy",
        help="Output directory for occupancy results"
    )
    parser.add_argument(
        "--lookback", type=int, default=None,
        help="Override lookback hours (for alignment)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Load configs ──────────────────────────────────────────────────
    data_config = load_config(Path(args.data_config))
    occ_config = load_config(Path(args.occupancy_config))

    # Merge with base configs for the pipeline
    base_data_cfg = load_config(PROJECT_ROOT / "configs" / "data.yaml")
    base_training_cfg = load_config(PROJECT_ROOT / "configs" / "training.yaml")
    full_config = merge_configs(base_data_cfg, base_training_cfg, data_config)

    detector_config = occ_config.get("detectors", occ_config)

    # Resolve lookback
    sph = full_config["data"].get("samples_per_hour", 1)
    lookback_hours = args.lookback or full_config["data"].get("lookback_hours", 24)
    lookback_steps = lookback_hours * sph

    print(f"\n{'='*60}")
    print(f"  Occupancy Inference Pipeline")
    print(f"  Data config: {args.data_config}")
    print(f"  Occupancy config: {args.occupancy_config}")
    print(f"  Predictions: {args.predictions_dir or 'None (actual only)'}")
    print(f"  Output: {output_dir}")
    print(f"  Lookback: {lookback_hours}h ({lookback_steps} steps)")
    print(f"{'='*60}\n")

    # ── Load data via pipeline ────────────────────────────────────────
    logger.info("Loading data via preprocessing pipeline...")
    raw_dir = Path(full_config["data"].get("raw_dir", "data/raw"))
    train_df, val_df, test_df = run_preprocessing_pipeline(
        raw_dir=raw_dir,
        variant_config=full_config,
    )

    logger.info(f"  Train: {len(train_df)} rows")
    logger.info(f"  Val:   {len(val_df)} rows")
    logger.info(f"  Test:  {len(test_df)} rows")
    logger.info(f"  Columns: {test_df.columns.tolist()}")

    # ── Load scaler for inverse-transforming predictions ──────────────
    scaler_path = None
    if args.predictions_dir:
        scaler_dir = Path(args.predictions_dir) / "scalers"
        if scaler_dir.exists():
            import joblib
            target_scaler_path = scaler_dir / "target_scaler.joblib"
            if target_scaler_path.exists():
                scaler_path = target_scaler_path
                logger.info(f"  Found target scaler: {scaler_path}")

    # ── Prepare actual CO2 test DataFrame ─────────────────────────────
    actual_co2 = test_df["CO2"].values.copy()
    has_predictions = False
    predicted_co2_aligned = None

    # ── Load and align predictions ────────────────────────────────────
    if args.predictions_dir:
        pred_dir = Path(args.predictions_dir)
        npz_path = pred_dir / "predictions.npz"

        if npz_path.exists():
            data = np.load(npz_path)
            y_pred_scaled = data["y_pred"]
            y_true_scaled = data["y_true"]

            logger.info(f"  Loaded predictions: y_pred={y_pred_scaled.shape}")

            # Detect whether predictions are already in original CO2 ppm units.
            # Training scripts typically inverse-transform before saving to npz.
            # If mean is in a plausible CO2 range (200-3000 ppm), skip inverse.
            pred_mean = float(np.mean(y_pred_scaled))
            if 200.0 < pred_mean < 3000.0:
                # Already in original ppm units
                y_pred_original = y_pred_scaled
                logger.info(
                    f"  Predictions appear to be in original units "
                    f"(mean={pred_mean:.1f} ppm) — skipping inverse transform"
                )
            elif scaler_path is not None:
                import joblib
                target_scaler = joblib.load(scaler_path)
                if y_pred_scaled.ndim == 1:
                    y_pred_original = target_scaler.inverse_transform(
                        y_pred_scaled.reshape(-1, 1)
                    ).flatten()
                else:
                    y_pred_original = np.column_stack([
                        target_scaler.inverse_transform(
                            y_pred_scaled[:, i].reshape(-1, 1)
                        ).flatten()
                        for i in range(y_pred_scaled.shape[1])
                    ])
                logger.info(
                    f"  Inverse-transformed predictions: "
                    f"range [{y_pred_original.min():.1f}, {y_pred_original.max():.1f}]"
                )
            else:
                y_pred_original = y_pred_scaled
                logger.warning(
                    "  No target scaler found — assuming predictions "
                    "are already in original units"
                )

            # Align predictions to test DataFrame indices
            horizon = y_pred_original.shape[1] if y_pred_original.ndim == 2 else 1
            predicted_co2_aligned = align_predictions_to_test_df(
                test_df, y_pred_original, lookback_steps, horizon
            )
            has_predictions = True
        else:
            logger.warning(f"  predictions.npz not found in {pred_dir}")

    # ── Build DataFrames for detectors ────────────────────────────────
    # For actual CO2: use all test rows
    df_actual = test_df.copy()
    # Ensure dCO2 exists
    if "dCO2" not in df_actual.columns:
        df_actual["dCO2"] = df_actual["CO2"].diff().fillna(0.0)

    # For predicted CO2: only use rows with valid predictions
    df_predicted = None
    if has_predictions and predicted_co2_aligned is not None:
        valid_mask = ~np.isnan(predicted_co2_aligned)
        df_predicted = build_detector_dataframe(
            test_df, predicted_co2_aligned, valid_mask
        )
        # Also subset actual for fair comparison
        df_actual_subset = build_detector_dataframe(
            test_df, actual_co2, valid_mask
        )
        logger.info(
            f"  Prediction-aligned subset: {len(df_predicted)} rows"
        )

    # ── Extract timestamps and hours for visualization ────────────────
    if "datetime" in test_df.columns:
        timestamps = pd.to_datetime(test_df["datetime"]).values
        hours = pd.to_datetime(test_df["datetime"]).dt.hour.values
    else:
        timestamps = None
        hours = np.zeros(len(test_df), dtype=int)

    # ── Run detectors on actual CO2 ───────────────────────────────────
    print("\n--- Running detectors on ACTUAL CO2 ---")
    results_actual = run_all_detectors(
        df_actual, detector_config, train_df=train_df
    )

    # ── Run detectors on predicted CO2 ────────────────────────────────
    results_predicted = None
    if df_predicted is not None:
        print("\n--- Running detectors on PREDICTED CO2 ---")
        results_predicted = run_all_detectors(
            df_predicted, detector_config, train_df=train_df
        )

    # ── Consensus evaluation ──────────────────────────────────────────
    evaluator = ConsensusEvaluator()

    print("\n--- Consensus Evaluation (Actual CO2) ---")
    consensus_actual = evaluator.evaluate(results_actual)
    consensus_actual.save(output_dir / "actual" / "consensus.json")

    # Save detections
    actual_dir = output_dir / "actual"
    actual_dir.mkdir(parents=True, exist_ok=True)
    np.savez(actual_dir / "detections.npz", **results_actual)

    consensus_predicted = None
    if results_predicted is not None:
        print("\n--- Consensus Evaluation (Predicted CO2) ---")
        consensus_predicted = evaluator.evaluate(results_predicted)
        pred_dir_out = output_dir / "predicted"
        pred_dir_out.mkdir(parents=True, exist_ok=True)
        consensus_predicted.save(pred_dir_out / "consensus.json")
        np.savez(pred_dir_out / "detections.npz", **results_predicted)

    # ── Actual vs Predicted comparison ────────────────────────────────
    comparison_kappas = {}
    if results_predicted is not None:
        print("\n--- Actual vs Predicted CO2 Comparison ---")
        comp_dir = output_dir / "comparison"
        comp_dir.mkdir(parents=True, exist_ok=True)

        # Use the subset of actual detections aligned to predictions
        results_actual_subset = run_all_detectors(
            df_actual_subset, detector_config, train_df=train_df
        )

        for name in results_actual_subset:
            if name in results_predicted:
                actual_arr = results_actual_subset[name]
                pred_arr = results_predicted[name]
                n = min(len(actual_arr), len(pred_arr))
                try:
                    kappa = cohen_kappa_score(actual_arr[:n], pred_arr[:n])
                    if np.isnan(kappa):
                        kappa = 0.0
                except Exception:
                    kappa = 0.0
                agreement = float(np.mean(actual_arr[:n] == pred_arr[:n]))
                comparison_kappas[name] = kappa
                print(
                    f"  {name:25s}: kappa={kappa:.4f}, "
                    f"agreement={agreement:.1%}"
                )

        comparison_data = {}
        for name, k in comparison_kappas.items():
            actual_arr = results_actual_subset[name]
            pred_arr = results_predicted[name]
            n_cmp = min(len(actual_arr), len(pred_arr))
            agree = float(np.mean(actual_arr[:n_cmp] == pred_arr[:n_cmp]))
            # Replace NaN with null for valid JSON
            kappa_val = None if (isinstance(k, float) and np.isnan(k)) else float(k)
            comparison_data[name] = {"kappa": kappa_val, "agreement": agree}
        with open(comp_dir / "actual_vs_predicted.json", "w", encoding="utf-8") as f:
            json.dump(comparison_data, f, indent=2)

    # ── Generate visualizations ───────────────────────────────────────
    print("\n--- Generating Visualizations ---")

    # 1. Detector timeline — actual CO2
    plot_detector_comparison_timeline(
        results_actual, actual_co2, timestamps,
        title="Occupancy Detection — Actual CO2",
        output_path=figures_dir / "fig1_detector_timeline_actual.png",
    )

    # 2. Detector timeline — predicted CO2 (if available)
    if results_predicted is not None and predicted_co2_aligned is not None:
        valid_mask = ~np.isnan(predicted_co2_aligned)
        plot_detector_comparison_timeline(
            results_predicted, predicted_co2_aligned[valid_mask],
            title="Occupancy Detection — Predicted CO2",
            output_path=figures_dir / "fig2_detector_timeline_predicted.png",
        )

    # 3. Occupancy rates bar chart
    all_rates = {"Actual: " + k: v for k, v in consensus_actual.occupancy_rates.items()}
    if consensus_predicted is not None:
        all_rates.update(
            {"Predicted: " + k: v for k, v in consensus_predicted.occupancy_rates.items()}
        )
    plot_occupancy_rates_bar(
        consensus_actual.occupancy_rates,
        title="Occupancy Rate by Detector (Actual CO2)",
        output_path=figures_dir / "fig3_occupancy_rates.png",
    )

    # 4. Agreement heatmap — Cohen's kappa
    plot_agreement_heatmap(
        consensus_actual.cohens_kappa_matrix,
        consensus_actual.detector_names,
        metric_name="Cohen's Kappa",
        title="Pairwise Cohen's Kappa (Actual CO2)",
        output_path=figures_dir / "fig4_agreement_heatmap_kappa.png",
    )

    # 4b. Agreement heatmap — raw agreement %
    plot_agreement_heatmap(
        consensus_actual.agreement_matrix,
        consensus_actual.detector_names,
        metric_name="Agreement %",
        title="Pairwise Agreement % (Actual CO2)",
        output_path=figures_dir / "fig4b_agreement_heatmap_pct.png",
    )

    # 5. Consistency over time
    plot_consistency_over_time(
        consensus_actual.consistency_scores, timestamps,
        title="Detector Consistency Over Time (Actual CO2)",
        output_path=figures_dir / "fig5_consistency.png",
    )

    # 6. Actual vs predicted occupancy comparison
    if results_predicted is not None:
        valid_mask = ~np.isnan(predicted_co2_aligned)  # type: ignore[arg-type]
        plot_actual_vs_predicted_occupancy(
            detections_actual=results_actual_subset,
            detections_predicted=results_predicted,
            co2_actual=actual_co2[valid_mask],
            co2_predicted=predicted_co2_aligned[valid_mask],  # type: ignore[index]
            title="Actual vs Predicted CO2: Occupancy Comparison",
            output_path=figures_dir / "fig6_actual_vs_predicted.png",
        )

    # 7. Daily occupancy profile
    plot_daily_occupancy_profile(
        results_actual, hours,
        title="Average Occupancy by Hour of Day",
        output_path=figures_dir / "fig7_daily_profile.png",
    )

    # 8. Confusion matrix grid (actual vs predicted per detector)
    if results_predicted is not None:
        plot_confusion_matrix_grid(
            results_actual_subset, results_predicted,
            title="Per-Detector Agreement: Actual vs Predicted CO2",
            output_path=figures_dir / "fig8_confusion_grid.png",
        )

    # 9. CO2 with consensus shading
    plot_co2_with_consensus(
        actual_co2, consensus_actual.majority_vote, timestamps,
        title="CO2 with Majority-Vote Occupancy (Actual)",
        output_path=figures_dir / "fig9_co2_with_consensus.png",
    )

    # 10. Summary table
    plot_summary_table(
        consensus_actual, consensus_predicted, comparison_kappas or None,
        title="Occupancy Detection Summary",
        output_path=figures_dir / "fig10_summary_table.png",
    )

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"\n  Fleiss' kappa (actual):    {consensus_actual.fleiss_kappa:.4f}")
    if consensus_predicted is not None:
        print(f"  Fleiss' kappa (predicted): {consensus_predicted.fleiss_kappa:.4f}")
    print(f"\n  Occupancy rates (actual CO2):")
    for name, rate in consensus_actual.occupancy_rates.items():
        print(f"    {name:25s}: {rate:.1%}")
    print(f"    {'CONSENSUS (majority)':25s}: {consensus_actual.majority_vote.mean():.1%}")

    if comparison_kappas:
        print(f"\n  Actual vs Predicted agreement:")
        for name, kappa in comparison_kappas.items():
            print(f"    {name:25s}: kappa={kappa:.4f}")

    print(f"\n  Mean consistency: {consensus_actual.consistency_scores.mean():.2%}")
    print(f"\n  Output: {output_dir}")
    print(f"  Figures: {figures_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
