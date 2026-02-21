"""Apply conformal prediction to calibrate uncertainty intervals.

Loads saved predictions from a model, splits into calibration and
evaluation sets, calibrates conformal intervals, and evaluates coverage.

Usage:
    python scripts/run_conformal.py --result-dir results/LSTM_h1_20260220_120000
    python scripts/run_conformal.py --result-dir results/ensemble --alpha 0.1
    python scripts/run_conformal.py --result-dir results/Seq2Seq_h24_* --alpha 0.05
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.conformal import SplitConformalPredictor
from src.evaluation.visualization import plot_predictions_with_intervals


def main() -> None:
    parser = argparse.ArgumentParser(description="Conformal prediction calibration")
    parser.add_argument("--result-dir", type=str, required=True,
                        help="Path to model result directory with predictions.npz")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Miscoverage rate (default: 0.1 for 90%% coverage)")
    parser.add_argument("--cal-ratio", type=float, default=0.5,
                        help="Fraction of data for calibration (default: 0.5)")
    args = parser.parse_args()

    # Resolve glob pattern if needed
    matches = glob.glob(args.result_dir)
    result_dir = Path(matches[0]) if matches else Path(args.result_dir)

    npz_path = result_dir / "predictions.npz"
    if not npz_path.exists():
        # Try common alternatives
        for alt in ["weighted_predictions.npz", "stacking_predictions.npz"]:
            if (result_dir / alt).exists():
                npz_path = result_dir / alt
                break
        else:
            print(f"ERROR: No predictions.npz found in {result_dir}")
            sys.exit(1)

    data = np.load(npz_path)
    y_true = data["y_true"]
    y_pred = data["y_pred"]

    # Ensure 2D: (n_samples, horizon)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    n_samples = y_true.shape[0]
    horizon = y_true.shape[1]
    n_cal = int(n_samples * args.cal_ratio)

    model_name = result_dir.name

    print(f"\n{'='*60}")
    print(f"  Conformal Prediction")
    print(f"  Model: {model_name}")
    print(f"  Samples: {n_samples} (cal={n_cal}, eval={n_samples - n_cal})")
    print(f"  Horizon: {horizon} steps")
    print(f"  Target coverage: {(1 - args.alpha)*100:.0f}%")
    print(f"{'='*60}\n")

    # Split into calibration and evaluation
    cal_true, eval_true = y_true[:n_cal], y_true[n_cal:]
    cal_pred, eval_pred = y_pred[:n_cal], y_pred[n_cal:]

    # Calibrate
    conformal = SplitConformalPredictor(alpha=args.alpha)
    conformal.calibrate(cal_true, cal_pred)

    # Evaluate
    coverage = conformal.evaluate_coverage(eval_true, eval_pred)

    print(f"  Target coverage:    {coverage['target_coverage']*100:.1f}%")
    print(f"  Empirical coverage: {coverage['empirical_coverage']*100:.1f}%")
    print(f"  Avg interval width: {coverage['avg_interval_width']:.2f} ppm")

    if horizon > 1:
        print(f"\n  Per-step coverage:")
        for t in range(min(horizon, 10)):
            step_cov = coverage['per_step_coverage'][t]
            step_width = coverage['per_step_width'][t]
            print(f"    Step {t+1}: coverage={step_cov*100:.1f}%, width={step_width:.2f} ppm")
        if horizon > 10:
            print(f"    ... ({horizon - 10} more steps)")

    # Save results
    output_dir = result_dir / "conformal"
    output_dir.mkdir(parents=True, exist_ok=True)

    conformal.save(output_dir / "calibration.json")
    with open(output_dir / "coverage.json", "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2)

    # Plot
    lower, upper = conformal.predict_intervals(eval_pred)
    plot_predictions_with_intervals(
        eval_true, eval_pred, lower, upper,
        model_name=model_name,
        output_path=output_dir / "intervals.png",
        coverage=1 - args.alpha,
    )

    print(f"\n  Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
