"""Build and evaluate ensemble from pre-trained model predictions.

Loads saved predictions (predictions.npz) from multiple model result
directories, fits ensemble weights on validation data, and evaluates
on test data.

Usage:
    python scripts/run_ensemble.py --result-dirs results/LSTM_h1_* results/Seq2Seq_h1_* results/XGBoost_h1_*
    python scripts/run_ensemble.py --result-dirs results/LSTM_h1_* results/Seq2Seq_h1_* --method stacking
    python scripts/run_ensemble.py --result-dirs results/LSTM_h1_* results/Seq2Seq_h1_* --method both
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_metrics, save_metrics
from src.models.ensemble import WeightedAverageEnsemble, StackingEnsemble


def load_predictions(result_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load predictions from a model's result directory.

    Args:
        result_dir: Path to model result directory containing predictions.npz.

    Returns:
        Tuple of (y_true, y_pred) arrays.
    """
    npz_path = result_dir / "predictions.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"No predictions.npz found in {result_dir}")

    data = np.load(npz_path)
    return data["y_true"], data["y_pred"]


def resolve_dirs(patterns: list[str]) -> list[Path]:
    """Resolve glob patterns to concrete directory paths.

    Args:
        patterns: List of glob patterns or concrete paths.

    Returns:
        List of resolved Path objects.
    """
    dirs = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            dirs.extend(Path(m) for m in matches)
        else:
            dirs.append(Path(pattern))
    return [d for d in dirs if d.is_dir()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and evaluate model ensemble")
    parser.add_argument("--result-dirs", nargs="+", required=True,
                        help="Paths or glob patterns to model result directories")
    parser.add_argument("--method", type=str, default="both",
                        choices=["weighted", "stacking", "both"],
                        help="Ensemble method (default: both)")
    parser.add_argument("--cal-ratio", type=float, default=0.5,
                        help="Fraction of test set for validation/calibration (default: 0.5)")
    parser.add_argument("--output-dir", type=str, default="results/ensemble",
                        help="Output directory for ensemble results")
    args = parser.parse_args()

    result_dirs = resolve_dirs(args.result_dirs)
    if len(result_dirs) < 2:
        print(f"ERROR: Need at least 2 model directories, found {len(result_dirs)}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions from all models
    model_names = []
    all_y_true = None
    all_predictions = []

    for d in result_dirs:
        y_true, y_pred = load_predictions(d)
        model_names.append(d.name)
        all_predictions.append(y_pred)

        # Verify all models were evaluated on the same ground truth
        if all_y_true is None:
            all_y_true = y_true
        else:
            if not np.allclose(all_y_true, y_true, atol=1e-3):
                print(f"WARNING: Ground truth mismatch in {d.name}")

    assert all_y_true is not None
    n_samples = all_y_true.shape[0]

    print(f"\n{'='*60}")
    print(f"  Ensemble Builder")
    print(f"  Models: {', '.join(model_names)}")
    print(f"  Samples: {n_samples}")
    print(f"  Horizon: {all_y_true.shape[1]} steps")
    print(f"{'='*60}\n")

    # Split into calibration (for fitting ensemble) and evaluation sets
    n_cal = int(n_samples * args.cal_ratio)
    cal_true = all_y_true[:n_cal]
    eval_true = all_y_true[n_cal:]
    cal_preds = [p[:n_cal] for p in all_predictions]
    eval_preds = [p[n_cal:] for p in all_predictions]

    # Print individual model metrics on eval set for comparison
    print("Individual model metrics (evaluation set):")
    for name, pred in zip(model_names, eval_preds):
        metrics = compute_metrics(eval_true, pred)
        print(f"  {name}: RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.4f}")

    # Weighted average ensemble
    if args.method in ("weighted", "both"):
        print(f"\n--- Weighted Average Ensemble ---")
        wa = WeightedAverageEnsemble()
        wa.fit(cal_preds, cal_true, model_names=model_names)
        y_wa = wa.predict(eval_preds)
        wa_metrics = compute_metrics(eval_true, y_wa)

        print(f"  Weights: {dict(zip(model_names, [f'{w:.3f}' for w in wa.weights_]))}")
        print(f"  RMSE={wa_metrics['rmse']:.2f}, MAE={wa_metrics['mae']:.2f}, "
              f"R2={wa_metrics['r2']:.4f}, MAPE={wa_metrics['mape']:.2f}%")

        save_metrics(wa_metrics, "WeightedEnsemble",
                     output_dir / "weighted_metrics.json")
        wa.save(output_dir / "weighted_ensemble.json")
        np.savez(output_dir / "weighted_predictions.npz",
                 y_true=eval_true, y_pred=y_wa)

    # Stacking ensemble
    if args.method in ("stacking", "both"):
        print(f"\n--- Stacking Ensemble (Ridge) ---")
        stack = StackingEnsemble(alpha=1.0)
        stack.fit(cal_preds, cal_true, model_names=model_names)
        y_stack = stack.predict(eval_preds)
        stack_metrics = compute_metrics(eval_true, y_stack)

        print(f"  RMSE={stack_metrics['rmse']:.2f}, MAE={stack_metrics['mae']:.2f}, "
              f"R2={stack_metrics['r2']:.4f}, MAPE={stack_metrics['mape']:.2f}%")

        save_metrics(stack_metrics, "StackingEnsemble",
                     output_dir / "stacking_metrics.json")
        stack.save(output_dir / "stacking_ensemble.json")
        np.savez(output_dir / "stacking_predictions.npz",
                 y_true=eval_true, y_pred=y_stack)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
