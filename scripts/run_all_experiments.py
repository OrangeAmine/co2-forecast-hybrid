"""Run all experiments: train all models under all experiment configurations.

Sequentially runs train_all.py for each experiment config, then
generates the cross-experiment comparison report.

Usage:
    python scripts/run_all_experiments.py
    python scripts/run_all_experiments.py --experiments preproc_A preproc_B
    python scripts/run_all_experiments.py --experiments preproc_C --horizon 1
    python scripts/run_all_experiments.py --epochs 50
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPERIMENT_CONFIGS: dict[str, str] = {
    "preproc_A": "configs/experiments/preproc_A_simple_5min.yaml",
    "preproc_B": "configs/experiments/preproc_B_simple_1h.yaml",
    "preproc_C": "configs/experiments/preproc_C_enhanced_5min.yaml",
    "preproc_D": "configs/experiments/preproc_D_enhanced_1h.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=list(EXPERIMENT_CONFIGS.keys()),
        help=f"Experiments to run (default: all). Choices: {list(EXPERIMENT_CONFIGS.keys())}",
    )
    parser.add_argument("--horizon", type=int, nargs="+", default=[1, 24])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lookback", type=int, default=None)
    args = parser.parse_args()

    # Build common args
    horizon_args: list[str] = []
    for h in args.horizon:
        horizon_args.extend(["--horizon", str(h)])

    extra_args: list[str] = []
    if args.epochs:
        extra_args.extend(["--epochs", str(args.epochs)])
    if args.lookback:
        extra_args.extend(["--lookback", str(args.lookback)])

    train_all_script = PROJECT_ROOT / "scripts" / "train_all.py"

    for exp_name in args.experiments:
        exp_config = EXPERIMENT_CONFIGS.get(exp_name)
        if exp_config is None:
            print(f"\nWARNING: Unknown experiment '{exp_name}'. Skipping.")
            continue

        exp_path = str(PROJECT_ROOT / exp_config)

        print(f"\n{'#' * 70}")
        print(f"  EXPERIMENT: {exp_name} ({exp_config})")
        print(f"{'#' * 70}\n")

        cmd = (
            [sys.executable, str(train_all_script)]
            + horizon_args
            + ["--experiment", exp_path]
            + extra_args
        )
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

        if result.returncode != 0:
            print(f"\nWARNING: Experiment {exp_name} had failures (exit code {result.returncode})")
            print("Continuing with next experiment...\n")

    # Final cross-experiment comparison
    print(f"\n{'#' * 70}")
    print("  GENERATING CROSS-EXPERIMENT COMPARISON")
    print(f"{'#' * 70}\n")

    evaluate_script = PROJECT_ROOT / "scripts" / "evaluate.py"
    subprocess.run([sys.executable, str(evaluate_script)], cwd=str(PROJECT_ROOT))

    print("\nAll experiments complete!")


if __name__ == "__main__":
    main()
