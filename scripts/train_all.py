"""Train all models sequentially and produce comparison report.

Runs all seven models (LSTM, CNN-LSTM, HMM-LSTM, TFT, SARIMA, XGBoost,
CatBoost) for both forecast horizons (1h, 24h), then generates comparison
charts and tables.

Usage:
    python scripts/train_all.py
    python scripts/train_all.py --horizon 1
    python scripts/train_all.py --epochs 50
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAINING_SCRIPTS = [
    "scripts/train_lstm.py",
    "scripts/train_cnn_lstm.py",
    "scripts/train_hmm_lstm.py",
    "scripts/train_tft.py",
    "scripts/train_sarima.py",
    "scripts/train_xgboost.py",
    "scripts/train_catboost.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all models")
    parser.add_argument("--horizon", type=int, nargs="+", default=[1, 24])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--experiment", type=str, default=None,
                        help="Path to experiment config YAML")
    args = parser.parse_args()

    horizon_args = []
    for h in args.horizon:
        horizon_args.extend(["--horizon", str(h)])

    extra_args = []
    if args.epochs:
        extra_args.extend(["--epochs", str(args.epochs)])
    if args.lookback:
        extra_args.extend(["--lookback", str(args.lookback)])
    if args.experiment:
        extra_args.extend(["--experiment", args.experiment])

    for script in TRAINING_SCRIPTS:
        script_path = PROJECT_ROOT / script
        print(f"\n{'='*70}")
        print(f"  RUNNING: {script}")
        print(f"{'='*70}\n")

        cmd = [sys.executable, str(script_path)] + horizon_args + extra_args
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

        if result.returncode != 0:
            print(f"\nWARNING: {script} exited with code {result.returncode}")
            print("Continuing with next model...\n")

    # Run comparison
    print(f"\n{'='*70}")
    print("  GENERATING COMPARISON REPORT")
    print(f"{'='*70}\n")

    compare_script = PROJECT_ROOT / "scripts" / "evaluate.py"
    subprocess.run([sys.executable, str(compare_script)], cwd=str(PROJECT_ROOT))


if __name__ == "__main__":
    main()
