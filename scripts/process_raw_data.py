"""Run the split-aware preprocessing pipeline.

Processes raw XLS sensor files through the new split-aware pipeline,
producing split DataFrames ready for model training. Outputs a
validation log (rows per split, NaN counts, etc.) instead of a
monolithic CSV.

Usage:
    python scripts/process_raw_data.py                              # Default: simple 5-min
    python scripts/process_raw_data.py --variant preproc_A          # Variant A (simple 5-min)
    python scripts/process_raw_data.py --variant preproc_C          # Variant C (enhanced 5-min)
    python scripts/process_raw_data.py --all                        # All 4 variants
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline import run_preprocessing_pipeline

VARIANT_CONFIGS = {
    "preproc_A": "configs/experiments/preproc_A_simple_5min.yaml",
    "preproc_B": "configs/experiments/preproc_B_simple_1h.yaml",
    "preproc_C": "configs/experiments/preproc_C_enhanced_5min.yaml",
    "preproc_D": "configs/experiments/preproc_D_enhanced_1h.yaml",
}


def load_variant_config(variant_name: str) -> dict:
    """Load and merge base configs with variant experiment config.

    Args:
        variant_name: Variant key (preproc_A, preproc_B, etc.).

    Returns:
        Merged config dictionary.
    """
    from src.utils.config import load_config

    config_path = VARIANT_CONFIGS[variant_name]
    config_files = [
        str(PROJECT_ROOT / "configs" / "training.yaml"),
        str(PROJECT_ROOT / "configs" / "data.yaml"),
        str(PROJECT_ROOT / config_path),
    ]
    return load_config(config_files)


def process_variant(variant_name: str, raw_dir: Path) -> None:
    """Process a single preprocessing variant.

    Args:
        variant_name: Variant key (preproc_A, preproc_B, etc.).
        raw_dir: Path to raw XLS data directory.
    """
    config = load_variant_config(variant_name)

    label = config.get("experiment", {}).get("label", variant_name)
    print(f"\n{'#' * 70}")
    print(f"  Processing variant: {variant_name} ({label})")
    print(f"{'#' * 70}\n")

    train_df, val_df, test_df = run_preprocessing_pipeline(
        raw_dir=raw_dir,
        variant_config=config,
    )

    # Validation summary
    print(f"\n  {'='*60}")
    print(f"  VALIDATION SUMMARY: {variant_name}")
    print(f"  {'='*60}")
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"  {name:>5}: {len(df):>7} rows  |  "
              f"NaN: {df.isna().any(axis=1).sum()}  |  "
              f"Columns: {len(df.columns)}")
    print(f"  Columns: {train_df.columns.tolist()}")
    print(f"  {'='*60}\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run split-aware preprocessing pipeline"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directory containing raw XLS files",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="preproc_A",
        choices=list(VARIANT_CONFIGS.keys()),
        help=f"Preprocessing variant (default: preproc_A)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all 4 preprocessing variants",
    )
    args = parser.parse_args()

    if args.all:
        for variant_name in VARIANT_CONFIGS:
            process_variant(variant_name, args.raw_dir)
    else:
        process_variant(args.variant, args.raw_dir)

    print("\nAll processing complete!")


if __name__ == "__main__":
    main()
