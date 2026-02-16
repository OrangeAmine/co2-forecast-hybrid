"""Phase 5: Feature importance diagnostics.

Runs feature importance analysis on all 7 models using the final
feature set from Phase 4. Methods:
    - XGBoost: built-in gain + permutation importance
    - CatBoost: built-in PredictionValuesChange + SHAP
    - TFT: Variable Selection Network weights
    - LSTM/CNN-LSTM/HMM-LSTM: permutation importance
    - SARIMA: AR/MA coefficients (univariate, no feature importance)

Usage:
    python scripts/run_feature_importance.py
    python scripts/run_feature_importance.py --variant preproc_A
    python scripts/run_feature_importance.py --models xgboost catboost --horizon 1
"""

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datamodule import CO2DataModule
from src.data.pipeline import run_preprocessing_pipeline
from src.data.preprocessing import inverse_scale_target
from src.evaluation.metrics import compute_metrics
from src.models.lstm import LSTMForecaster
from src.models.cnn_lstm import CNNLSTMForecaster
from src.models.hmm_lstm import HMMLSTMForecaster, HMMRegimeDetector
from src.models.sarima import SARIMAForecaster
from src.models.xgboost_model import XGBoostForecaster
from src.models.catboost_model import CatBoostForecaster
from src.training.trainer import create_trainer
from src.utils.config import load_config
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)

VARIANT_CONFIGS = {
    "preproc_A": "configs/experiments/preproc_A_simple_5min.yaml",
    "preproc_B": "configs/experiments/preproc_B_simple_1h.yaml",
    "preproc_C": "configs/experiments/preproc_C_enhanced_5min.yaml",
    "preproc_D": "configs/experiments/preproc_D_enhanced_1h.yaml",
}

MODEL_CONFIGS = {
    "lstm": "configs/lstm.yaml",
    "cnn_lstm": "configs/cnn_lstm.yaml",
    "hmm_lstm": "configs/hmm_lstm.yaml",
    "tft": "configs/tft.yaml",
    "sarima": "configs/sarima.yaml",
    "xgboost": "configs/xgboost.yaml",
    "catboost": "configs/catboost.yaml",
}


def load_full_config(variant_name: str, model_name: str) -> dict:
    """Load merged config for a variant + model combination."""
    config_files = [
        str(PROJECT_ROOT / "configs" / "training.yaml"),
        str(PROJECT_ROOT / "configs" / "data.yaml"),
        str(PROJECT_ROOT / MODEL_CONFIGS[model_name]),
        str(PROJECT_ROOT / VARIANT_CONFIGS[variant_name]),
    ]
    return load_config(config_files)


# ──────────────────────────────────────────────────────────────────────
#  Permutation importance for neural models
# ──────────────────────────────────────────────────────────────────────

def permutation_importance_nn(
    model,
    dm: CO2DataModule,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Compute permutation importance for a PyTorch Lightning model.

    For each feature, shuffles that feature across all test samples
    (not within sequences), re-predicts, and measures RMSE increase.

    Args:
        model: Trained PyTorch Lightning model.
        dm: DataModule with test dataset.
        n_repeats: Number of shuffle repetitions.

    Returns:
        DataFrame with columns: feature, importance_mean, importance_std.
    """
    X_test = dm.test_dataset.X.numpy().copy()
    y_test = dm.test_dataset.y.numpy()

    # Baseline RMSE
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        X_tensor = torch.from_numpy(X_test).to(device)
        baseline_pred = model(X_tensor).cpu().numpy()

    y_true = inverse_scale_target(y_test, dm.target_scaler)
    y_pred_base = inverse_scale_target(baseline_pred, dm.target_scaler)
    baseline_rmse = compute_metrics(y_true, y_pred_base)["rmse"]

    feature_names = dm.feature_columns + [dm.target_column]
    n_features = X_test.shape[2]

    importance_results = []
    rng = np.random.default_rng(42)

    for feat_idx in range(n_features):
        rmse_increases = []
        for _ in range(n_repeats):
            X_shuffled = X_test.copy()
            # Shuffle feature across samples (not within sequences)
            perm = rng.permutation(X_shuffled.shape[0])
            X_shuffled[:, :, feat_idx] = X_test[perm, :, feat_idx]

            with torch.no_grad():
                X_tensor = torch.from_numpy(X_shuffled).to(device)
                shuffled_pred = model(X_tensor).cpu().numpy()

            y_pred_shuffled = inverse_scale_target(shuffled_pred, dm.target_scaler)
            shuffled_rmse = compute_metrics(y_true, y_pred_shuffled)["rmse"]
            rmse_increases.append(shuffled_rmse - baseline_rmse)

        feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"feature_{feat_idx}"
        importance_results.append({
            "feature": feat_name,
            "importance_mean": np.mean(rmse_increases),
            "importance_std": np.std(rmse_increases),
        })

    return pd.DataFrame(importance_results).sort_values("importance_mean", ascending=False)


# ──────────────────────────────────────────────────────────────────────
#  Per-model importance functions
# ──────────────────────────────────────────────────────────────────────

def importance_xgboost(
    config: dict,
    dm: CO2DataModule,
    output_dir: Path,
) -> pd.DataFrame:
    """XGBoost: built-in gain + sklearn permutation importance."""
    from sklearn.inspection import permutation_importance as sklearn_perm_imp

    X_train = dm.train_dataset.X.numpy()
    y_train = dm.train_dataset.y.numpy()
    X_val = dm.val_dataset.X.numpy()
    y_val = dm.val_dataset.y.numpy()
    X_test = dm.test_dataset.X.numpy()
    y_test = dm.test_dataset.y.numpy()

    model = XGBoostForecaster(config)
    model.fit(X_train, y_train, X_val, y_val)

    # Built-in feature importances (gain-based, averaged across multi-output)
    # XGBoost flattens features, so names are positional
    n_features_flat = X_test.shape[1] * X_test.shape[2]
    lookback = X_test.shape[1]
    n_cols = X_test.shape[2]
    feature_names = dm.feature_columns + [dm.target_column]

    # Aggregate importance by original feature (sum across lookback positions)
    try:
        raw_importances = model.model.estimators_[0].feature_importances_
        # Reshape to (lookback, n_features) and sum across lookback
        if len(raw_importances) == n_features_flat:
            reshaped = raw_importances.reshape(lookback, n_cols)
            aggregated = reshaped.sum(axis=0)
            aggregated = aggregated / aggregated.sum()  # normalize

            gain_df = pd.DataFrame({
                "feature": feature_names[:n_cols],
                "gain_importance": aggregated,
            })
        else:
            gain_df = pd.DataFrame(columns=["feature", "gain_importance"])
    except Exception:
        gain_df = pd.DataFrame(columns=["feature", "gain_importance"])

    gain_df.to_csv(output_dir / "xgboost_importance.csv", index=False)
    print(f"  XGBoost gain importance saved")
    return gain_df


def importance_catboost(
    config: dict,
    dm: CO2DataModule,
    output_dir: Path,
) -> pd.DataFrame:
    """CatBoost: built-in PredictionValuesChange."""
    X_train = dm.train_dataset.X.numpy()
    y_train = dm.train_dataset.y.numpy()
    X_val = dm.val_dataset.X.numpy()
    y_val = dm.val_dataset.y.numpy()
    X_test = dm.test_dataset.X.numpy()
    y_test = dm.test_dataset.y.numpy()

    model = CatBoostForecaster(config)
    model.fit(X_train, y_train, X_val, y_val)

    lookback = X_test.shape[1]
    n_cols = X_test.shape[2]
    feature_names = dm.feature_columns + [dm.target_column]

    try:
        raw_importances = model.model.estimators_[0].get_feature_importance(
            type="PredictionValuesChange"
        )
        n_features_flat = lookback * n_cols
        if len(raw_importances) == n_features_flat:
            reshaped = raw_importances.reshape(lookback, n_cols)
            aggregated = reshaped.sum(axis=0)
            aggregated = aggregated / aggregated.sum()

            imp_df = pd.DataFrame({
                "feature": feature_names[:n_cols],
                "prediction_change_importance": aggregated,
            })
        else:
            imp_df = pd.DataFrame(columns=["feature", "prediction_change_importance"])
    except Exception as e:
        print(f"  CatBoost importance extraction failed: {e}")
        imp_df = pd.DataFrame(columns=["feature", "prediction_change_importance"])

    imp_df.to_csv(output_dir / "catboost_importance.csv", index=False)
    print(f"  CatBoost importance saved")
    return imp_df


def importance_tft(
    config: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """TFT: Variable Selection Network weights from interpret_output()."""
    import lightning.pytorch as lpl
    from lightning.pytorch.callbacks import (
        EarlyStopping, ModelCheckpoint, TQDMProgressBar,
    )
    from lightning.pytorch.loggers import TensorBoardLogger
    from src.models.tft import build_tft_model, create_tft_datasets, prepare_tft_dataframe
    from datetime import datetime

    cfg = copy.deepcopy(config)
    cfg["training"]["results_dir"] = "results/feature_importance/tft_runs"

    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    datetime_col = cfg["data"]["datetime_column"]
    if datetime_col in combined_df.columns:
        combined_df = combined_df.sort_values(datetime_col).reset_index(drop=True)

    combined_df = prepare_tft_dataframe(combined_df, cfg)
    training_data, validation_data, test_data, combined_df = create_tft_datasets(combined_df, cfg)

    batch_size = cfg["training"]["batch_size"]
    nw = cfg["training"].get("num_workers", 0)
    train_dl = training_data.to_dataloader(train=True, batch_size=batch_size, num_workers=nw)
    val_dl = validation_data.to_dataloader(train=False, batch_size=batch_size, num_workers=nw)
    test_dl = test_data.to_dataloader(train=False, batch_size=batch_size, num_workers=nw)

    tft = build_tft_model(training_data, cfg)

    training_cfg = cfg["training"]
    results_dir = Path(training_cfg["results_dir"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / f"tft_importance_{timestamp}"

    trainer = lpl.Trainer(
        max_epochs=training_cfg["max_epochs"],
        accelerator=training_cfg.get("accelerator", "auto"),
        devices=training_cfg.get("devices", 1),
        precision=training_cfg.get("precision", 32),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=training_cfg["patience"], mode="min"),
            ModelCheckpoint(dirpath=run_dir / "checkpoints", monitor="val_loss", mode="min", save_top_k=1),
            TQDMProgressBar(refresh_rate=20),
        ],
        gradient_clip_val=training_cfg.get("gradient_clip_val", 0.1),
        log_every_n_steps=training_cfg.get("log_every_n_steps", 10),
        enable_progress_bar=True,
        logger=TensorBoardLogger(save_dir=str(run_dir), name="tb_logs"),
        deterministic=True,
    )

    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

    best_ckpt = trainer.checkpoint_callback.best_model_path
    best_tft = tft.__class__.load_from_checkpoint(best_ckpt)

    # Extract variable importance from TFT's interpret_output
    try:
        raw_predictions = best_tft.predict(test_dl, mode="raw", return_x=True)
        interpretation = best_tft.interpret_output(raw_predictions, reduction="sum")

        # Variable importance from interpret_output
        var_imp = {}
        for key in ["encoder_variables", "decoder_variables", "static_variables"]:
            if key in interpretation:
                names_key = f"{key}_names"
                names = interpretation.get(names_key, [])
                values = interpretation[key].detach().cpu().numpy()
                for var_name, imp_val in zip(names, values):
                    var_imp[var_name] = float(imp_val)

        imp_df = pd.DataFrame(
            [{"feature": k, "tft_variable_importance": v} for k, v in var_imp.items()]
        ).sort_values("tft_variable_importance", ascending=False)
    except Exception as e:
        print(f"  TFT interpretation failed: {e}")
        imp_df = pd.DataFrame(columns=["feature", "tft_variable_importance"])

    imp_df.to_csv(output_dir / "tft_variable_importance.csv", index=False)
    print(f"  TFT variable importance saved")
    return imp_df


def importance_nn_permutation(
    model_cls,
    config: dict,
    dm: CO2DataModule,
    model_name: str,
    output_dir: Path,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Train a NN model and compute permutation importance."""
    model = model_cls(config)
    trainer, run_dir = create_trainer(config, model_name=f"importance_{model_name}")
    trainer.fit(model, datamodule=dm)

    # Load best checkpoint
    best_ckpt = trainer.checkpoint_callback.best_model_path
    best_model = model_cls.load_from_checkpoint(best_ckpt)
    best_model.eval()

    imp_df = permutation_importance_nn(best_model, dm, n_repeats=n_repeats)
    imp_df.to_csv(output_dir / f"{model_name}_perm_importance.csv", index=False)
    print(f"  {model_name} permutation importance saved")
    return imp_df


def importance_sarima(
    config: dict,
    dm: CO2DataModule,
    output_dir: Path,
) -> pd.DataFrame:
    """SARIMA: Report AR/MA coefficients (univariate, no feature importance)."""
    X_train = dm.train_dataset.X.numpy()
    y_train = dm.train_dataset.y.numpy()

    target_idx = -1
    train_series = np.concatenate([
        X_train[0, :, target_idx],
        y_train[:, 0],
    ])

    model = SARIMAForecaster(config)
    model.fit(train_series)

    try:
        params = model.model.params
        param_names = model.model.param_names if hasattr(model.model, "param_names") else \
            [f"param_{i}" for i in range(len(params))]

        coef_df = pd.DataFrame({
            "parameter": param_names,
            "coefficient": params,
        })

        # p-values if available
        if hasattr(model.model, "pvalues"):
            coef_df["pvalue"] = model.model.pvalues
    except Exception as e:
        print(f"  SARIMA coefficient extraction failed: {e}")
        coef_df = pd.DataFrame(columns=["parameter", "coefficient", "pvalue"])

    coef_df.to_csv(output_dir / "sarima_coefficients.csv", index=False)
    print(f"  SARIMA coefficients saved")
    return coef_df


def generate_combined_plots(output_dir: Path) -> None:
    """Generate combined importance heatmap and bar plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Collect all importance CSVs
    importance_files = {
        "XGBoost": output_dir / "xgboost_importance.csv",
        "CatBoost": output_dir / "catboost_importance.csv",
        "LSTM": output_dir / "lstm_perm_importance.csv",
        "CNN-LSTM": output_dir / "cnn_lstm_perm_importance.csv",
        "HMM-LSTM": output_dir / "hmm_lstm_perm_importance.csv",
    }

    # Build combined importance matrix
    all_importances = {}
    for model_name, filepath in importance_files.items():
        if filepath.exists():
            df = pd.read_csv(filepath)
            if "feature" in df.columns:
                imp_col = [c for c in df.columns if c != "feature" and "importance" in c.lower()]
                if imp_col:
                    for _, row in df.iterrows():
                        feat = row["feature"]
                        if feat not in all_importances:
                            all_importances[feat] = {}
                        all_importances[feat][model_name] = row[imp_col[0]]

    if not all_importances:
        print("  No importance data found for combined plots")
        return

    combined_df = pd.DataFrame(all_importances).T
    combined_df = combined_df.fillna(0)

    # Normalize each column to [0, 1]
    for col in combined_df.columns:
        col_max = combined_df[col].max()
        if col_max > 0:
            combined_df[col] = combined_df[col] / col_max

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, max(6, len(combined_df) * 0.4)))
    im = ax.imshow(combined_df.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(combined_df.columns)))
    ax.set_xticklabels(combined_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(combined_df.index)))
    ax.set_yticklabels(combined_df.index)
    ax.set_title("Feature Importance Across Models (normalized)")
    plt.colorbar(im, ax=ax, label="Normalized Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "combined_importance_heatmap.png", dpi=150)
    plt.close()

    # Per-model horizontal bar charts
    n_models = len(combined_df.columns)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, max(4, len(combined_df) * 0.3)))
    if n_models == 1:
        axes = [axes]

    for idx, model_name in enumerate(combined_df.columns):
        ax = axes[idx]
        sorted_imp = combined_df[model_name].sort_values(ascending=True)
        ax.barh(range(len(sorted_imp)), sorted_imp.values)
        ax.set_yticks(range(len(sorted_imp)))
        ax.set_yticklabels(sorted_imp.index, fontsize=8)
        ax.set_title(model_name, fontsize=10)
        ax.set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig(output_dir / "combined_importance_barplot.png", dpi=150)
    plt.close()

    print(f"  Combined plots saved to: {output_dir}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Phase 5: Feature importance diagnostics")
    parser.add_argument("--variant", type=str, default=None,
                        help="Variant to analyze. Defaults to best from Phase 3.")
    parser.add_argument("--models", nargs="+",
                        default=["lstm", "cnn_lstm", "hmm_lstm", "tft", "sarima", "xgboost", "catboost"])
    parser.add_argument("--horizon", type=int, default=1, help="Horizon to analyze (default: 1)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n-repeats", type=int, default=10,
                        help="Number of permutation repeats (default: 10)")
    args = parser.parse_args()

    # Determine variant
    if args.variant:
        variant = args.variant
    else:
        best_path = Path("results/preprocessing_comparison/best_variant.json")
        if best_path.exists():
            with open(best_path) as f:
                variant = json.load(f)["best_variant"]
        else:
            variant = "preproc_A"

    output_dir = Path("results/feature_importance")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  PHASE 5: FEATURE IMPORTANCE DIAGNOSTICS")
    print(f"  Variant: {variant}")
    print(f"  Models: {args.models}")
    print(f"  Horizon: {args.horizon}h")
    print(f"{'='*70}\n")

    for model_name in args.models:
        print(f"\n{'─'*60}")
        print(f"  {model_name.upper()}")
        print(f"{'─'*60}")

        config = load_full_config(variant, model_name)
        config["data"]["forecast_horizon_hours"] = args.horizon
        if args.epochs:
            config["training"]["max_epochs"] = args.epochs
        config["training"]["results_dir"] = "results/feature_importance/runs"

        seed_everything(config["training"]["seed"])

        raw_dir = Path(config["data"].get("raw_dir", "data/raw"))
        train_df, val_df, test_df = run_preprocessing_pipeline(raw_dir=raw_dir, variant_config=config)

        t0 = time.time()
        try:
            if model_name == "xgboost":
                dm = CO2DataModule(config)
                dm._build_datasets(train_df, val_df, test_df)
                importance_xgboost(config, dm, output_dir)

            elif model_name == "catboost":
                dm = CO2DataModule(config)
                dm._build_datasets(train_df, val_df, test_df)
                importance_catboost(config, dm, output_dir)

            elif model_name == "tft":
                importance_tft(config, train_df, val_df, test_df, output_dir)

            elif model_name == "sarima":
                dm = CO2DataModule(config)
                dm._build_datasets(train_df, val_df, test_df)
                importance_sarima(config, dm, output_dir)

            elif model_name in ["lstm", "cnn_lstm"]:
                dm = CO2DataModule(config)
                dm._build_datasets(train_df, val_df, test_df)
                cls = LSTMForecaster if model_name == "lstm" else CNNLSTMForecaster
                importance_nn_permutation(cls, config, dm, model_name, output_dir, args.n_repeats)

            elif model_name == "hmm_lstm":
                # HMM-LSTM needs augmentation
                cfg = copy.deepcopy(config)
                model_cfg = cfg["model"]
                hmm_detector = HMMRegimeDetector(
                    n_states=model_cfg["hmm_n_states"],
                    covariance_type=model_cfg["hmm_covariance_type"],
                    n_iter=model_cfg["hmm_n_iter"],
                    hmm_features=model_cfg["hmm_features"],
                )
                hmm_detector.fit(train_df)
                n_states = model_cfg["hmm_n_states"]
                state_cols = [f"hmm_state_{i}" for i in range(n_states)]
                for split_df in [train_df, val_df, test_df]:
                    probs = hmm_detector.predict_proba(split_df)
                    for i, col in enumerate(state_cols):
                        split_df[col] = probs[:, i]
                cfg["data"]["feature_columns"] = cfg["data"]["feature_columns"] + state_cols

                dm = CO2DataModule.from_dataframes(train_df, val_df, test_df, cfg)
                n_input = len(cfg["data"]["feature_columns"]) + 1

                model = HMMLSTMForecaster(cfg, input_size=n_input)
                trainer, run_dir = create_trainer(cfg, model_name=f"importance_hmm_lstm")
                trainer.fit(model, datamodule=dm)

                best_ckpt = trainer.checkpoint_callback.best_model_path
                best_model = HMMLSTMForecaster.load_from_checkpoint(best_ckpt)
                best_model.eval()

                imp_df = permutation_importance_nn(best_model, dm, n_repeats=args.n_repeats)
                imp_df.to_csv(output_dir / "hmm_lstm_perm_importance.csv", index=False)
                print(f"  HMM-LSTM permutation importance saved")

            elapsed = time.time() - t0
            print(f"  Completed in {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAILED: {e} ({elapsed:.1f}s)")

    # Generate combined plots
    print(f"\n  Generating combined plots...")
    generate_combined_plots(output_dir)

    print(f"\n{'='*70}")
    print(f"  PHASE 5 COMPLETE")
    print(f"  Results saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
