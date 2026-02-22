"""TFT Interpretability Study: Deep analysis of Temporal Fusion Transformer internals.

Performs comprehensive interpretability analysis on preproc_D (Enhanced 1h) data:
  A) Gate dynamics analysis (GLU gates in GRN layers, analogous to LSTM gates)
  B) Gradient-based feature attribution (input saliency maps)
  C) Hidden state structural analysis (PCA, clustering on LSTM states within TFT)
  D) Temporal pattern analysis (FFT, autocorrelation on attention/hidden states)
  E) TFT-specific interpretability (VSN weights, multi-head attention, predictions)

Usage:
    python -u scripts/run_tft_interpretability.py
    python -u scripts/run_tft_interpretability.py --horizons 1
    python -u scripts/run_tft_interpretability.py --horizons 1 24 --epochs 30
"""

import argparse
import copy
import gc
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.gridspec import GridSpec
from scipy.fft import fft, fftfreq
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import lightning.pytorch as lpl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger

from src.data.pipeline import run_preprocessing_pipeline
from src.evaluation.metrics import compute_metrics, save_metrics
from src.models.tft import (
    build_tft_model,
    create_tft_datasets,
    prepare_tft_dataframe,
)
from src.utils.config import load_config
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-whitegrid")

RESULTS_BASE = Path("results/tft_interpretability")

# Color palette
C_PRIMARY = "#2196F3"
C_SECONDARY = "#FF5722"
C_ACCENT = "#4CAF50"
C_WARN = "#FFC107"
C_NEUTRAL = "#607D8B"


# ======================================================================
#  Configuration
# ======================================================================

def load_interpretability_config(
    horizon: int,
    epochs_override: int | None = None,
) -> dict:
    """Load merged config for TFT + preproc_D + specified horizon."""
    config_files = [
        str(PROJECT_ROOT / "configs" / "training.yaml"),
        str(PROJECT_ROOT / "configs" / "data.yaml"),
        str(PROJECT_ROOT / "configs" / "tft.yaml"),
        str(PROJECT_ROOT / "configs" / "experiments" / "preproc_D_enhanced_1h.yaml"),
    ]
    config = load_config(config_files)
    config["data"]["forecast_horizon_hours"] = horizon
    if epochs_override is not None:
        config["training"]["max_epochs"] = epochs_override
    config["training"]["results_dir"] = str(
        RESULTS_BASE / f"h{horizon}" / "training_runs"
    )
    return config


# ======================================================================
#  Hook Manager for TFT Internal State Extraction
# ======================================================================

class TFTHookManager:
    """Register forward hooks to extract TFT internal activations.

    Captures:
      - LSTM encoder/decoder hidden states and outputs
      - GLU gate outputs from all GateAddNorm layers
      - Post-attention enrichment representations
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.captures: dict[str, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._register_hooks(model)

    def _register_hooks(self, model: torch.nn.Module) -> None:
        # LSTM encoder: captures (output, (h_n, c_n))
        if hasattr(model, "lstm_encoder"):
            self._hooks.append(
                model.lstm_encoder.register_forward_hook(self._lstm_hook("lstm_encoder"))
            )
        if hasattr(model, "lstm_decoder"):
            self._hooks.append(
                model.lstm_decoder.register_forward_hook(self._lstm_hook("lstm_decoder"))
            )

        # GLU gates inside GateAddNorm layers
        for name, module in model.named_modules():
            cls_name = module.__class__.__name__
            if cls_name == "GatedLinearUnit":
                self._hooks.append(
                    module.register_forward_hook(self._tensor_hook(f"glu_{name}"))
                )
            elif cls_name == "GatedResidualNetwork" and any(
                key in name
                for key in [
                    "static_enrichment",
                    "pos_wise_ff",
                    "static_context_enrichment",
                ]
            ):
                self._hooks.append(
                    module.register_forward_hook(self._tensor_hook(f"grn_{name}"))
                )

    def _lstm_hook(self, name: str):
        def hook(module, inp, output):
            # LSTM returns (output_seq, (h_n, c_n))
            self.captures[f"{name}_output"] = output[0].detach()
            self.captures[f"{name}_hidden"] = output[1][0].detach()
            self.captures[f"{name}_cell"] = output[1][1].detach()
        return hook

    def _tensor_hook(self, name: str):
        def hook(module, inp, output):
            if isinstance(output, torch.Tensor):
                self.captures[name] = output.detach()
            elif isinstance(output, (tuple, list)):
                self.captures[name] = output[0].detach()
        return hook

    def clear(self) -> None:
        self.captures.clear()

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ======================================================================
#  TFT Training and Extraction
# ======================================================================

def train_tft(
    config: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
) -> tuple:
    """Train TFT and return best model + test dataloader + raw predictions.

    Returns:
        (best_tft, test_dl, raw_predictions, training_data)
    """
    cfg = copy.deepcopy(config)

    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    datetime_col = cfg["data"]["datetime_column"]
    if datetime_col in combined_df.columns:
        combined_df = combined_df.sort_values(datetime_col).reset_index(drop=True)

    combined_df = prepare_tft_dataframe(combined_df, cfg)
    training_data, validation_data, test_data, combined_df = create_tft_datasets(
        combined_df, cfg
    )

    batch_size = cfg["training"]["batch_size"]
    nw = cfg["training"].get("num_workers", 0)
    train_dl = training_data.to_dataloader(
        train=True, batch_size=batch_size, num_workers=nw
    )
    val_dl = validation_data.to_dataloader(
        train=False, batch_size=batch_size, num_workers=nw
    )
    test_dl = test_data.to_dataloader(
        train=False, batch_size=batch_size, num_workers=nw
    )

    tft = build_tft_model(training_data, cfg)

    training_cfg = cfg["training"]
    results_dir = Path(training_cfg["results_dir"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / f"tft_interp_h{horizon}_{timestamp}"

    trainer = lpl.Trainer(
        max_epochs=training_cfg["max_epochs"],
        accelerator=training_cfg.get("accelerator", "auto"),
        devices=training_cfg.get("devices", 1),
        precision=training_cfg.get("precision", 32),
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=training_cfg["patience"],
                mode="min",
            ),
            ModelCheckpoint(
                dirpath=run_dir / "checkpoints",
                filename="best-{epoch}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            ),
            LearningRateMonitor(logging_interval="epoch"),
            TQDMProgressBar(refresh_rate=20),
        ],
        gradient_clip_val=training_cfg.get("gradient_clip_val", 0.1),
        log_every_n_steps=training_cfg.get("log_every_n_steps", 10),
        enable_progress_bar=True,
        logger=TensorBoardLogger(save_dir=str(run_dir), name="tb_logs"),
        deterministic="warn",  # CRITICAL: True crashes TFT on CUDA
    )

    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

    best_ckpt = trainer.checkpoint_callback.best_model_path  # type: ignore[union-attr]
    best_tft = tft.__class__.load_from_checkpoint(best_ckpt)

    raw_predictions = best_tft.predict(test_dl, mode="raw", return_x=True)

    return best_tft, test_dl, raw_predictions, training_data


def extract_predictions(best_tft, test_dl) -> dict[str, np.ndarray]:
    """Extract actual and predicted values from test set."""
    predictions = best_tft.predict(test_dl, mode="prediction", return_x=False)
    y_pred = torch.cat([p for p in predictions], dim=0).numpy()
    actuals = torch.cat([y[0] for x, y in iter(test_dl)], dim=0).numpy()
    return {"y_true": actuals.ravel(), "y_pred": y_pred.ravel()}


# ======================================================================
#  Section A: Gate Dynamics Analysis
# ======================================================================

def run_gate_dynamics(
    best_tft,
    test_dl,
    horizon: int,
    output_dir: Path,
) -> None:
    """Extract and visualize GLU gate activations from GRN layers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("  [A] Extracting gate dynamics via forward hooks...")

    hook_mgr = TFTHookManager(best_tft)
    best_tft.eval()

    # Collect gate activations across test batches (limit to avoid OOM)
    all_gate_stats: dict[str, list[np.ndarray]] = {}
    max_batches = 20

    with torch.no_grad():
        for i, (x, y) in enumerate(test_dl):
            if i >= max_batches:
                break
            best_tft(x)
            for key, tensor in hook_mgr.captures.items():
                if "glu_" in key:
                    arr = tensor.cpu().numpy()
                    if key not in all_gate_stats:
                        all_gate_stats[key] = []
                    all_gate_stats[key].append(arr.reshape(-1, arr.shape[-1]))
            hook_mgr.clear()

    hook_mgr.remove_hooks()

    if not all_gate_stats:
        print("  [A] No GLU gates captured. Skipping gate dynamics.")
        return

    # Consolidate
    gate_data = {}
    for key, arrays in all_gate_stats.items():
        gate_data[key] = np.concatenate(arrays, axis=0)

    # Plot 1: Gate activation distributions (histogram per layer)
    gate_names = sorted(gate_data.keys())
    n_gates = len(gate_names)
    n_cols = min(3, n_gates)
    n_rows = (n_gates + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = np.array(axes).ravel() if n_gates > 1 else [axes]

    for idx, gname in enumerate(gate_names):
        ax = axes_flat[idx]
        vals = gate_data[gname].ravel()
        # Subsample for performance
        if len(vals) > 100000:
            vals = np.random.default_rng(42).choice(vals, 100000, replace=False)
        ax.hist(vals, bins=80, color=C_PRIMARY, alpha=0.7, edgecolor="white", density=True)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        short_name = gname.replace("glu_", "").split(".")[-2] if "." in gname else gname
        ax.set_title(short_name, fontsize=9)
        ax.set_xlabel("Activation")
        ax.set_ylabel("Density")

    for idx in range(n_gates, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"GLU Gate Activation Distributions - {horizon}h", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / f"gate_distributions_h{horizon}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: Gate activation statistics table
    fig, ax = plt.subplots(figsize=(10, max(3, 0.4 * n_gates)))
    ax.axis("off")

    rows = []
    for gname in gate_names:
        vals = gate_data[gname].ravel()
        sparsity = (np.abs(vals) < 0.01).mean() * 100
        short_name = gname.replace("glu_", "").split(".")[-2] if "." in gname else gname
        rows.append([
            short_name,
            f"{vals.mean():.4f}",
            f"{vals.std():.4f}",
            f"{vals.min():.4f}",
            f"{vals.max():.4f}",
            f"{sparsity:.1f}%",
        ])

    table = ax.table(
        cellText=rows,
        colLabels=["Gate Layer", "Mean", "Std", "Min", "Max", "Sparsity (<0.01)"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    ax.set_title(f"GLU Gate Statistics - {horizon}h", fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / f"gate_statistics_h{horizon}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: Gate activation heatmap for a key layer (post_lstm or pos_wise_ff)
    # Pick the layer with the most captures
    largest_key = max(gate_data.keys(), key=lambda k: gate_data[k].shape[0])
    heatmap_data = gate_data[largest_key]
    # Take mean across samples, show first 64 hidden dims x available timesteps
    if heatmap_data.ndim >= 2:
        n_show = min(64, heatmap_data.shape[1])
        # Use first 200 samples
        subset = heatmap_data[:min(200, len(heatmap_data)), :n_show]
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(subset.T, aspect="auto", cmap="RdBu_r", interpolation="nearest")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Hidden Dimension")
        short_name = largest_key.replace("glu_", "").split(".")[-2] if "." in largest_key else largest_key
        ax.set_title(f"Gate Activations: {short_name} - {horizon}h")
        plt.colorbar(im, ax=ax, label="Activation")
        plt.tight_layout()
        plt.savefig(
            output_dir / f"gate_heatmap_h{horizon}.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    print(f"  [A] Gate dynamics: {n_gates} layers analyzed, plots saved")


# ======================================================================
#  Section B: Gradient-Based Feature Attribution
# ======================================================================

def run_gradient_attribution(
    best_tft,
    test_dl,
    config: dict,
    horizon: int,
    output_dir: Path,
) -> None:
    """Compute input gradients to identify important features and timesteps."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("  [B] Computing gradient-based feature attribution...")

    best_tft.eval()
    device = next(best_tft.parameters()).device

    all_grads = []
    max_batches = 30

    for i, (x, y) in enumerate(test_dl):
        if i >= max_batches:
            break

        # x is a dict with keys like 'encoder_cont', 'decoder_cont', etc.
        # Enable gradients on encoder continuous inputs
        if "encoder_cont" not in x:
            print("  [B] No encoder_cont in batch. Skipping gradient attribution.")
            return

        encoder_cont = x["encoder_cont"].clone().detach().to(device).requires_grad_(True)
        x_modified = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
        x_modified["encoder_cont"] = encoder_cont

        out = best_tft(x_modified)
        pred = out["prediction"]

        # Use mean prediction as scalar target for backward
        target = pred.mean()
        target.backward()

        if encoder_cont.grad is not None:
            # |grad| shape: (batch, encoder_len, n_features)
            grad_abs = encoder_cont.grad.abs().detach().cpu().numpy()
            all_grads.append(grad_abs)

        best_tft.zero_grad()

    if not all_grads:
        print("  [B] No gradients collected. Skipping.")
        return

    # Mean across all samples: (encoder_len, n_features)
    grads = np.concatenate(all_grads, axis=0)
    avg_grad = grads.mean(axis=0)

    # Feature names from config
    model_cfg = config["model"]
    unknown_reals = model_cfg.get("time_varying_unknown_reals", [])
    known_reals = model_cfg.get("time_varying_known_reals", [])
    # Encoder continuous features = unknown_reals + known_reals (pytorch-forecasting order)
    feature_names = unknown_reals + known_reals
    if len(feature_names) != avg_grad.shape[1]:
        feature_names = [f"feat_{i}" for i in range(avg_grad.shape[1])]

    # Plot 4: Gradient attribution heatmap (timesteps x features)
    fig, ax = plt.subplots(figsize=(max(8, len(feature_names) * 0.5), 6))
    im = ax.imshow(avg_grad.T, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Encoder Timestep (hours ago)")
    ax.set_ylabel("Feature")
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=8)
    lookback = avg_grad.shape[0]
    tick_positions = np.linspace(0, lookback - 1, min(6, lookback)).astype(int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"-{lookback - t}" for t in tick_positions])
    ax.set_title(f"Gradient Attribution Heatmap - {horizon}h Horizon")
    plt.colorbar(im, ax=ax, label="|Gradient|")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"gradient_heatmap_h{horizon}.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # Plot 5: Per-feature gradient magnitude (bar chart)
    feat_importance = avg_grad.mean(axis=0)
    feat_importance_pct = feat_importance / feat_importance.sum() * 100
    sort_idx = np.argsort(feat_importance_pct)

    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.35)))
    ax.barh(range(len(sort_idx)), feat_importance_pct[sort_idx], color=C_ACCENT)
    ax.set_yticks(range(len(sort_idx)))
    ax.set_yticklabels([feature_names[i] for i in sort_idx], fontsize=9)
    ax.set_xlabel("Gradient Importance (%)")
    ax.set_title(f"Feature Attribution (Gradient) - {horizon}h")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / f"gradient_feature_ranking_h{horizon}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Plot 6: Temporal gradient profile (which lookback timesteps matter most)
    temporal_importance = avg_grad.sum(axis=1)
    temporal_importance_norm = temporal_importance / temporal_importance.sum()
    time_idx = np.arange(-lookback, 0)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(time_idx, temporal_importance_norm, color=C_PRIMARY, alpha=0.8)
    ax.set_xlabel("Hours Ago")
    ax.set_ylabel("Normalized Gradient Magnitude")
    ax.set_title(f"Temporal Gradient Profile - {horizon}h Horizon")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / f"gradient_temporal_profile_h{horizon}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Save gradient data
    grad_df = pd.DataFrame(avg_grad, columns=feature_names)
    grad_df.to_csv(output_dir / f"gradient_attribution_h{horizon}.csv", index=False)

    print(f"  [B] Gradient attribution: {grads.shape[0]} samples, {len(feature_names)} features")


# ======================================================================
#  Section C: Hidden State Structural Analysis
# ======================================================================

def run_hidden_state_analysis(
    best_tft,
    test_dl,
    test_df: pd.DataFrame,
    config: dict,
    horizon: int,
    output_dir: Path,
) -> None:
    """PCA and clustering on LSTM hidden states within TFT."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("  [C] Extracting LSTM hidden states via hooks...")

    hook_mgr = TFTHookManager(best_tft)
    best_tft.eval()

    all_encoder_outputs = []
    all_targets = []
    max_batches = 30

    with torch.no_grad():
        for i, (x, y) in enumerate(test_dl):
            if i >= max_batches:
                break
            best_tft(x)

            if "lstm_encoder_output" in hook_mgr.captures:
                enc_out = hook_mgr.captures["lstm_encoder_output"].cpu().numpy()
                # Take last timestep hidden state per sample
                last_hidden = enc_out[:, -1, :]
                all_encoder_outputs.append(last_hidden)
                all_targets.append(y[0].cpu().numpy().ravel())
            hook_mgr.clear()

    hook_mgr.remove_hooks()

    if not all_encoder_outputs:
        print("  [C] No LSTM hidden states captured. Skipping.")
        return

    hidden_states = np.concatenate(all_encoder_outputs, axis=0)
    target_values = np.concatenate(all_targets, axis=0)
    # Trim to match
    n = min(len(hidden_states), len(target_values))
    hidden_states = hidden_states[:n]
    target_values = target_values[:n]

    print(f"  [C] Hidden states shape: {hidden_states.shape}")

    # PCA
    n_components = min(20, hidden_states.shape[1])
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(hidden_states)

    # Plot 7: PCA explained variance
    fig, ax = plt.subplots(figsize=(8, 5))
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    ax.bar(range(1, n_components + 1), pca.explained_variance_ratio_ * 100, color=C_PRIMARY, alpha=0.7)
    ax.plot(range(1, n_components + 1), cumvar, "r-o", markersize=4, label="Cumulative")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title(f"PCA on Encoder Hidden States - {horizon}h")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / f"pca_variance_h{horizon}.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # CO2 level categories for coloring
    co2_bins = [0, 500, 1000, np.inf]
    co2_labels = ["Low (<500)", "Medium (500-1000)", "High (>1000)"]
    co2_cats = pd.cut(target_values, bins=co2_bins, labels=co2_labels)
    cat_colors = {
        "Low (<500)": C_ACCENT,
        "Medium (500-1000)": C_PRIMARY,
        "High (>1000)": C_SECONDARY,
    }

    # Plot 8: PCA 2D scatter colored by CO2 level
    fig, ax = plt.subplots(figsize=(8, 7))
    for label in co2_labels:
        mask = co2_cats == label
        if mask.sum() > 0:
            ax.scatter(
                pca_result[mask, 0],
                pca_result[mask, 1],
                c=cat_colors[label],
                label=f"{label} (n={mask.sum()})",
                alpha=0.5,
                s=10,
            )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"LSTM Hidden States by CO2 Level - {horizon}h")
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / f"pca_co2_level_h{horizon}.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # Plot 9: PCA 2D scatter colored by hour of day
    datetime_col = config["data"]["datetime_column"]
    if datetime_col in test_df.columns:
        # Align dates to hidden states (tail alignment)
        test_dates = pd.DatetimeIndex(test_df[datetime_col])
        if len(test_dates) >= n:
            hours = test_dates[-n:].hour
        else:
            hours = np.zeros(n)
    else:
        hours = np.zeros(n)

    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(
        pca_result[:, 0],
        pca_result[:, 1],
        c=hours,
        cmap="twilight",
        alpha=0.5,
        s=10,
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"LSTM Hidden States by Hour of Day - {horizon}h")
    plt.colorbar(scatter, ax=ax, label="Hour of Day")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"pca_hour_h{horizon}.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # Plot 10: K-means clustering
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(hidden_states)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter colored by cluster
    ax = axes[0]
    for c in range(n_clusters):
        mask = cluster_labels == c
        ax.scatter(
            pca_result[mask, 0], pca_result[mask, 1],
            label=f"Cluster {c} (n={mask.sum()})", alpha=0.5, s=10,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("K-Means Clusters in PCA Space")
    ax.legend(fontsize=8)

    # Cluster vs CO2 regime crosstab
    ax = axes[1]
    ax.axis("off")
    cross = pd.crosstab(
        pd.Series(cluster_labels, name="Cluster"),
        pd.Series(co2_cats, name="CO2 Level"),
    )
    rows_data = []
    for c_idx in range(n_clusters):
        row = [f"Cluster {c_idx}"]
        for label in co2_labels:
            val = cross.loc[c_idx, label] if c_idx in cross.index and label in cross.columns else 0
            row.append(str(val))
        rows_data.append(row)

    tbl = ax.table(
        cellText=rows_data,
        colLabels=["Cluster"] + co2_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.5)
    ax.set_title("Cluster vs CO2 Regime", fontsize=11, pad=20)

    fig.suptitle(f"Hidden State Clustering - {horizon}h", fontsize=13)
    plt.tight_layout()
    plt.savefig(
        output_dir / f"clustering_h{horizon}.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    print(f"  [C] Hidden state analysis complete: {n} samples, {n_components} PCs")


# ======================================================================
#  Section D: Temporal Pattern Analysis (FFT + Autocorrelation)
# ======================================================================

def run_temporal_patterns(
    best_tft,
    test_dl,
    raw_predictions: dict,
    horizon: int,
    output_dir: Path,
) -> None:
    """FFT and autocorrelation on attention weights and hidden states."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("  [D] Analyzing temporal patterns...")

    # Extract attention from interpret_output
    try:
        interp = best_tft.interpret_output(raw_predictions.output, reduction="none")
        attn_per_sample = interp["attention"].detach().cpu().numpy()
    except Exception as e:
        print(f"  [D] Could not extract attention: {e}")
        return

    # Average attention profile across samples
    avg_attn = attn_per_sample.mean(axis=0)
    n_steps = len(avg_attn)

    # Also collect per-sample attention for time series analysis
    # Use the attention weight at a fixed encoder position (e.g., most recent = last)
    # across sequential test samples
    if attn_per_sample.ndim == 2 and attn_per_sample.shape[0] > 50:
        # Each sample's total attention is a profile; take mean attention per sample
        attn_time_series = attn_per_sample.mean(axis=1)
    else:
        attn_time_series = avg_attn

    # Extract hidden states for temporal analysis
    hook_mgr = TFTHookManager(best_tft)
    best_tft.eval()
    all_hidden = []
    max_batches = 50

    with torch.no_grad():
        for i, (x, y) in enumerate(test_dl):
            if i >= max_batches:
                break
            best_tft(x)
            if "lstm_encoder_output" in hook_mgr.captures:
                enc_out = hook_mgr.captures["lstm_encoder_output"].cpu().numpy()
                all_hidden.append(enc_out[:, -1, :])
            hook_mgr.clear()
    hook_mgr.remove_hooks()

    # Plot 11: FFT of attention profile
    fig, ax = plt.subplots(figsize=(10, 5))
    if n_steps > 4:
        # FFT of the average attention profile
        yf = np.abs(fft(avg_attn - avg_attn.mean()))
        xf = fftfreq(n_steps, d=1.0)  # d=1 hour for 1h resolution
        pos_mask = xf > 0
        ax.plot(1.0 / xf[pos_mask], yf[pos_mask], color=C_PRIMARY, linewidth=1.5)
        ax.set_xlabel("Period (hours)")
        ax.set_ylabel("FFT Magnitude")
        ax.set_title(f"FFT of Attention Profile - {horizon}h")
        ax.set_xlim(0, min(n_steps, 168))  # Up to 1 week
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Not enough attention steps for FFT", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(f"FFT of Attention Profile - {horizon}h (N/A)")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"fft_attention_h{horizon}.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # Plot 12: Autocorrelation of hidden state PCs
    if all_hidden:
        hidden_concat = np.concatenate(all_hidden, axis=0)
        n_samples_h = hidden_concat.shape[0]

        if n_samples_h > 50:
            pca_temp = PCA(n_components=3)
            pcs = pca_temp.fit_transform(hidden_concat)
            max_lag = min(72, n_samples_h // 3)

            fig, axes = plt.subplots(3, 1, figsize=(12, 9))
            for pc_idx in range(3):
                ax = axes[pc_idx]
                pc_series = pcs[:, pc_idx]
                pc_series = pc_series - pc_series.mean()
                norm = np.sum(pc_series**2)
                if norm > 0:
                    autocorr = np.correlate(pc_series, pc_series, mode="full")
                    autocorr = autocorr[len(autocorr) // 2:]
                    autocorr = autocorr[:max_lag] / norm
                else:
                    autocorr = np.zeros(max_lag)

                ax.bar(range(max_lag), autocorr, color=C_PRIMARY, alpha=0.7, width=0.8)
                ax.axhline(0, color="black", linewidth=0.5)
                ax.set_ylabel(f"PC{pc_idx+1} ACF")
                ax.set_title(f"PC{pc_idx+1} ({pca_temp.explained_variance_ratio_[pc_idx]*100:.1f}% var)")
                ax.grid(axis="y", alpha=0.3)

            axes[-1].set_xlabel("Lag (hours)")
            fig.suptitle(f"Autocorrelation of Hidden State PCs - {horizon}h", fontsize=13)
            plt.tight_layout()
            plt.savefig(
                output_dir / f"autocorrelation_hidden_pcs_h{horizon}.png",
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)
        else:
            print("  [D] Not enough hidden state samples for autocorrelation")

    # Plot 13: Attention time series with periodic patterns
    if len(attn_time_series) > 50:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # Raw attention time series
        ax = axes[0]
        ax.plot(attn_time_series[:min(500, len(attn_time_series))],
                color=C_PRIMARY, linewidth=0.8, alpha=0.8)
        ax.set_xlabel("Test Sample Index")
        ax.set_ylabel("Mean Attention Weight")
        ax.set_title(f"Attention Weight Time Series - {horizon}h")
        ax.grid(alpha=0.3)

        # FFT of attention time series
        ax = axes[1]
        n_ts = len(attn_time_series)
        yf = np.abs(fft(attn_time_series - attn_time_series.mean()))
        xf = fftfreq(n_ts, d=1.0)
        pos_mask = xf > 0
        periods = 1.0 / xf[pos_mask]
        magnitudes = yf[pos_mask]
        valid = periods < 200  # Limit to reasonable periods
        ax.plot(periods[valid], magnitudes[valid], color=C_SECONDARY, linewidth=1)
        ax.set_xlabel("Period (hours)")
        ax.set_ylabel("FFT Magnitude")
        ax.set_title("FFT of Attention Time Series")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / f"attention_periodicity_h{horizon}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

    print(f"  [D] Temporal pattern analysis complete")


# ======================================================================
#  Section E1: Variable Selection Network Analysis
# ======================================================================

def run_vsn_analysis(
    best_tft,
    raw_predictions: dict,
    horizon: int,
    output_dir: Path,
) -> dict:
    """Extract and visualize VSN weights (encoder + decoder)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("  [E1] Extracting Variable Selection Network weights...")

    # Aggregate importance
    interp_sum = best_tft.interpret_output(raw_predictions.output, reduction="sum")

    encoder_names = list(best_tft.encoder_variables)
    decoder_names = list(best_tft.decoder_variables)

    enc_imp = interp_sum["encoder_variables"].detach().cpu().numpy()
    dec_imp = interp_sum["decoder_variables"].detach().cpu().numpy()

    enc_pct = enc_imp / enc_imp.sum() * 100 if enc_imp.sum() > 0 else enc_imp
    dec_pct = dec_imp / dec_imp.sum() * 100 if dec_imp.sum() > 0 else dec_imp

    encoder_df = pd.DataFrame({
        "variable": encoder_names,
        "importance": enc_imp,
        "importance_pct": enc_pct,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    decoder_df = pd.DataFrame({
        "variable": decoder_names,
        "importance": dec_imp,
        "importance_pct": dec_pct,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Per-sample importance
    interp_none = best_tft.interpret_output(raw_predictions.output, reduction="none")
    enc_per_sample = interp_none["encoder_variables"].detach().cpu().numpy()
    dec_per_sample = interp_none["decoder_variables"].detach().cpu().numpy()

    var_imp = {
        "encoder_agg": encoder_df,
        "decoder_agg": decoder_df,
        "encoder_per_sample": enc_per_sample,
        "decoder_per_sample": dec_per_sample,
        "encoder_names": encoder_names,
        "decoder_names": decoder_names,
    }

    # Plot 14: Encoder variable importance
    for var_type, df, color in [
        ("encoder", encoder_df, C_PRIMARY),
        ("decoder", decoder_df, C_SECONDARY),
    ]:
        fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.35)))
        df_sorted = df.sort_values("importance_pct", ascending=True)
        ax.barh(
            range(len(df_sorted)),
            df_sorted["importance_pct"].values,
            color=color,
            edgecolor="white",
        )
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted["variable"].values, fontsize=9)
        ax.set_xlabel("Importance (%)")
        ax.set_title(f"TFT {var_type.title()} VSN Importance - {horizon}h")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{var_type}_importance_h{horizon}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

    # Save CSVs
    encoder_df.to_csv(output_dir / f"encoder_importance_h{horizon}.csv", index=False)
    decoder_df.to_csv(output_dir / f"decoder_importance_h{horizon}.csv", index=False)

    # Plot 16: VSN weight dynamics over time (stacked area)
    n_total = enc_per_sample.shape[0]
    if n_total > 50:
        window_size = min(100, n_total // 3)
        starts = [0, max(0, n_total // 2 - window_size // 2), max(0, n_total - window_size)]

        # Normalize per sample
        row_sums = enc_per_sample.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        enc_norm = enc_per_sample / row_sums

        # Top 8 by aggregate
        agg_imp_arr = enc_norm.mean(axis=0)
        top_k_idx = np.argsort(agg_imp_arr)[-8:][::-1]
        other_idx = [j for j in range(len(encoder_names)) if j not in top_k_idx]
        cmap = plt.colormaps.get_cmap("tab20").resampled(len(top_k_idx) + 1)

        fig, axes = plt.subplots(len(starts), 1, figsize=(14, 4 * len(starts)))
        for w_idx, start in enumerate(starts):
            ax = axes[w_idx]
            end = min(start + window_size, n_total)
            window = enc_norm[start:end]

            stack_data = [window[:, fi] for fi in top_k_idx]
            other_sum = np.sum(window[:, other_idx], axis=1)
            stack_data.append(other_sum)

            stack_labels = [encoder_names[fi] for fi in top_k_idx] + ["Other"]

            ax.stackplot(
                range(start, end),
                *stack_data,
                labels=stack_labels if w_idx == 0 else [None] * len(stack_labels),
                colors=[cmap(j) for j in range(len(stack_data))],
                alpha=0.8,
            )
            ax.set_ylabel("Importance Share")
            ax.set_title(f"Samples {start}-{end}")
            ax.set_ylim(0, 1)

        axes[-1].set_xlabel("Test Sample Index")
        axes[0].legend(loc="upper right", fontsize=7, ncol=3)
        fig.suptitle(f"Encoder VSN Weight Dynamics - {horizon}h", fontsize=13)
        plt.tight_layout()
        plt.savefig(
            output_dir / f"vsn_dynamics_h{horizon}.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    # Plot 17: VSN weight distribution per variable (violin plot)
    if n_total > 10:
        # Use top 10 most important variables for readability
        top_10_idx = np.argsort(enc_pct)[-10:][::-1]
        violin_data = [enc_per_sample[:, fi] for fi in top_10_idx]
        violin_names = [encoder_names[fi] for fi in top_10_idx]

        fig, ax = plt.subplots(figsize=(10, 6))
        parts = ax.violinplot(violin_data, positions=range(len(violin_names)), showmeans=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(C_PRIMARY)
            pc.set_alpha(0.6)
        ax.set_xticks(range(len(violin_names)))
        ax.set_xticklabels(violin_names, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("VSN Weight")
        ax.set_title(f"Encoder VSN Weight Distribution (Top 10) - {horizon}h")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            output_dir / f"vsn_violin_h{horizon}.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    # Print top-5
    print("  Top 5 encoder variables:")
    for _, row in encoder_df.head(5).iterrows():
        print(f"    {row['variable']}: {row['importance_pct']:.1f}%")

    print(f"  [E1] VSN analysis complete")
    return var_imp


# ======================================================================
#  Section E2: Multi-Head Attention Analysis
# ======================================================================

def run_attention_analysis(
    best_tft,
    raw_predictions: dict,
    horizon: int,
    output_dir: Path,
) -> None:
    """Analyze per-head attention patterns."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("  [E2] Analyzing multi-head attention...")

    out = raw_predictions.output

    try:
        enc_attn_raw = out["encoder_attention"]
    except (KeyError, IndexError, AttributeError):
        print("  [E2] No encoder_attention in output. Skipping.")
        return

    if enc_attn_raw is None:
        print("  [E2] encoder_attention is None. Skipping.")
        return

    # Shape: (batch, decoder_len, n_heads, encoder_len)
    enc_attn = enc_attn_raw.detach().cpu().numpy()
    n_samples, dec_len, n_heads, enc_len = enc_attn.shape

    # Average across samples and decoder positions
    # -> (n_heads, encoder_len)
    avg_head_attn = enc_attn.mean(axis=(0, 1))

    # Plot 18: Per-head attention heatmap
    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 5))
    if n_heads == 1:
        axes = [axes]

    for h_idx in range(n_heads):
        ax = axes[h_idx]
        # Average across samples, show (decoder_len, encoder_len) for this head
        head_attn = enc_attn[:, :, h_idx, :].mean(axis=0)
        im = ax.imshow(head_attn, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_xlabel("Encoder Step")
        ax.set_ylabel("Decoder Step")
        ax.set_title(f"Head {h_idx}")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"Per-Head Attention Patterns - {horizon}h", fontsize=13)
    plt.tight_layout()
    plt.savefig(
        output_dir / f"per_head_attention_h{horizon}.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # Plot 19: Average attention profile (all heads combined)
    interp_sum = best_tft.interpret_output(raw_predictions.output, reduction="sum")
    attn_profile = interp_sum["attention"].detach().cpu().numpy()
    max_enc_len = best_tft.hparams.max_encoder_length

    attn_norm = attn_profile / attn_profile.sum() if attn_profile.sum() > 0 else attn_profile
    time_idx = np.arange(-max_enc_len, len(attn_norm) - max_enc_len)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(time_idx, attn_norm, color=C_PRIMARY, alpha=0.8, width=0.8)
    ax.axvline(-0.5, color="red", linewidth=1.5, linestyle="--", label="Encoder/Decoder boundary")
    ax.set_xlabel("Relative Time (hours, negative=past)")
    ax.set_ylabel("Normalized Attention")
    ax.set_title(f"Average Attention Profile - {horizon}h")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / f"attention_profile_h{horizon}.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # Plot 20: Head similarity matrix
    # Cosine similarity between heads' average attention patterns
    head_profiles = avg_head_attn  # (n_heads, encoder_len)
    norms = np.linalg.norm(head_profiles, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    head_normed = head_profiles / norms
    similarity = head_normed @ head_normed.T

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(similarity, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_heads))
    ax.set_xticklabels([f"Head {i}" for i in range(n_heads)])
    ax.set_yticklabels([f"Head {i}" for i in range(n_heads)])
    for i_h in range(n_heads):
        for j_h in range(n_heads):
            ax.text(j_h, i_h, f"{similarity[i_h, j_h]:.2f}", ha="center", va="center", fontsize=10)
    ax.set_title(f"Head Attention Similarity - {horizon}h")
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"head_similarity_h{horizon}.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    print(f"  [E2] Attention analysis: {n_heads} heads, encoder_len={enc_len}")


# ======================================================================
#  Section E3: Prediction Quality Analysis
# ======================================================================

def run_prediction_analysis(
    preds: dict[str, np.ndarray],
    metrics: dict[str, float],
    test_df: pd.DataFrame,
    config: dict,
    horizon: int,
    output_dir: Path,
) -> None:
    """Prediction overlay, scatter, residuals, error by CO2 level."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("  [E3] Generating prediction analysis plots...")

    y_true = preds["y_true"]
    y_pred = preds["y_pred"]
    residuals = y_true - y_pred

    # Plot 21: Predictions overlay
    n_show = min(500, len(y_true))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_true[:n_show], label="Actual", color=C_PRIMARY, linewidth=1.0, alpha=0.8)
    ax.plot(y_pred[:n_show], label="Predicted", color=C_SECONDARY, linewidth=1.0, alpha=0.8)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("CO2 (ppm)")
    ax.set_title(f"TFT Predictions vs Actual - {horizon}h")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / f"predictions_overlay_h{horizon}.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # Plot 22: Scatter with R2
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.3, s=5, color=C_PRIMARY)
    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())
    ax.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=1.5, label="Perfect")
    ax.text(
        0.05, 0.95,
        f"R2 = {metrics['r2']:.4f}\nRMSE = {metrics['rmse']:.2f}\nMAE = {metrics['mae']:.2f}",
        transform=ax.transAxes, fontsize=11, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax.set_xlabel("Actual CO2 (ppm)")
    ax.set_ylabel("Predicted CO2 (ppm)")
    ax.set_title(f"TFT Scatter - {horizon}h")
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / f"scatter_r2_h{horizon}.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # Plot 23: 4-panel residual analysis
    datetime_col = config["data"]["datetime_column"]
    test_dates = None
    if datetime_col in test_df.columns:
        test_dates = pd.DatetimeIndex(test_df[datetime_col])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Distribution
    ax = axes[0, 0]
    ax.hist(residuals, bins=50, color=C_PRIMARY, alpha=0.7, edgecolor="white", density=True)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Residual (ppm)")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution")
    ax.text(
        0.95, 0.95,
        f"Mean: {residuals.mean():.2f}\nStd: {residuals.std():.2f}",
        transform=ax.transAxes, fontsize=9, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Over time
    ax = axes[0, 1]
    ax.scatter(np.arange(len(residuals)), residuals, alpha=0.3, s=3, color=C_SECONDARY)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Residual (ppm)")
    ax.set_title("Residuals Over Time")

    # By hour of day
    ax = axes[1, 0]
    if test_dates is not None and len(test_dates) >= len(residuals):
        hours = test_dates[-len(residuals):].hour
        res_df = pd.DataFrame({"hour": hours, "residual": residuals})
        res_df.boxplot(column="residual", by="hour", ax=ax)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Residual (ppm)")
        ax.set_title("Residuals by Hour")
        plt.suptitle("")
    else:
        ax.text(0.5, 0.5, "No datetime available", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Residuals by Hour (N/A)")

    # By day of week
    ax = axes[1, 1]
    if test_dates is not None and len(test_dates) >= len(residuals):
        dow = test_dates[-len(residuals):].dayofweek
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        res_df = pd.DataFrame({"day_num": dow, "residual": residuals})
        res_df.boxplot(column="residual", by="day_num", ax=ax)
        ax.set_xticklabels(day_names)
        ax.set_xlabel("Day of Week")
        ax.set_ylabel("Residual (ppm)")
        ax.set_title("Residuals by Day")
        plt.suptitle("")
    else:
        ax.text(0.5, 0.5, "No datetime available", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Residuals by Day (N/A)")

    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"residual_analysis_h{horizon}.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # Plot 24: Error by CO2 level
    bins = [0, 500, 1000, np.inf]
    labels = ["Low (<500)", "Medium (500-1000)", "High (>1000)"]
    categories = pd.cut(y_true, bins=bins, labels=labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    abs_errors = np.abs(residuals)
    err_df = pd.DataFrame({"category": categories, "abs_error": abs_errors})
    err_df.boxplot(column="abs_error", by="category", ax=ax)
    ax.set_xlabel("CO2 Level")
    ax.set_ylabel("Absolute Error (ppm)")
    ax.set_title(f"Error by CO2 Level - {horizon}h")
    plt.suptitle("")

    ax = axes[1]
    ax.axis("off")
    rows_data = []
    for label in labels:
        mask = categories == label
        if mask.sum() > 0:
            bin_res = residuals[mask]
            rows_data.append([
                label,
                f"{mask.sum()}",
                f"{np.sqrt(np.mean(bin_res**2)):.2f}",
                f"{np.mean(np.abs(bin_res)):.2f}",
                f"{np.mean(bin_res):.2f}",
            ])
        else:
            rows_data.append([label, "0", "N/A", "N/A", "N/A"])

    tbl = ax.table(
        cellText=rows_data,
        colLabels=["CO2 Level", "N", "RMSE", "MAE", "Mean Bias"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.5)

    plt.tight_layout()
    plt.savefig(
        output_dir / f"error_by_co2_level_h{horizon}.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)

    print(f"  [E3] Prediction analysis complete")


# ======================================================================
#  Cross-Horizon Comparison and Summary
# ======================================================================

def run_cross_horizon_comparison(
    all_var_imp: dict[int, dict],
    all_metrics: dict[int, dict],
    output_dir: Path,
) -> None:
    """Compare variable importance and metrics across horizons."""
    output_dir.mkdir(parents=True, exist_ok=True)
    horizons = sorted(all_var_imp.keys())
    print(f"  Generating cross-horizon comparison for horizons: {horizons}")

    if len(horizons) < 2:
        print("  Need >= 2 horizons for comparison. Skipping.")
        return

    # Plot 25: Side-by-side encoder importance
    fig, axes = plt.subplots(1, len(horizons), figsize=(8 * len(horizons), 8))
    if len(horizons) == 1:
        axes = [axes]
    colors = [C_PRIMARY, C_SECONDARY, C_ACCENT, C_WARN]

    for idx, h in enumerate(horizons):
        ax = axes[idx]
        df = all_var_imp[h]["encoder_agg"].sort_values("importance_pct", ascending=True)
        ax.barh(range(len(df)), df["importance_pct"].values, color=colors[idx % len(colors)])
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["variable"].values, fontsize=9)
        ax.set_xlabel("Importance (%)")
        ax.set_title(f"{h}h Horizon")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Encoder Variable Importance: Cross-Horizon", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        output_dir / "cross_horizon_encoder_importance.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)

    # Plot 26: Attention profile comparison
    # Read attention profiles from saved data
    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, h in enumerate(horizons):
        attn_csv = RESULTS_BASE / f"h{h}" / "attention" / f"attention_profile_h{h}.png"
        # Instead of reading image, re-extract from summary data
        label = f"{h}h"
        ax.text(0.5, 0.5 - idx * 0.15, f"See h{h}/attention/ for detailed attention profiles",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_title("Attention Profile Comparison")
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(
        output_dir / "cross_horizon_attention_note.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # Plot 27: Summary metrics table
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    rows_data = []
    for h in horizons:
        m = all_metrics[h]
        rows_data.append([
            f"{h}h",
            f"{m['rmse']:.2f}",
            f"{m['mae']:.2f}",
            f"{m['r2']:.4f}",
            f"{m['mape']:.2f}%",
        ])

    tbl = ax.table(
        cellText=rows_data,
        colLabels=["Horizon", "RMSE", "MAE", "R2", "MAPE"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 2.0)
    ax.set_title("TFT Performance Summary (preproc_D Enhanced 1h)", fontsize=13, pad=30)
    plt.tight_layout()
    plt.savefig(
        output_dir / "metrics_summary.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def generate_summary_figure(
    all_metrics: dict[int, dict],
    all_var_imp: dict[int, dict],
    output_dir: Path,
) -> None:
    """Multi-panel summary figure with key findings."""
    output_dir.mkdir(parents=True, exist_ok=True)
    horizons = sorted(all_metrics.keys())

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # [0,0] Metrics table
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    rows_data = []
    for h in horizons:
        m = all_metrics[h]
        rows_data.append([
            f"{h}h", f"{m['rmse']:.2f}", f"{m['mae']:.2f}",
            f"{m['r2']:.4f}", f"{m['mape']:.2f}%",
        ])
    tbl = ax.table(
        cellText=rows_data,
        colLabels=["Horizon", "RMSE", "MAE", "R2", "MAPE"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 1.8)
    ax.set_title("Performance Metrics", fontsize=12, pad=20)

    # [0,1] and [0,2]: Encoder importance per horizon
    colors_list = [C_PRIMARY, C_SECONDARY]
    for col, h in enumerate(horizons[:2], start=1):
        ax = fig.add_subplot(gs[0, col])
        df = all_var_imp[h]["encoder_agg"].sort_values("importance_pct", ascending=True).tail(10)
        ax.barh(range(len(df)), df["importance_pct"].values, color=colors_list[col - 1])
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["variable"].values, fontsize=8)
        ax.set_xlabel("Importance (%)")
        ax.set_title(f"Top Encoder Vars - {h}h", fontsize=10)

    # Bottom row: Analysis highlights
    section_names = [
        "Gate Dynamics (A)",
        "Gradient Attribution (B)",
        "Hidden State Analysis (C)",
    ]
    for col in range(3):
        ax = fig.add_subplot(gs[1, col])
        ax.text(
            0.5, 0.5,
            f"Section {section_names[col]}\nSee detailed plots in\nresults/tft_interpretability/",
            ha="center", va="center", fontsize=10, style="italic",
            transform=ax.transAxes,
        )
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "TFT Interpretability Study Summary - preproc_D (Enhanced 1h)",
        fontsize=15, fontweight="bold",
    )
    plt.savefig(output_dir / "summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_study_results(
    all_metrics: dict[int, dict],
    all_var_imp: dict[int, dict],
    output_dir: Path,
) -> None:
    """Save all metrics and importance data to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "study": "TFT Interpretability",
        "variant": "preproc_D (Enhanced 1h)",
        "timestamp": datetime.now().isoformat(),
        "horizons": {},
    }

    for h in sorted(all_metrics.keys()):
        results["horizons"][str(h)] = {
            "metrics": all_metrics[h],
            "encoder_importance": all_var_imp[h]["encoder_agg"].to_dict(orient="records"),
            "decoder_importance": all_var_imp[h]["decoder_agg"].to_dict(orient="records"),
        }

    with open(output_dir / "study_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved to: {output_dir / 'study_results.json'}")


# ======================================================================
#  Main
# ======================================================================

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="TFT Interpretability Study on preproc_D (Enhanced 1h)"
    )
    parser.add_argument(
        "--horizons", nargs="+", type=int, default=[1, 24],
        help="Forecast horizons in hours (default: 1 24)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override max epochs for TFT training",
    )
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  TFT INTERPRETABILITY STUDY (DEEP ANALYSIS)")
    print(f"  Variant: preproc_D (Enhanced 1h)")
    print(f"  Horizons: {args.horizons}")
    print(f"  Sections: A(gates) B(gradients) C(hidden) D(temporal) E(TFT-specific)")
    print(f"{'='*70}\n")

    # Load pipeline data once
    pipeline_config = load_interpretability_config(horizon=1, epochs_override=args.epochs)
    seed_everything(pipeline_config["training"]["seed"])

    raw_dir = Path(pipeline_config["data"].get("raw_dir", "data/raw"))
    print("  Loading preprocessing pipeline (preproc_D)...")
    train_df, val_df, test_df = run_preprocessing_pipeline(
        raw_dir=raw_dir, variant_config=pipeline_config,
    )
    print(f"  Pipeline loaded: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    all_metrics: dict[int, dict] = {}
    all_var_imp: dict[int, dict] = {}

    for horizon in args.horizons:
        print(f"\n{'-'*60}")
        print(f"  HORIZON: {horizon}h")
        print(f"{'-'*60}\n")

        config = load_interpretability_config(horizon=horizon, epochs_override=args.epochs)
        seed_everything(config["training"]["seed"])

        output_dir = RESULTS_BASE / f"h{horizon}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # ---- Train TFT ----
        t0 = time.time()
        print(f"  Training TFT for {horizon}h horizon...")
        best_tft, test_dl, raw_predictions, training_data = train_tft(
            config, train_df.copy(), val_df.copy(), test_df.copy(), horizon,
        )
        elapsed = time.time() - t0
        print(f"  Training completed in {elapsed:.1f}s")

        # ---- Extract predictions + metrics ----
        print("  Extracting predictions...")
        preds = extract_predictions(best_tft, test_dl)
        metrics = compute_metrics(preds["y_true"], preds["y_pred"])
        all_metrics[horizon] = metrics
        print(f"  RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  "
              f"R2={metrics['r2']:.4f}  MAPE={metrics['mape']:.2f}%")

        # ---- Section A: Gate Dynamics ----
        run_gate_dynamics(best_tft, test_dl, horizon, output_dir / "gate_dynamics")

        # ---- Section B: Gradient Attribution ----
        run_gradient_attribution(best_tft, test_dl, config, horizon, output_dir / "gradient_attribution")

        # ---- Section C: Hidden State Analysis ----
        run_hidden_state_analysis(
            best_tft, test_dl, test_df, config, horizon, output_dir / "hidden_state_analysis"
        )

        # ---- Section D: Temporal Patterns ----
        run_temporal_patterns(best_tft, test_dl, raw_predictions, horizon, output_dir / "temporal_patterns")

        # ---- Section E1: VSN Analysis ----
        var_imp = run_vsn_analysis(
            best_tft, raw_predictions, horizon, output_dir / "variable_importance"
        )
        all_var_imp[horizon] = var_imp

        # ---- Section E2: Attention Analysis ----
        run_attention_analysis(best_tft, raw_predictions, horizon, output_dir / "attention")

        # ---- Section E3: Prediction Analysis ----
        run_prediction_analysis(preds, metrics, test_df, config, horizon, output_dir / "predictions")

        # ---- Save metrics + predictions ----
        save_metrics(
            metrics, f"TFT_h{horizon}", output_dir / "metrics.json",
            experiment_info={
                "name": "tft_interpretability",
                "label": f"TFT Deep Analysis h={horizon}",
                "description": "preproc_D Enhanced 1h variant",
            },
        )
        np.savez(output_dir / "predictions.npz", y_true=preds["y_true"], y_pred=preds["y_pred"])

        # ---- GPU cleanup ----
        del best_tft, test_dl, raw_predictions, training_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  GPU memory freed\n")

    # ---- Cross-horizon comparison ----
    if len(args.horizons) > 1:
        print(f"\n{'-'*60}")
        print(f"  CROSS-HORIZON COMPARISON")
        print(f"{'-'*60}\n")
        run_cross_horizon_comparison(all_var_imp, all_metrics, RESULTS_BASE / "comparison")

    # ---- Summary ----
    generate_summary_figure(all_metrics, all_var_imp, RESULTS_BASE)
    save_study_results(all_metrics, all_var_imp, RESULTS_BASE)

    print(f"\n{'='*70}")
    print(f"  TFT INTERPRETABILITY STUDY COMPLETE")
    print(f"  Results saved to: {RESULTS_BASE}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
