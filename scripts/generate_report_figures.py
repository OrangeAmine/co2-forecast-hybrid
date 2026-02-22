"""Generate all comparison figures for the experimental report.

Creates:
  1. Bar chart comparing h=1 models (baseline vs preproc_E vs ensemble)
  2. Bar chart comparing h=24 models (direct vs recursive — the breakthrough)
  3. Per-step error growth for recursive h=24
  4. Conformal prediction interval visualization
  5. Optuna hyperparameter importance / convergence
  6. Preprocessing pipeline comparison (A→D→E→F)
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

matplotlib.use("Agg")
sns.set_theme(style="whitegrid", font_scale=1.1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
REPORT_FIGS = PROJECT_ROOT / "reports" / "figures"
REPORT_FIGS.mkdir(parents=True, exist_ok=True)


def load_metrics(path: Path) -> dict:
    """Load metrics from JSON, handling nested format."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("metrics", data)


# ────────────────────────────────────────────────────────────────────
#  Figure 1: h=1 Model Comparison Bar Chart
# ────────────────────────────────────────────────────────────────────
def fig1_h1_comparison():
    models = {
        "SARIMA": RESULTS / "sarima_interpretability/h1/metrics.json",
        "CatBoost": RESULTS / "catboost_interpretability/h1/metrics.json",
        "XGBoost": RESULTS / "xgboost_interpretability/h1/metrics.json",
        "CNN-LSTM": RESULTS / "cnn_lstm_interpretability/h1/metrics.json",
        "TFT": RESULTS / "tft_interpretability/h1/metrics.json",
        "LSTM\n(baseline)": RESULTS / "lstm_interpretability/h1/metrics.json",
        "LSTM+Occ\n(preproc_E)": RESULTS / "preproc_E_LSTM_h1_20260221_220750/metrics.json",
        "Seq2Seq+Occ\n(preproc_E)": RESULTS / "preproc_E_Seq2Seq_h1_20260221_215359/metrics.json",
        "LSTM+Occ\n(lookback=36)": RESULTS / "preproc_E_LSTM_h1_20260221_232331/metrics.json",
        "Weighted\nEnsemble": RESULTS / "ensemble_preproc_E_h1/weighted_metrics.json",
    }

    names, rmses, r2s = [], [], []
    for name, path in models.items():
        if path.exists():
            m = load_metrics(path)
            names.append(name)
            rmses.append(m["rmse"])
            r2s.append(m["r2"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = ["#d62728" if "SARIMA" in n else
              "#ff7f0e" if "baseline" in n else
              "#2ca02c" if "Ensemble" in n else
              "#1f77b4" if "preproc" in n or "lookback" in n else
              "#9467bd" for n in names]

    bars1 = ax1.barh(names, rmses, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("RMSE (ppm)", fontsize=12)
    ax1.set_title("1-Hour Forecast: RMSE Comparison", fontsize=14, fontweight="bold")
    ax1.invert_yaxis()
    for bar, val in zip(bars1, rmses):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}", va="center", fontsize=9)

    bars2 = ax2.barh(names, r2s, color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("R² Score", fontsize=12)
    ax2.set_title("1-Hour Forecast: R² Comparison", fontsize=14, fontweight="bold")
    ax2.set_xlim(0.4, 1.0)
    ax2.invert_yaxis()
    for bar, val in zip(bars2, r2s):
        ax2.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(REPORT_FIGS / "fig1_h1_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Fig 1: h=1 comparison saved")


# ────────────────────────────────────────────────────────────────────
#  Figure 2: h=24 Direct vs Recursive — The Breakthrough
# ────────────────────────────────────────────────────────────────────
def fig2_h24_breakthrough():
    models_direct = {
        "SARIMA\n(direct)": RESULTS / "sarima_interpretability/h24/metrics.json",
        "CNN-LSTM\n(direct)": RESULTS / "cnn_lstm_interpretability/h24/metrics.json",
        "LSTM\n(direct)": RESULTS / "lstm_interpretability/h24/metrics.json",
        "XGBoost\n(direct)": RESULTS / "xgboost_interpretability/h24/metrics.json",
        "CatBoost\n(direct)": RESULTS / "catboost_interpretability/h24/metrics.json",
        "Seq2Seq+Occ\n(direct)": RESULTS / "preproc_E_Seq2Seq_h24_20260221_215838/metrics.json",
        "LSTM+Occ\n(direct)": RESULTS / "preproc_E_LSTM_h24_20260221_220951/metrics.json",
    }
    models_recursive = {
        "LSTM+Occ\nrecursive\n(lookback=24)": RESULTS / "preproc_E_LSTM_recursive_h24_20260221_224729/metrics.json",
        "LSTM+Wavelet\nrecursive": RESULTS / "preproc_F_LSTM_recursive_h24_20260221_225042/metrics.json",
        "LSTM+Occ\nrecursive\n(lookback=36)": RESULTS / "preproc_E_LSTM_recursive_h24_20260221_232807/metrics.json",
    }

    names, r2s, categories = [], [], []
    for name, path in models_direct.items():
        if path.exists():
            m = load_metrics(path)
            names.append(name)
            r2s.append(m["r2"])
            categories.append("Direct")
    for name, path in models_recursive.items():
        if path.exists():
            m = load_metrics(path)
            names.append(name)
            r2s.append(m["r2"])
            categories.append("Recursive")

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#d62728" if c == "Direct" else "#2ca02c" for c in categories]
    bars = ax.barh(names, r2s, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("R² Score", fontsize=13)
    ax.set_title("24-Hour Forecast: Direct vs Recursive Approach\n(The Breakthrough)",
                 fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.axvline(x=0, color="black", linewidth=0.5)

    for bar, val in zip(bars, r2s):
        ax.text(max(bar.get_width() + 0.01, 0.02), bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=10, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#d62728", label="Direct (single-shot)"),
                       Patch(facecolor="#2ca02c", label="Recursive (iterative h=1)")]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=11)

    plt.tight_layout()
    plt.savefig(REPORT_FIGS / "fig2_h24_breakthrough.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Fig 2: h=24 breakthrough saved")


# ────────────────────────────────────────────────────────────────────
#  Figure 3: Per-step Error Growth (Recursive h=24)
# ────────────────────────────────────────────────────────────────────
def fig3_per_step_error():
    # Load recursive h=24 predictions to compute per-step RMSE
    pred_path = RESULTS / "preproc_E_LSTM_recursive_h24_20260221_232807/predictions.npz"
    if not pred_path.exists():
        print("  Fig 3: Skipped (predictions not found)")
        return

    data = np.load(pred_path)
    y_true = data["y_true"]  # (n_samples, 24)
    y_pred = data["y_pred"]

    n_steps = y_true.shape[1]
    per_step_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    per_step_mae = np.mean(np.abs(y_true - y_pred), axis=0)
    per_step_r2 = []
    for t in range(n_steps):
        ss_res = np.sum((y_true[:, t] - y_pred[:, t]) ** 2)
        ss_tot = np.sum((y_true[:, t] - np.mean(y_true[:, t])) ** 2)
        per_step_r2.append(1 - ss_res / ss_tot)

    steps = np.arange(1, n_steps + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(steps, per_step_rmse, "o-", color="#1f77b4", label="RMSE", linewidth=2)
    ax1.plot(steps, per_step_mae, "s-", color="#ff7f0e", label="MAE", linewidth=2)
    ax1.set_xlabel("Forecast Step (hours ahead)", fontsize=12)
    ax1.set_ylabel("Error (ppm CO₂)", fontsize=12)
    ax1.set_title("Recursive LSTM: Error Growth Over Horizon", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.set_xticks(steps)

    ax2.plot(steps, per_step_r2, "D-", color="#2ca02c", linewidth=2)
    ax2.set_xlabel("Forecast Step (hours ahead)", fontsize=12)
    ax2.set_ylabel("R² Score", fontsize=12)
    ax2.set_title("Recursive LSTM: R² Decay Over Horizon", fontsize=13, fontweight="bold")
    ax2.set_xticks(steps)
    ax2.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="R²=0.95")
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(REPORT_FIGS / "fig3_per_step_error.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Fig 3: Per-step error growth saved")


# ────────────────────────────────────────────────────────────────────
#  Figure 4: Conformal Prediction Intervals
# ────────────────────────────────────────────────────────────────────
def fig4_conformal():
    # Load h=24 conformal results
    conf_dir = RESULTS / "preproc_E_Seq2Seq_h24_20260221_215838/conformal"
    if not conf_dir.exists():
        print("  Fig 4: Skipped (conformal not found)")
        return

    with open(conf_dir / "coverage.json", encoding="utf-8") as f:
        coverage = json.load(f)

    per_step_cov = coverage.get("per_step_coverage", [])
    per_step_width = coverage.get("per_step_width", [])

    if not per_step_cov:
        print("  Fig 4: Skipped (no per-step data)")
        return

    steps = np.arange(1, len(per_step_cov) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(steps, [c * 100 for c in per_step_cov], color="#2ca02c", alpha=0.8)
    ax1.axhline(y=90, color="red", linestyle="--", linewidth=2, label="Target: 90%")
    ax1.set_xlabel("Forecast Step (hours ahead)", fontsize=12)
    ax1.set_ylabel("Empirical Coverage (%)", fontsize=12)
    ax1.set_title("Conformal Prediction: Coverage per Step", fontsize=13, fontweight="bold")
    ax1.set_ylim(85, 100)
    ax1.legend(fontsize=11)

    ax2.plot(steps, per_step_width, "o-", color="#d62728", linewidth=2)
    ax2.set_xlabel("Forecast Step (hours ahead)", fontsize=12)
    ax2.set_ylabel("Interval Width (ppm CO₂)", fontsize=12)
    ax2.set_title("Conformal Prediction: Interval Width Growth", fontsize=13, fontweight="bold")
    ax2.fill_between(steps, 0, per_step_width, alpha=0.2, color="#d62728")

    plt.tight_layout()
    plt.savefig(REPORT_FIGS / "fig4_conformal_h24.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Fig 4: Conformal prediction saved")


# ────────────────────────────────────────────────────────────────────
#  Figure 5: Ensemble Weight Distribution
# ────────────────────────────────────────────────────────────────────
def fig5_ensemble_weights():
    ens_path = RESULTS / "ensemble_preproc_E_h1/weighted_ensemble.json"
    if not ens_path.exists():
        print("  Fig 5: Skipped (ensemble not found)")
        return

    with open(ens_path, encoding="utf-8") as f:
        ens = json.load(f)

    weights = ens["weights"]
    short_names = {
        "preproc_E_Seq2Seq_h1_20260221_215359": "Seq2Seq",
        "preproc_E_LSTM_h1_20260221_220750": "LSTM",
        "preproc_E_XGBoost_h1_20260221_221107": "XGBoost",
    }
    labels = [short_names.get(k, k) for k in weights.keys()]
    values = list(weights.values())

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct="%1.1f%%",
        colors=["#1f77b4", "#ff7f0e", "#2ca02c"],
        startangle=90, textprops={"fontsize": 13},
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    for t in autotexts:
        t.set_fontweight("bold")
    ax.set_title("Weighted Ensemble: Model Contributions\n(h=1 Forecast)",
                 fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(REPORT_FIGS / "fig5_ensemble_weights.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Fig 5: Ensemble weights saved")


# ────────────────────────────────────────────────────────────────────
#  Figure 6: Optuna Convergence
# ────────────────────────────────────────────────────────────────────
def fig6_optuna():
    optuna_path = RESULTS / "optuna/lstm/lstm_h1_best_params.json"
    if not optuna_path.exists():
        print("  Fig 6: Skipped (optuna results not found)")
        return

    with open(optuna_path, encoding="utf-8") as f:
        optuna_results = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 5))

    params = optuna_results["best_params"]
    param_names = list(params.keys())
    param_values = [str(round(v, 4)) if isinstance(v, float) else str(v) for v in params.values()]

    # Create a table-style visualization
    cell_text = [[v] for v in param_values]
    table = ax.table(
        cellText=cell_text,
        rowLabels=param_names,
        colLabels=["Best Value"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(0.8, 2.0)
    ax.axis("off")
    ax.set_title(
        f"Optuna Best Hyperparameters\n"
        f"Best val_loss = {optuna_results['best_value']:.6f} | "
        f"{optuna_results['n_trials']} trials ({optuna_results['n_complete']} complete, "
        f"{optuna_results['n_pruned']} pruned)",
        fontsize=13, fontweight="bold", pad=20,
    )

    plt.tight_layout()
    plt.savefig(REPORT_FIGS / "fig6_optuna.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Fig 6: Optuna results saved")


# ────────────────────────────────────────────────────────────────────
#  Figure 7: Sample Prediction Overlay (h=1 best and h=24 recursive)
# ────────────────────────────────────────────────────────────────────
def fig7_prediction_samples():
    # h=1 best (LSTM lookback=36)
    h1_path = RESULTS / "preproc_E_LSTM_h1_20260221_232331/predictions.npz"
    # h=24 recursive best
    h24_path = RESULTS / "preproc_E_LSTM_recursive_h24_20260221_232807/predictions.npz"

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    if h1_path.exists():
        data = np.load(h1_path)
        y_true = data["y_true"][:200, 0]
        y_pred = data["y_pred"][:200, 0]
        ax = axes[0]
        ax.plot(y_true, color="#1f77b4", label="Actual CO₂", linewidth=1.5, alpha=0.8)
        ax.plot(y_pred, color="#d62728", label="Predicted CO₂", linewidth=1.5, alpha=0.8)
        ax.set_title("h=1 Forecast: LSTM+Occupancy (lookback=36h) — R²=0.9644",
                     fontsize=13, fontweight="bold")
        ax.set_ylabel("CO₂ (ppm)", fontsize=12)
        ax.legend(fontsize=11, loc="upper right")
        ax.set_xlabel("Sample Index (test set)", fontsize=11)

    if h24_path.exists():
        data = np.load(h24_path)
        # Show step 1, 6, 12, 24 predictions
        y_true_all = data["y_true"][:200]
        y_pred_all = data["y_pred"][:200]
        ax = axes[1]
        ax.plot(y_true_all[:, 0], color="#1f77b4", label="Actual (t+1h)", linewidth=1.5, alpha=0.8)
        ax.plot(y_pred_all[:, 0], color="#2ca02c", label="Pred (t+1h)", linewidth=1, alpha=0.7, linestyle="--")
        ax.plot(y_true_all[:, 11], color="#1f77b4", linewidth=1.5, alpha=0.5)
        ax.plot(y_pred_all[:, 11], color="#ff7f0e", label="Pred (t+12h)", linewidth=1, alpha=0.7, linestyle="--")
        ax.plot(y_true_all[:, 23], color="#1f77b4", linewidth=1.5, alpha=0.3)
        ax.plot(y_pred_all[:, 23], color="#d62728", label="Pred (t+24h)", linewidth=1, alpha=0.7, linestyle="--")
        ax.set_title("h=24 Recursive Forecast: LSTM+Occupancy (lookback=36h) — R²=0.9582",
                     fontsize=13, fontweight="bold")
        ax.set_ylabel("CO₂ (ppm)", fontsize=12)
        ax.set_xlabel("Sample Index (test set)", fontsize=11)
        ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    plt.savefig(REPORT_FIGS / "fig7_prediction_samples.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Fig 7: Prediction samples saved")


# ────────────────────────────────────────────────────────────────────
#  Figure 8: Comprehensive Results Summary Table
# ────────────────────────────────────────────────────────────────────
def fig8_summary_table():
    rows = []
    all_models = [
        ("SARIMA", "h=1", "sarima_interpretability/h1/metrics.json"),
        ("CatBoost", "h=1", "catboost_interpretability/h1/metrics.json"),
        ("XGBoost", "h=1", "xgboost_interpretability/h1/metrics.json"),
        ("CNN-LSTM", "h=1", "cnn_lstm_interpretability/h1/metrics.json"),
        ("TFT", "h=1", "tft_interpretability/h1/metrics.json"),
        ("LSTM (baseline)", "h=1", "lstm_interpretability/h1/metrics.json"),
        ("Seq2Seq+Occ", "h=1", "preproc_E_Seq2Seq_h1_20260221_215359/metrics.json"),
        ("LSTM+Occ (lb=36)", "h=1", "preproc_E_LSTM_h1_20260221_232331/metrics.json"),
        ("LSTM+Wavelet", "h=1", "preproc_F_LSTM_h1_20260221_224757/metrics.json"),
        ("Weighted Ensemble", "h=1", "ensemble_preproc_E_h1/weighted_metrics.json"),
        ("LSTM (baseline)", "h=24", "lstm_interpretability/h24/metrics.json"),
        ("CatBoost", "h=24", "catboost_interpretability/h24/metrics.json"),
        ("Seq2Seq+Occ (direct)", "h=24", "preproc_E_Seq2Seq_h24_20260221_215838/metrics.json"),
        ("LSTM recursive (lb=24)", "h=24", "preproc_E_LSTM_recursive_h24_20260221_224729/metrics.json"),
        ("LSTM+Wav recursive", "h=24", "preproc_F_LSTM_recursive_h24_20260221_225042/metrics.json"),
        ("LSTM recursive (lb=36)", "h=24", "preproc_E_LSTM_recursive_h24_20260221_232807/metrics.json"),
    ]

    for name, horizon, path in all_models:
        full_path = RESULTS / path
        if full_path.exists():
            m = load_metrics(full_path)
            rows.append([name, horizon, f"{m['rmse']:.2f}", f"{m['mae']:.2f}",
                        f"{m['r2']:.4f}", f"{m['mape']:.2f}%"])

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Model", "Horizon", "RMSE", "MAE", "R²", "MAPE"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Color header
    for j in range(6):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Color best results
    for i, row in enumerate(rows):
        if "Ensemble" in row[0] and row[1] == "h=1":
            for j in range(6):
                table[i + 1, j].set_facecolor("#E2EFDA")
        elif "recursive (lb=36)" in row[0]:
            for j in range(6):
                table[i + 1, j].set_facecolor("#E2EFDA")

    ax.set_title("Complete Experimental Results Summary",
                 fontsize=15, fontweight="bold", pad=30)

    plt.tight_layout()
    plt.savefig(REPORT_FIGS / "fig8_summary_table.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Fig 8: Summary table saved")


if __name__ == "__main__":
    print("Generating report figures...")
    fig1_h1_comparison()
    fig2_h24_breakthrough()
    fig3_per_step_error()
    fig4_conformal()
    fig5_ensemble_weights()
    fig6_optuna()
    fig7_prediction_samples()
    fig8_summary_table()
    print(f"\nAll figures saved to: {REPORT_FIGS}")
