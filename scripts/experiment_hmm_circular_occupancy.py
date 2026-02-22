"""Experiment: Mixed Gaussian + von Mises HMM vs Gaussian HMM for occupancy.

Compares three approaches for incorporating temporal features into HMM:
  A. Gaussian HMM on CO2+dCO2+Noise (baseline — no temporal features)
  B. Gaussian HMM on CO2+dCO2+Noise+sin/cos (linear treatment of circular data)
  C. Mixed Gaussian+vonMises HMM (proper circular modeling of time)

The hypothesis: the standard Gaussian HMM fails with sin/cos features because
it treats them as independent linear variables. A mixed-emission HMM that uses
von Mises distributions for angular features should let the model learn
time-conditioned occupancy states without "drowning out" the CO2 signal.

Von Mises PDF:  f(theta; mu, kappa) = exp(kappa*cos(theta-mu)) / (2*pi*I_0(kappa))
  - Properly wraps around the circle (midnight = 0h = 24h)
  - kappa controls concentration (0 = uniform, large = tight peak)
  - EM M-step uses circular sufficient statistics (weighted cos/sin sums)

Usage:
    python scripts/experiment_hmm_circular_occupancy.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import i0
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hmm_lstm import HMMRegimeDetector
from src.models.mixed_hmm import MixedHMMRegimeDetector
from src.data.pipeline import run_preprocessing_pipeline
from src.occupancy.detectors import run_all_detectors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(*configs: dict) -> dict:
    result: dict = {}
    for cfg in configs:
        for key, value in cfg.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
    return result


def compute_kappa_safe(y1: np.ndarray, y2: np.ndarray) -> float:
    try:
        k = cohen_kappa_score(y1, y2)
        return 0.0 if np.isnan(k) else float(k)
    except Exception:
        return 0.0


def find_best_binary_mapping(
    hmm_states: np.ndarray,
    consensus: np.ndarray,
    n_states: int,
) -> tuple[list[int], float, np.ndarray]:
    """Find the binary state→occupied mapping that maximizes kappa vs consensus.

    Exhaustively tests all 2^n_states - 2 non-trivial binary mappings.

    Returns:
        (occupied_states, best_kappa, hmm_binary_best)
    """
    best_kappa = -2.0
    best_mapping: list[int] = []
    for bitmask in range(1, 2**n_states - 1):
        occ_states = [s for s in range(n_states) if bitmask & (1 << s)]
        hmm_binary = np.isin(hmm_states, occ_states).astype(int)
        k = compute_kappa_safe(consensus, hmm_binary)
        if k > best_kappa:
            best_kappa = k
            best_mapping = occ_states

    hmm_binary_best = np.isin(hmm_states, best_mapping).astype(int)
    return best_mapping, best_kappa, hmm_binary_best


def run_experiment(
    name: str,
    hmm_states: np.ndarray,
    n_states: int,
    occupancy_results: dict[str, np.ndarray],
    consensus: np.ndarray,
    test_df: pd.DataFrame,
    feature_names: list[str],
) -> dict:
    """Analyze HMM states vs occupancy for one experiment configuration."""

    # State profiles
    profiles = []
    for s in range(n_states):
        mask = hmm_states == s
        count = int(mask.sum())
        pct = float(count / len(hmm_states) * 100)
        means = {}
        for feat in ["CO2", "dCO2", "Noise"]:
            if feat in test_df.columns:
                means[feat] = float(test_df.loc[mask, feat].mean()) if count > 0 else 0.0

        consensus_occ = float(consensus[mask].mean() * 100) if count > 0 else 0.0
        profiles.append({
            "state": s, "count": count, "pct": round(pct, 1),
            "means": {k: round(v, 2) for k, v in means.items()},
            "consensus_occ_pct": round(consensus_occ, 1),
        })

    # Best binary mapping
    best_mapping, best_kappa, hmm_binary = find_best_binary_mapping(
        hmm_states, consensus, n_states
    )

    # Per-detector kappas
    kappas = {}
    for det_name, det_arr in occupancy_results.items():
        kappas[det_name] = round(compute_kappa_safe(det_arr, hmm_binary), 4)
    kappas["consensus"] = round(best_kappa, 4)

    return {
        "name": name,
        "n_states": n_states,
        "features": feature_names,
        "state_profiles": profiles,
        "best_binary_mapping": {
            "occupied_states": best_mapping,
            "kappa_vs_consensus": round(best_kappa, 4),
            "occupancy_rate": round(float(hmm_binary.mean() * 100), 1),
        },
        "kappas": kappas,
    }


def main() -> None:
    # ── Load data ─────────────────────────────────────────────────────
    data_config = load_config(
        PROJECT_ROOT / "configs" / "experiments" / "preproc_E_occupancy_1h.yaml"
    )
    base_data_cfg = load_config(PROJECT_ROOT / "configs" / "data.yaml")
    base_training_cfg = load_config(PROJECT_ROOT / "configs" / "training.yaml")
    full_config = merge_configs(base_data_cfg, base_training_cfg, data_config)

    occ_config = load_config(PROJECT_ROOT / "configs" / "occupancy.yaml")
    detector_config = occ_config.get("detectors", occ_config)

    raw_dir = Path(full_config["data"].get("raw_dir", "data/raw"))
    train_df, val_df, test_df = run_preprocessing_pipeline(
        raw_dir=raw_dir, variant_config=full_config
    )

    if "dCO2" not in test_df.columns:
        test_df["dCO2"] = test_df["CO2"].diff().fillna(0.0)
    if "dCO2" not in train_df.columns:
        train_df["dCO2"] = train_df["CO2"].diff().fillna(0.0)

    # ── Run occupancy detectors ───────────────────────────────────────
    results_actual = run_all_detectors(test_df, detector_config, train_df=train_df)
    det_matrix = np.column_stack(list(results_actual.values()))
    consensus = (det_matrix.mean(axis=1) >= 0.5).astype(int)

    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)} rows")
    logger.info(f"Consensus occupancy: {consensus.mean():.1%}")

    # ── Define experiments ────────────────────────────────────────────
    all_results = {}

    print(f"\n{'='*75}")
    print(f"  GAUSSIAN vs MIXED (Gaussian+vonMises) HMM for OCCUPANCY DETECTION")
    print(f"{'='*75}")

    # === GROUP A: Baselines (pure Gaussian HMM) =======================

    # A1: CO2+dCO2+Noise (best from prior experiments)
    for n_st in [4]:
        exp_name = f"A1: Gaussian CO2+dCO2+Noise ({n_st}st)"
        hmm_a1 = HMMRegimeDetector(
            n_states=n_st, hmm_features=["CO2", "dCO2", "Noise"],
            n_iter=200, covariance_type="full",
        )
        hmm_a1.fit(train_df)
        states_a1 = hmm_a1.predict_states(test_df)
        all_results[exp_name] = run_experiment(
            exp_name, states_a1, n_st, results_actual, consensus,
            test_df, ["CO2", "dCO2", "Noise"],
        )

    # A2: CO2+dCO2+Noise+Day_sin/cos (sin/cos as Gaussian — expected to fail)
    for n_st in [4]:
        exp_name = f"A2: Gaussian +Day_sin/cos ({n_st}st)"
        hmm_a2 = HMMRegimeDetector(
            n_states=n_st,
            hmm_features=["CO2", "dCO2", "Noise", "Day_sin", "Day_cos"],
            n_iter=200, covariance_type="full",
        )
        hmm_a2.fit(train_df)
        states_a2 = hmm_a2.predict_states(test_df)
        all_results[exp_name] = run_experiment(
            exp_name, states_a2, n_st, results_actual, consensus,
            test_df, ["CO2", "dCO2", "Noise", "Day_sin", "Day_cos"],
        )

    # A3: Gaussian + Day + Weekday sin/cos
    for n_st in [4]:
        exp_name = f"A3: Gaussian +Day+Weekday sin/cos ({n_st}st)"
        hmm_a3 = HMMRegimeDetector(
            n_states=n_st,
            hmm_features=[
                "CO2", "dCO2", "Noise",
                "Day_sin", "Day_cos", "Weekday_sin", "Weekday_cos",
            ],
            n_iter=200, covariance_type="full",
        )
        hmm_a3.fit(train_df)
        states_a3 = hmm_a3.predict_states(test_df)
        all_results[exp_name] = run_experiment(
            exp_name, states_a3, n_st, results_actual, consensus,
            test_df, ["CO2", "dCO2", "Noise", "Day_sin", "Day_cos",
                       "Weekday_sin", "Weekday_cos"],
        )

    # === GROUP B: Mixed Gaussian + von Mises HMM =======================

    # B1: CO2+dCO2+Noise (Gaussian) + hour_angle (von Mises)
    for n_st in [4, 5, 6]:
        exp_name = f"B1: Mixed +hour_angle ({n_st}st)"
        hmm_b1 = MixedHMMRegimeDetector(
            n_states=n_st,
            gaussian_features=["CO2", "dCO2", "Noise"],
            circular_features=[{"column": "hour", "period": 24}],
            n_iter=200,
        )
        hmm_b1.fit(train_df)
        states_b1 = hmm_b1.predict_states(test_df)
        all_results[exp_name] = run_experiment(
            exp_name, states_b1, n_st, results_actual, consensus,
            test_df, ["CO2", "dCO2", "Noise", "hour(VM)"],
        )

    # B2: CO2+dCO2+Noise (Gaussian) + hour + dayofweek (von Mises)
    for n_st in [4, 5, 6]:
        exp_name = f"B2: Mixed +hour+dow ({n_st}st)"
        hmm_b2 = MixedHMMRegimeDetector(
            n_states=n_st,
            gaussian_features=["CO2", "dCO2", "Noise"],
            circular_features=[
                {"column": "hour", "period": 24},
                {"column": "dayofweek", "period": 7},
            ],
            n_iter=200,
        )
        hmm_b2.fit(train_df)
        states_b2 = hmm_b2.predict_states(test_df)
        all_results[exp_name] = run_experiment(
            exp_name, states_b2, n_st, results_actual, consensus,
            test_df, ["CO2", "dCO2", "Noise", "hour(VM)", "dow(VM)"],
        )

    # B3: CO2+dCO2+Noise (Gaussian) + hour only (von Mises), more states
    for n_st in [8]:
        exp_name = f"B3: Mixed +hour ({n_st}st, more granular)"
        hmm_b3 = MixedHMMRegimeDetector(
            n_states=n_st,
            gaussian_features=["CO2", "dCO2", "Noise"],
            circular_features=[{"column": "hour", "period": 24}],
            n_iter=200,
        )
        hmm_b3.fit(train_df)
        states_b3 = hmm_b3.predict_states(test_df)
        all_results[exp_name] = run_experiment(
            exp_name, states_b3, n_st, results_actual, consensus,
            test_df, ["CO2", "dCO2", "Noise", "hour(VM)"],
        )

    # === Print results ═════════════════════════════════════════════════
    for exp_name, res in all_results.items():
        bm = res["best_binary_mapping"]
        print(f"\n{'─'*75}")
        print(f"  {exp_name}")
        print(f"  Features: {res['features']}")
        print(f"{'─'*75}")

        print(f"\n  {'State':>5}  {'Count':>6}  {'%':>5}  {'CO2':>8}  {'dCO2':>8}  "
              f"{'Noise':>7}  {'Consensus Occ%':>15}")
        for sp in res["state_profiles"]:
            m = sp["means"]
            print(
                f"  {sp['state']:>5d}  {sp['count']:>6d}  {sp['pct']:>5.1f}  "
                f"{m.get('CO2', 0):>8.1f}  {m.get('dCO2', 0):>8.2f}  "
                f"{m.get('Noise', 0):>7.1f}  {sp['consensus_occ_pct']:>14.1f}%"
            )

        print(f"\n  Best mapping: states {bm['occupied_states']} -> occupied")
        print(f"  kappa vs consensus: {bm['kappa_vs_consensus']:.4f}")
        print(f"  HMM occupancy rate: {bm['occupancy_rate']:.1f}%")

        print(f"\n  Per-detector kappas:")
        for det, k in res["kappas"].items():
            print(f"    {det:25s}: {k:.4f}")

    # === Summary table ═════════════════════════════════════════════════
    print(f"\n\n{'='*75}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*75}")
    print(f"\n  {'Experiment':48s}  {'kappa':>7}  {'Occ%':>6}  {'Mapping':>14}")
    print(f"  {'─'*82}")

    # Sort by kappa descending
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]["best_binary_mapping"]["kappa_vs_consensus"],
        reverse=True,
    )

    for exp_name, res in sorted_results:
        bm = res["best_binary_mapping"]
        mapping_str = str(bm["occupied_states"])
        print(
            f"  {exp_name:48s}  {bm['kappa_vs_consensus']:>7.4f}  "
            f"{bm['occupancy_rate']:>5.1f}%  {mapping_str:>14}"
        )

    # ── Save results ──────────────────────────────────────────────────
    output_dir = PROJECT_ROOT / "results" / "hmm_circular_vs_gaussian"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "experiment_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ── Visualization ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # Split results into Gaussian vs Mixed groups
    gauss_names = [n for n in all_results if n.startswith("A")]
    mixed_names = [n for n in all_results if n.startswith("B")]

    all_names_sorted = [n for n, _ in sorted_results]
    kappas_sorted = [res["best_binary_mapping"]["kappa_vs_consensus"]
                     for _, res in sorted_results]

    # Color by group
    colors = []
    for name in all_names_sorted:
        if name.startswith("A"):
            colors.append("#e74c3c" if "sin/cos" in name else "#3498db")
        else:
            colors.append("#2ecc71")

    # Short labels
    short_labels = []
    for name in all_names_sorted:
        short_labels.append(name.split(":")[1].strip() if ":" in name else name)

    bars = axes[0].barh(short_labels, kappas_sorted, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_xlabel("Cohen's kappa vs Consensus", fontsize=12)
    axes[0].set_title("HMM Occupancy Alignment: Gaussian vs Mixed (Gaussian+vonMises)", fontsize=13)
    axes[0].axvline(x=0, color="black", linewidth=0.5)
    for bar, val in zip(bars, kappas_sorted):
        axes[0].text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9,
        )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3498db", label="Gaussian (no temporal)"),
        Patch(facecolor="#e74c3c", label="Gaussian (sin/cos temporal)"),
        Patch(facecolor="#2ecc71", label="Mixed Gauss+vonMises (circular temporal)"),
    ]
    axes[0].legend(handles=legend_elements, loc="lower right", fontsize=9)

    # Plot 2: Per-detector kappas for top experiments
    top_n = min(6, len(sorted_results))
    top_results = sorted_results[:top_n]
    detector_names = list(top_results[0][1]["kappas"].keys())
    x = np.arange(len(detector_names))
    width = 0.8 / top_n

    for i, (exp_name, res) in enumerate(top_results):
        kvals = [res["kappas"].get(d, 0) for d in detector_names]
        short_name = exp_name.split(":")[1].strip() if ":" in exp_name else exp_name
        color = "#3498db" if exp_name.startswith("A1") else (
            "#e74c3c" if exp_name.startswith("A") else "#2ecc71"
        )
        axes[1].bar(
            x + i * width - 0.4 + width/2, kvals, width,
            label=short_name[:25],
            color=color, alpha=0.7 + 0.05*i,
            edgecolor="black", linewidth=0.3,
        )

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(detector_names, rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("Cohen's kappa", fontsize=11)
    axes[1].set_title("Per-Detector kappa (Top Configurations)", fontsize=13)
    axes[1].legend(fontsize=7, loc="upper left", ncol=2)
    axes[1].axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    fig_path = output_dir / "gaussian_vs_mixed_hmm.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Figure saved to {fig_path}")

    # ── Von Mises state interpretation figure ─────────────────────────
    # For the best mixed-emission experiment, show the learned circular
    # means (converted to hours) and concentrations
    best_mixed_name = None
    best_mixed_kappa = -2.0
    for name, res in all_results.items():
        if name.startswith("B"):
            k = res["best_binary_mapping"]["kappa_vs_consensus"]
            if k > best_mixed_kappa:
                best_mixed_kappa = k
                best_mixed_name = name

    if best_mixed_name:
        best_res = all_results[best_mixed_name]
        n_st = best_res["n_states"]

        # Re-fit to extract von Mises parameters
        # Parse config from the experiment name
        if "+hour+dow" in best_mixed_name:
            circ_feats = [
                {"column": "hour", "period": 24},
                {"column": "dayofweek", "period": 7},
            ]
        else:
            circ_feats = [{"column": "hour", "period": 24}]

        hmm_refit = MixedHMMRegimeDetector(
            n_states=n_st,
            gaussian_features=["CO2", "dCO2", "Noise"],
            circular_features=circ_feats,
            n_iter=200,
        )
        hmm_refit.fit(train_df)
        refit_states = hmm_refit.predict_states(test_df)

        # Extract parameters
        vm_mus = hmm_refit.hmm.vm_mus_     # (n_states, n_circular)
        vm_kappas = hmm_refit.hmm.vm_kappas_
        g_means = hmm_refit.hmm.gauss_means_  # (n_states, 3) for CO2, dCO2, Noise

        # Convert hour angle back to hours: hour = (mu + pi) / (2*pi) * 24
        hour_means = (vm_mus[:, 0] + np.pi) / (2 * np.pi) * 24.0
        hour_kappas = vm_kappas[:, 0]

        # Polar plot of von Mises hour distributions
        fig, axes_vm = plt.subplots(1, 2, figsize=(14, 6),
                                     subplot_kw={"projection": "polar"} if True else {})

        # Need a fresh figure with mixed subplot types
        fig = plt.figure(figsize=(16, 6))
        ax_polar = fig.add_subplot(121, projection="polar")
        ax_bar = fig.add_subplot(122)

        # Polar plot: von Mises concentration direction for each state
        theta_grid = np.linspace(-np.pi, np.pi, 500)
        state_colors = plt.cm.Set1(np.linspace(0, 1, n_st))

        for s in range(n_st):
            mu_s = vm_mus[s, 0]
            kappa_s = vm_kappas[s, 0]
            # Von Mises PDF on the circle
            pdf_vals = np.exp(kappa_s * np.cos(theta_grid - mu_s)) / (
                2 * np.pi * i0(kappa_s)
            )
            # Convert theta to "clock angle" (0=midnight at top, clockwise)
            clock_theta = (theta_grid + np.pi) / (2 * np.pi) * 2 * np.pi
            occ_pct = best_res["state_profiles"][s]["consensus_occ_pct"]
            co2_mean = best_res["state_profiles"][s]["means"].get("CO2", 0)
            ax_polar.plot(
                clock_theta, pdf_vals,
                color=state_colors[s], linewidth=2,
                label=f"S{s}: CO2={co2_mean:.0f}, occ={occ_pct:.0f}%"
            )
            # Arrow at the mean direction
            mu_clock = (mu_s + np.pi) / (2 * np.pi) * 2 * np.pi
            ax_polar.annotate(
                "", xy=(mu_clock, max(pdf_vals) * 0.8),
                xytext=(0, 0),
                arrowprops=dict(
                    arrowstyle="->", color=state_colors[s], lw=2
                ),
            )

        # Set hour labels (clockwise, 0h at top)
        ax_polar.set_theta_zero_location("N")
        ax_polar.set_theta_direction(-1)  # clockwise
        hour_ticks = np.linspace(0, 2*np.pi, 24, endpoint=False)
        ax_polar.set_xticks(hour_ticks)
        ax_polar.set_xticklabels([f"{h:02d}h" for h in range(24)], fontsize=7)
        ax_polar.set_title(f"Von Mises Hour Distributions\n{best_mixed_name}", fontsize=11, pad=20)
        ax_polar.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1.0))

        # Bar chart: Gaussian means per state
        x_states = np.arange(n_st)
        bar_width = 0.25
        ax_bar.bar(x_states - bar_width, g_means[:, 0], bar_width,
                    label="CO2 (ppm)", color="#3498db", edgecolor="black", linewidth=0.5)
        ax_bar.bar(x_states, g_means[:, 1] * 10, bar_width,  # scale dCO2 for visibility
                    label="dCO2 (ppm/h × 10)", color="#e74c3c", edgecolor="black", linewidth=0.5)
        ax_bar.bar(x_states + bar_width, g_means[:, 2] * 10, bar_width,  # scale Noise
                    label="Noise (dB × 10)", color="#2ecc71", edgecolor="black", linewidth=0.5)

        ax_bar.set_xticks(x_states)
        state_labels = []
        for s in range(n_st):
            h = hour_means[s]
            k = hour_kappas[s]
            state_labels.append(f"S{s}\nhour={h:.1f}\nkappa={k:.1f}")
        ax_bar.set_xticklabels(state_labels, fontsize=8)
        ax_bar.set_ylabel("Parameter Value (scaled)")
        ax_bar.set_title(f"Gaussian Emission Means per State\n{best_mixed_name}", fontsize=11)
        ax_bar.legend(fontsize=8)
        ax_bar.axhline(y=0, color="black", linewidth=0.5)

        plt.tight_layout()
        fig_path2 = output_dir / "mixed_hmm_state_interpretation.png"
        plt.savefig(fig_path2, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Interpretation figure saved to {fig_path2}")

    print(f"\n  Results saved to: {output_dir}")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
