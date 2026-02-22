"""Experiment: HMM with temporal features vs occupancy detectors.

Tests whether adding temporal (cyclical time) features to the HMM observation
model helps it discover states that align with occupancy patterns rather than
seasonal temperature regimes.

Experiments:
  1. CO2 + dCO2 + Noise (baseline — best from prior experiments)
  2. CO2 + dCO2 + Noise + Day_sin/cos (24h cycle)
  3. CO2 + dCO2 + Noise + Day_sin/cos + Weekday_sin/cos (24h + 7d cycles)
  4. CO2 + dCO2 + Noise + hour_weekday_sin/cos (168h combined cycle)
  5. CO2 + dCO2 + Noise + Day_sin/cos + Weekday_sin/cos + Year_sin/cos (all)

Each experiment fits HMM on train, decodes test states via Viterbi, then
computes Cohen's kappa against each of the 6 occupancy detectors and the
majority-vote consensus label.

Usage:
    python scripts/experiment_hmm_temporal_occupancy.py
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
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hmm_lstm import HMMRegimeDetector
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
    """Cohen's kappa with NaN/constant-output protection."""
    try:
        k = cohen_kappa_score(y1, y2)
        return 0.0 if np.isnan(k) else float(k)
    except Exception:
        return 0.0


def analyze_hmm_states_vs_occupancy(
    hmm_states: np.ndarray,
    occupancy_results: dict[str, np.ndarray],
    consensus_label: np.ndarray,
    n_states: int,
    feature_names: list[str],
    test_df: pd.DataFrame,
) -> dict:
    """Compare HMM states against occupancy detectors.

    For each HMM state, computes:
    - Feature means (CO2, dCO2, Noise, temporal features)
    - % of timesteps classified as occupied by consensus
    - Cohen's kappa per detector (treating each HMM state as binary occupied)

    Returns a dict with state profiles and kappa scores.
    """
    result = {"n_states": n_states, "features": feature_names}

    # State profiles
    state_profiles = []
    for s in range(n_states):
        mask = hmm_states == s
        count = int(mask.sum())
        pct = float(count / len(hmm_states) * 100)

        # Feature means for this state
        means = {}
        for feat in feature_names:
            if feat in test_df.columns:
                means[feat] = float(test_df.loc[mask, feat].mean())

        # What fraction of this state is consensus-occupied?
        consensus_occ_pct = float(consensus_label[mask].mean() * 100) if count > 0 else 0.0

        # Per-detector occupancy within this state
        detector_occ = {}
        for det_name, det_arr in occupancy_results.items():
            detector_occ[det_name] = float(det_arr[mask].mean() * 100) if count > 0 else 0.0

        state_profiles.append({
            "state": s,
            "count": count,
            "pct": round(pct, 1),
            "means": {k: round(v, 2) for k, v in means.items()},
            "consensus_occupied_pct": round(consensus_occ_pct, 1),
            "detector_occupied_pct": {k: round(v, 1) for k, v in detector_occ.items()},
        })

    result["state_profiles"] = state_profiles

    # Best binary mapping: which HMM states should be labelled "occupied"?
    # Try all 2^n_states - 2 non-trivial binary mappings and pick the one
    # with highest kappa against consensus.
    best_kappa = -2.0
    best_mapping = []
    for bitmask in range(1, 2**n_states - 1):
        occ_states = [s for s in range(n_states) if bitmask & (1 << s)]
        hmm_binary = np.isin(hmm_states, occ_states).astype(int)
        k = compute_kappa_safe(consensus_label, hmm_binary)
        if k > best_kappa:
            best_kappa = k
            best_mapping = occ_states

    # Apply best mapping
    hmm_binary_best = np.isin(hmm_states, best_mapping).astype(int)
    result["best_binary_mapping"] = {
        "occupied_states": best_mapping,
        "kappa_vs_consensus": round(best_kappa, 4),
        "occupancy_rate": round(float(hmm_binary_best.mean() * 100), 1),
    }

    # Kappa per detector using best mapping
    kappas = {}
    for det_name, det_arr in occupancy_results.items():
        kappas[det_name] = round(compute_kappa_safe(det_arr, hmm_binary_best), 4)
    kappas["consensus"] = round(best_kappa, 4)
    result["kappas"] = kappas

    return result


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

    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)} rows")
    logger.info(f"Available columns: {test_df.columns.tolist()}")

    # ── Run occupancy detectors on actual CO2 ─────────────────────────
    if "dCO2" not in test_df.columns:
        test_df["dCO2"] = test_df["CO2"].diff().fillna(0.0)
    if "dCO2" not in train_df.columns:
        train_df["dCO2"] = train_df["CO2"].diff().fillna(0.0)

    results_actual = run_all_detectors(test_df, detector_config, train_df=train_df)

    # Compute majority-vote consensus
    det_matrix = np.column_stack(list(results_actual.values()))
    consensus = (det_matrix.mean(axis=1) >= 0.5).astype(int)
    consensus_rate = float(consensus.mean())
    logger.info(f"Consensus occupancy rate: {consensus_rate:.1%}")

    # ── Define experiments ────────────────────────────────────────────
    experiments = {
        "Exp0: CO2+dCO2+Noise (baseline, 4 states)": {
            "features": ["CO2", "dCO2", "Noise"],
            "n_states": 4,
        },
        "Exp1: +Day_sin/cos (24h cycle)": {
            "features": ["CO2", "dCO2", "Noise", "Day_sin", "Day_cos"],
            "n_states": 4,
        },
        "Exp2: +Day+Weekday (24h+7d cycles)": {
            "features": [
                "CO2", "dCO2", "Noise",
                "Day_sin", "Day_cos", "Weekday_sin", "Weekday_cos",
            ],
            "n_states": 4,
        },
        "Exp3: +hour_weekday (168h cycle)": {
            "features": [
                "CO2", "dCO2", "Noise",
                "hour_weekday_sin", "hour_weekday_cos",
            ],
            "n_states": 4,
        },
        "Exp4: All temporal features": {
            "features": [
                "CO2", "dCO2", "Noise",
                "Day_sin", "Day_cos", "Weekday_sin", "Weekday_cos",
                "Year_sin", "Year_cos",
            ],
            "n_states": 4,
        },
        "Exp5: +Day_sin/cos (24h), 5 states": {
            "features": ["CO2", "dCO2", "Noise", "Day_sin", "Day_cos"],
            "n_states": 5,
        },
        "Exp6: +Day+Weekday, 6 states": {
            "features": [
                "CO2", "dCO2", "Noise",
                "Day_sin", "Day_cos", "Weekday_sin", "Weekday_cos",
            ],
            "n_states": 6,
        },
    }

    # ── Verify all features exist ─────────────────────────────────────
    all_features = set()
    for exp in experiments.values():
        all_features.update(exp["features"])

    missing = all_features - set(test_df.columns)
    if missing:
        logger.error(f"Missing columns in test_df: {missing}")
        logger.info("Available columns: %s", test_df.columns.tolist())
        sys.exit(1)

    # ── Run experiments ───────────────────────────────────────────────
    output_dir = PROJECT_ROOT / "results" / "hmm_temporal_occupancy"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    print(f"\n{'='*70}")
    print(f"  HMM TEMPORAL FEATURES vs OCCUPANCY DETECTORS")
    print(f"{'='*70}")

    for exp_name, exp_cfg in experiments.items():
        features = exp_cfg["features"]
        n_states = exp_cfg["n_states"]

        print(f"\n{'─'*70}")
        print(f"  {exp_name}")
        print(f"  Features: {features}")
        print(f"  States: {n_states}")
        print(f"{'─'*70}")

        # Fit HMM on train data
        hmm = HMMRegimeDetector(
            n_states=n_states,
            covariance_type="full",
            n_iter=200,
            hmm_features=features,
        )
        hmm.fit(train_df)

        # Decode test states
        test_states = hmm.predict_states(test_df)

        # Analyze vs occupancy
        exp_result = analyze_hmm_states_vs_occupancy(
            hmm_states=test_states,
            occupancy_results=results_actual,
            consensus_label=consensus,
            n_states=n_states,
            feature_names=features,
            test_df=test_df,
        )

        # Store transition matrix
        exp_result["transmat"] = hmm.hmm.transmat_.tolist()
        exp_result["converged"] = bool(hmm.hmm.monitor_.converged)

        all_results[exp_name] = exp_result

        # Print state profiles
        print(f"\n  {'State':>5}  {'Count':>6}  {'%':>5}  {'CO2':>8}  {'dCO2':>8}  "
              f"{'Noise':>7}  {'Consensus Occ%':>15}")
        for sp in exp_result["state_profiles"]:
            m = sp["means"]
            print(
                f"  {sp['state']:>5d}  {sp['count']:>6d}  {sp['pct']:>5.1f}  "
                f"{m.get('CO2', 0):>8.1f}  {m.get('dCO2', 0):>8.2f}  "
                f"{m.get('Noise', 0):>7.1f}  {sp['consensus_occupied_pct']:>14.1f}%"
            )

        bm = exp_result["best_binary_mapping"]
        print(f"\n  Best binary mapping: states {bm['occupied_states']} → occupied")
        print(f"  κ vs consensus: {bm['kappa_vs_consensus']:.4f}")
        print(f"  HMM occupancy rate: {bm['occupancy_rate']:.1f}%")

        print(f"\n  Per-detector kappas (using best mapping):")
        for det, k in exp_result["kappas"].items():
            print(f"    {det:25s}: κ = {k:.4f}")

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY: κ vs CONSENSUS")
    print(f"{'='*70}")
    print(f"\n  {'Experiment':50s}  {'κ':>7}  {'Occ%':>6}  {'States→Occ':>12}")
    print(f"  {'─'*80}")
    for exp_name, res in all_results.items():
        bm = res["best_binary_mapping"]
        occ_str = str(bm["occupied_states"])
        print(
            f"  {exp_name:50s}  {bm['kappa_vs_consensus']:>7.4f}  "
            f"{bm['occupancy_rate']:>5.1f}%  {occ_str:>12}"
        )

    # ── Save results ──────────────────────────────────────────────────
    results_path = output_dir / "experiment_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    # ── Generate comparison figure ────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: kappa vs consensus for each experiment
    exp_names_short = [
        name.split(":")[0].strip() for name in all_results.keys()
    ]
    kappas_consensus = [
        res["best_binary_mapping"]["kappa_vs_consensus"]
        for res in all_results.values()
    ]
    occ_rates = [
        res["best_binary_mapping"]["occupancy_rate"]
        for res in all_results.values()
    ]

    colors = plt.cm.Set2(np.linspace(0, 1, len(exp_names_short)))
    bars = axes[0].barh(exp_names_short, kappas_consensus, color=colors)
    axes[0].set_xlabel("Cohen's κ vs Consensus")
    axes[0].set_title("HMM State Alignment with Occupancy Consensus")
    axes[0].axvline(x=0, color="black", linewidth=0.5)
    for bar, val in zip(bars, kappas_consensus):
        axes[0].text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9
        )
    axes[0].set_xlim(-0.1, max(kappas_consensus) + 0.15)

    # Plot 2: per-detector kappas for all experiments
    detector_names = list(next(iter(all_results.values()))["kappas"].keys())
    x = np.arange(len(detector_names))
    width = 0.8 / len(all_results)

    for i, (exp_name, res) in enumerate(all_results.items()):
        kvals = [res["kappas"].get(d, 0) for d in detector_names]
        axes[1].bar(
            x + i * width - 0.4 + width/2, kvals, width,
            label=exp_name.split(":")[0].strip(),
            color=colors[i], edgecolor="black", linewidth=0.3
        )

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(detector_names, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("Cohen's κ")
    axes[1].set_title("Per-Detector κ by HMM Configuration")
    axes[1].legend(fontsize=7, loc="upper left", ncol=2)
    axes[1].axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    fig_path = output_dir / "hmm_temporal_vs_occupancy.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Figure saved to {fig_path}")

    # ── State-hour heatmap for best experiment ────────────────────────
    # Find experiment with highest consensus kappa
    best_exp_name = max(
        all_results, key=lambda n: all_results[n]["best_binary_mapping"]["kappa_vs_consensus"]
    )
    best_exp = all_results[best_exp_name]

    # Re-run best config to get states for the heatmap
    best_cfg = experiments[best_exp_name]
    hmm_best = HMMRegimeDetector(
        n_states=best_cfg["n_states"],
        covariance_type="full",
        n_iter=200,
        hmm_features=best_cfg["features"],
    )
    hmm_best.fit(train_df)
    best_states = hmm_best.predict_states(test_df)

    # State distribution by hour of day
    if "datetime" in test_df.columns:
        hours = pd.to_datetime(test_df["datetime"]).dt.hour.values
    elif "Day_sin" in test_df.columns and "Day_cos" in test_df.columns:
        # Recover hour from sin/cos encoding
        angles = np.arctan2(test_df["Day_sin"].values, test_df["Day_cos"].values)
        hours = ((angles / (2 * np.pi)) * 24) % 24
        hours = np.round(hours).astype(int) % 24
    else:
        hours = np.zeros(len(test_df), dtype=int)

    # Build hour-state frequency matrix
    n_hours = 24
    hour_state_matrix = np.zeros((best_cfg["n_states"], n_hours))
    for h in range(n_hours):
        mask = hours == h
        if mask.sum() > 0:
            for s in range(best_cfg["n_states"]):
                hour_state_matrix[s, h] = float((best_states[mask] == s).sum()) / mask.sum()

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(hour_state_matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(best_cfg["n_states"]))

    # Label states with their consensus occupancy %
    state_labels = []
    for sp in best_exp["state_profiles"]:
        co2 = sp["means"].get("CO2", 0)
        occ = sp["consensus_occupied_pct"]
        state_labels.append(f"S{sp['state']} (CO2={co2:.0f}, occ={occ:.0f}%)")
    ax.set_yticklabels(state_labels, fontsize=9)

    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)])
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("HMM State")
    ax.set_title(f"State Distribution by Hour — {best_exp_name}")
    plt.colorbar(im, ax=ax, label="P(state | hour)")

    fig_path2 = output_dir / "hmm_state_hour_heatmap.png"
    plt.savefig(fig_path2, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Heatmap saved to {fig_path2}")

    print(f"\n  Best experiment: {best_exp_name}")
    print(f"  All results saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
