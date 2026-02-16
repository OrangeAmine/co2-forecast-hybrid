"""Tests for experiment config loading, merging, and naming conventions."""

import json
from pathlib import Path

import pytest
import yaml

from src.utils.config import _deep_merge, load_config
from src.evaluation.metrics import save_metrics


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
EXPERIMENTS_DIR = CONFIGS_DIR / "experiments"


# ---------------------------------------------------------------------------
#  Deep merge
# ---------------------------------------------------------------------------

class TestDeepMerge:
    """Tests for config _deep_merge utility."""

    def test_flat_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        merged = _deep_merge(base, override)
        assert merged == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"data": {"csv": "a.csv", "scaler": "standard"}}
        override = {"data": {"scaler": "robust"}}
        merged = _deep_merge(base, override)
        assert merged["data"]["csv"] == "a.csv"
        assert merged["data"]["scaler"] == "robust"

    def test_base_unchanged(self):
        base = {"a": {"x": 1}}
        override = {"a": {"x": 2}}
        _ = _deep_merge(base, override)
        assert base["a"]["x"] == 1  # Original should not be mutated


# ---------------------------------------------------------------------------
#  load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    """Tests for YAML config loading and merging."""

    def test_load_single_config(self, tmp_path):
        """Should load a single YAML file correctly."""
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text("data:\n  csv: test.csv\n")
        result = load_config([cfg_path])
        assert result["data"]["csv"] == "test.csv"

    def test_load_and_merge(self, tmp_path):
        """Later configs should override earlier ones."""
        base = tmp_path / "base.yaml"
        base.write_text("data:\n  csv: base.csv\n  scaler: standard\n")
        override = tmp_path / "override.yaml"
        override.write_text("data:\n  scaler: robust\n")
        result = load_config([base, override])
        assert result["data"]["csv"] == "base.csv"
        assert result["data"]["scaler"] == "robust"

    def test_missing_file_raises(self):
        """Missing config file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config([Path("nonexistent.yaml")])


# ---------------------------------------------------------------------------
#  Experiment configs
# ---------------------------------------------------------------------------

class TestExperimentConfigs:
    """Validate the actual experiment config files in configs/experiments/."""

    @pytest.mark.parametrize("filename", [
        "preproc_A_simple_5min.yaml",
        "preproc_B_simple_1h.yaml",
        "preproc_C_enhanced_5min.yaml",
        "preproc_D_enhanced_1h.yaml",
    ])
    def test_experiment_yaml_loads(self, filename):
        """Each experiment YAML should load without errors."""
        path = EXPERIMENTS_DIR / filename
        if not path.exists():
            pytest.skip(f"{path} not found")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert "experiment" in cfg
        assert "data" in cfg
        assert "name" in cfg["experiment"]

    @pytest.mark.parametrize("filename", [
        "preproc_A_simple_5min.yaml",
        "preproc_B_simple_1h.yaml",
        "preproc_C_enhanced_5min.yaml",
        "preproc_D_enhanced_1h.yaml",
    ])
    def test_experiment_has_pipeline_variant(self, filename):
        """Experiment config should specify a pipeline_variant."""
        exp_path = EXPERIMENTS_DIR / filename
        if not exp_path.exists():
            pytest.skip("Config file not found")
        with open(exp_path) as f:
            cfg = yaml.safe_load(f)
        assert "pipeline_variant" in cfg["data"]
        assert cfg["data"]["pipeline_variant"] in ("simple", "enhanced")

    def test_enhanced_has_more_features_than_simple(self):
        """Enhanced config should have more features than simple."""
        a_path = EXPERIMENTS_DIR / "preproc_A_simple_5min.yaml"
        c_path = EXPERIMENTS_DIR / "preproc_C_enhanced_5min.yaml"
        if not a_path.exists() or not c_path.exists():
            pytest.skip("Config files not found")
        with open(a_path) as f:
            simple = yaml.safe_load(f)
        with open(c_path) as f:
            enhanced = yaml.safe_load(f)
        assert len(enhanced["data"]["feature_columns"]) > len(simple["data"]["feature_columns"])

    def test_enhanced_has_preprocessing_section(self):
        """Enhanced variants should have a preprocessing section."""
        for filename in ["preproc_C_enhanced_5min.yaml", "preproc_D_enhanced_1h.yaml"]:
            path = EXPERIMENTS_DIR / filename
            if not path.exists():
                pytest.skip(f"{path} not found")
            with open(path) as f:
                cfg = yaml.safe_load(f)
            assert "preprocessing" in cfg
            assert "denoising" in cfg["preprocessing"]
            assert "outlier_detection" in cfg["preprocessing"]


# ---------------------------------------------------------------------------
#  Experiment-tagged model naming
# ---------------------------------------------------------------------------

class TestExperimentNaming:
    """Tests for experiment-aware model naming convention."""

    def test_model_name_format(self):
        """Model name should be '{exp}_{model}_h{horizon}'."""
        exp_name = "preproc_A"
        model_base = "LSTM"
        horizon = 1
        model_name = f"{exp_name}_{model_base}_h{horizon}"
        assert model_name == "preproc_A_LSTM_h1"

    def test_model_name_cnn_lstm(self):
        """CNN-LSTM name should handle the hyphen correctly."""
        exp_name = "preproc_C"
        model_base = "CNN-LSTM"
        horizon = 24
        model_name = f"{exp_name}_{model_base}_h{horizon}"
        assert model_name == "preproc_C_CNN-LSTM_h24"


# ---------------------------------------------------------------------------
#  save_metrics with experiment info
# ---------------------------------------------------------------------------

class TestSaveMetricsExperiment:
    """Tests for save_metrics including experiment metadata."""

    def test_metrics_json_includes_experiment(self, tmp_path):
        """save_metrics with experiment_info should include it in JSON."""
        metrics = {"mse": 0.01, "rmse": 0.1, "mae": 0.08, "mape": 2.0, "r2": 0.95}
        exp_info = {"name": "preproc_A", "label": "Simple (5-min)", "description": "test"}
        out_path = tmp_path / "metrics.json"
        save_metrics(metrics, "preproc_A_LSTM_h1", out_path, experiment_info=exp_info)

        with open(out_path) as f:
            data = json.load(f)
        assert data["model"] == "preproc_A_LSTM_h1"
        assert data["experiment"]["name"] == "preproc_A"
        assert "metrics" in data
        assert data["metrics"]["rmse"] == 0.1

    def test_metrics_json_without_experiment(self, tmp_path):
        """save_metrics without experiment_info should omit experiment key."""
        metrics = {"mse": 0.01, "rmse": 0.1, "mae": 0.08, "mape": 2.0, "r2": 0.95}
        out_path = tmp_path / "metrics.json"
        save_metrics(metrics, "LSTM_h1", out_path)

        with open(out_path) as f:
            data = json.load(f)
        assert "experiment" not in data
