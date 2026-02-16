"""Tests for GPU enforcement in trainer configuration."""

import pytest
import torch

from src.training.trainer import create_trainer


class TestGPUAvailability:
    """Tests for CUDA GPU availability."""

    def test_cuda_available(self):
        """torch.cuda.is_available() should return True on this machine."""
        assert torch.cuda.is_available(), (
            "CUDA is not available. This test verifies that the training "
            "machine has a GPU. If running on a CPU-only machine, skip "
            "this test with: pytest -k 'not test_cuda_available'"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device_count(self):
        """At least one CUDA device should be present."""
        assert torch.cuda.device_count() >= 1


class TestTrainerGPUConfig:
    """Tests for create_trainer GPU configuration."""

    def _base_config(self, accelerator: str = "gpu") -> dict:
        return {
            "training": {
                "seed": 42,
                "accelerator": accelerator,
                "devices": 1,
                "precision": 32,
                "num_workers": 0,
                "pin_memory": False,
                "results_dir": "results",
                "log_every_n_steps": 1,
                "enable_progress_bar": False,
                "max_epochs": 1,
                "patience": 5,
                "gradient_clip_val": 1.0,
            },
        }

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_trainer_uses_gpu(self, tmp_path):
        """Trainer created with accelerator='gpu' should target GPU."""
        config = self._base_config("gpu")
        config["training"]["results_dir"] = str(tmp_path)
        trainer, run_dir = create_trainer(config, "test_model")
        assert trainer.accelerator is not None
        # The Trainer should be configured for GPU
        assert "gpu" in str(type(trainer.accelerator)).lower() or \
               "cuda" in str(type(trainer.accelerator)).lower()

    def test_gpu_not_available_raises(self):
        """If CUDA is not available and accelerator='gpu', should raise RuntimeError."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available, cannot test the error path")
        config = self._base_config("gpu")
        with pytest.raises(RuntimeError, match="GPU required"):
            create_trainer(config, "test_model")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_auto_accelerator_still_works(self, tmp_path):
        """accelerator='auto' should still create a valid trainer."""
        config = self._base_config("auto")
        config["training"]["results_dir"] = str(tmp_path)
        trainer, run_dir = create_trainer(config, "test_model")
        assert trainer is not None
