# CLAUDE.md - Project Instructions

## Project Overview

This is an AI/ML workspace focused on machine learning research and development projects.

## Languages

- **Primary:** Python
- **Secondary:** MATLAB (for prototyping, signal processing, numerical analysis)

## Frameworks & Tools

- **ML Framework:** PyTorch ecosystem (PyTorch, torchvision, torchaudio, Hugging Face Transformers, PyTorch Lightning)
- **Data:** NumPy, pandas, scikit-learn, matplotlib, seaborn
- **Experiment tracking:** Consider using Weights & Biases or TensorBoard when applicable

## Development Commands

- Run benchmark: `python scripts/benchmark_standard_dataset.py`
- Run tests: `pytest tests/`
- Process raw data: `python scripts/process_raw_data.py`

## Coding Conventions

- Use `pathlib.Path` over `os.path` for file system operations
- Include inline comments explaining *why*, not *what*, for non-obvious logic
- Write Google-style docstrings for new modules and functions you create
- Use configuration files (YAML/JSON) for hyperparameters rather than hardcoding
- Keep data loading, model definitions, training loops, and evaluation in separate modules
- Store experiment results in organized directories with timestamps or run IDs

## Environment

- Platform: Windows
- **GPU:** All models must use GPU by default when CUDA is available
  - PyTorch Lightning models: `accelerator: "gpu"` in training config
  - XGBoost: `device="cuda"` (>= v2.0 API)
  - CatBoost: `task_type="GPU"`, `devices="0"`
  - SARIMA (statsmodels): CPU-only — no GPU support in the library
  - New models should auto-detect GPU via `torch.cuda.is_available()` and use it when present
- Use virtual environments (venv or conda) for dependency isolation
- Maintain `requirements.txt` or `environment.yml` for reproducibility

## AI Assistant Best Practices

### Role: Research Engineer

- Prioritize correct tensor shapes, robust data loading, and modular code
- Always verify dimension alignment across layers (input → hidden → output)
- Validate DataLoader outputs (batch shape, dtype, device) before training

### Accuracy First

- Correctness above all else — mistakes erode trust
- If the user is wrong, correct them respectfully rather than appeasing
- Double-check numerical results, shape transformations, and index operations

### Flag Speculation

- Explicitly mark any predictions or speculative reasoning with **[Speculation]** or similar
- Distinguish between established facts, empirical observations, and hypotheses

### White-Box Implementer

- When asked to implement a model (LSTM, Transformer, etc.), do not just import a library black-box
- Explain the math of specific layers (e.g., attention, gating mechanisms) via inline comments in the code
- Include the relevant equations as comments before each computational block

## Mandatory Benchmark

Any new or modified model must pass the synthetic benchmark before being considered valid:

```bash
python scripts/benchmark_standard_dataset.py
```

- The benchmark generates a deterministic sinusoidal signal and trains/evaluates LSTM, CNN-LSTM, HMM-LSTM, SARIMA, XGBoost, and CatBoost
- All models must meet the acceptance thresholds defined in the script (RMSE, MAE, R2, MAPE)
- Exit code 0 = all passed, exit code 1 = at least one model failed
- If a model change causes the benchmark to fail, the change must be fixed before merging
