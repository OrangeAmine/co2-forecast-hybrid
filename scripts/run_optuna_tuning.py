"""Run Optuna hyperparameter tuning for specified models.

Systematically searches over learning rate, architecture size, dropout,
lookback window, and batch size to find optimal configurations.

Usage:
    python scripts/run_optuna_tuning.py --model lstm --horizon 1 --trials 100
    python scripts/run_optuna_tuning.py --model seq2seq --horizon 24 --trials 50
    python scripts/run_optuna_tuning.py --model lstm --horizon 1 --trials 5 --experiment configs/experiments/preproc_E_occupancy_1h.yaml
"""

import argparse
import copy
import sys
from pathlib import Path

import optuna

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm import LSTMForecaster
from src.models.cnn_lstm import CNNLSTMForecaster
from src.models.seq2seq import Seq2SeqForecaster
from src.training.optuna_tuner import OptunaTuner
from src.utils.config import load_config
from src.utils.seed import seed_everything


# ──────────────────────────────────────────────────────────────────────
#  Search space definitions
# ──────────────────────────────────────────────────────────────────────

def lstm_search_space(trial: optuna.Trial) -> dict:
    """Define LSTM hyperparameter search space.

    Covers the most impactful hyperparameters:
    - Learning rate (log-uniform): largest effect on convergence
    - Hidden size: model capacity
    - Num layers: depth vs. vanishing gradients
    - Dropout: regularization strength
    - Lookback hours: how much history the model sees
    - Batch size: interacts with learning rate
    """
    return {
        "training.learning_rate": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "model.hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256, 512]),
        "model.num_layers": trial.suggest_int("num_layers", 1, 3),
        "model.dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "data.lookback_hours": trial.suggest_int("lookback_hours", 12, 72),
        "training.batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }


def cnn_lstm_search_space(trial: optuna.Trial) -> dict:
    """Define CNN-LSTM hyperparameter search space."""
    return {
        "training.learning_rate": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "model.lstm_hidden_size": trial.suggest_categorical("lstm_hidden", [64, 128, 256]),
        "model.lstm_num_layers": trial.suggest_int("lstm_layers", 1, 3),
        "model.lstm_dropout": trial.suggest_float("lstm_dropout", 0.1, 0.5),
        "model.fc_hidden_size": trial.suggest_categorical("fc_hidden", [32, 64, 128]),
        "data.lookback_hours": trial.suggest_int("lookback_hours", 12, 72),
        "training.batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }


def seq2seq_search_space(trial: optuna.Trial) -> dict:
    """Define Seq2Seq hyperparameter search space.

    Includes encoder/decoder hidden sizes, teacher forcing ratio,
    and standard training hyperparameters.
    """
    return {
        "training.learning_rate": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "model.encoder_hidden_size": trial.suggest_categorical("enc_hidden", [64, 128, 256]),
        "model.decoder_hidden_size": trial.suggest_categorical("dec_hidden", [64, 128, 256]),
        "model.encoder_num_layers": trial.suggest_int("enc_layers", 1, 3),
        "model.encoder_dropout": trial.suggest_float("enc_dropout", 0.1, 0.5),
        "model.teacher_forcing_ratio": trial.suggest_float("tf_ratio", 0.0, 0.7),
        "data.lookback_hours": trial.suggest_int("lookback_hours", 24, 72),
        "training.batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }


SEARCH_SPACES = {
    "lstm": (LSTMForecaster, lstm_search_space, "lstm.yaml"),
    "cnn_lstm": (CNNLSTMForecaster, cnn_lstm_search_space, "cnn_lstm.yaml"),
    "seq2seq": (Seq2SeqForecaster, seq2seq_search_space, "seq2seq.yaml"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(SEARCH_SPACES.keys()),
                        help="Model to tune")
    parser.add_argument("--horizon", type=int, default=1,
                        help="Forecast horizon in hours (default: 1)")
    parser.add_argument("--trials", type=int, default=100,
                        help="Number of Optuna trials (default: 100)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Maximum time in seconds (default: unlimited)")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Path to experiment config YAML")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (default: in-memory)")
    args = parser.parse_args()

    model_class, search_space_fn, model_config_name = SEARCH_SPACES[args.model]

    config_files = [
        str(PROJECT_ROOT / "configs" / "training.yaml"),
        str(PROJECT_ROOT / "configs" / "data.yaml"),
        str(PROJECT_ROOT / "configs" / model_config_name),
    ]
    if args.experiment:
        config_files.append(args.experiment)

    config = load_config(config_files)
    config["data"]["forecast_horizon_hours"] = args.horizon

    study_name = f"{args.model}_h{args.horizon}"
    results_dir = Path("results") / "optuna" / args.model

    seed_everything(config["training"]["seed"])

    print(f"\n{'='*60}")
    print(f"  Optuna Tuning: {args.model.upper()}")
    print(f"  Horizon: {args.horizon}h")
    print(f"  Trials: {args.trials}")
    print(f"  Timeout: {args.timeout}s" if args.timeout else "  Timeout: unlimited")
    print(f"{'='*60}\n")

    tuner = OptunaTuner(
        base_config=config,
        model_class=model_class,
        search_space_fn=search_space_fn,
        n_trials=args.trials,
        timeout=args.timeout,
        study_name=study_name,
        storage=args.storage,
        results_dir=results_dir,
    )

    best_params, study = tuner.run()

    print(f"\n{'='*60}")
    print(f"  TUNING COMPLETE")
    print(f"  Best val_loss: {study.best_value:.6f}")
    print(f"  Best params:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")
    print(f"  Results saved to: {results_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
