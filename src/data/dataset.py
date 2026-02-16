"""PyTorch Dataset for windowed time series data."""

import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset wrapping pre-computed sliding window sequences.

    Args:
        X: Input windows of shape (n_sequences, lookback, n_features).
        y: Target values of shape (n_sequences, horizon).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
