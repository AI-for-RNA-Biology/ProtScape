"""Fit a linear probe to fixed protein features."""

import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    """
    Simple linear probe (logistic regression) for multi-label classification.
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one logit per target label."""
        return self.linear(x)
