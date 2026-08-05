# linear.py
# Linear probe model for downstream tasks

import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    """
    Simple linear probe (logistic regression) for multi-label classification.
    """

    def __init__(self, input_dim: int, num_classes: int):
        """
        Args:
            input_dim: Dimension of input features
            num_classes: Number of output classes (labels)
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.linear(x)
