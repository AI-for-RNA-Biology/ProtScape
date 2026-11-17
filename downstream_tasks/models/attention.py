"""Attention weights normalized across each protein's cell contexts."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as pyg_softmax


def normalize_attention(
    scores: torch.Tensor,
    batch: torch.Tensor = None,
    ptr: torch.Tensor = None,
) -> torch.Tensor:
    """Normalize each attention head within its protein's context bag."""
    if batch is None and ptr is None:
        return F.softmax(scores, dim=0)

    return pyg_softmax(scores, index=batch, ptr=ptr, dim=0)


class GatedContextAttention(nn.Module):
    """Score contexts with tanh(Vx) * sigmoid(Ux)."""

    def __init__(self, dim: int, hidden_dim: int, num_heads: int = 1, att_dropout: float = 0.0):
        super().__init__()
        self.v_layer = nn.Linear(dim, hidden_dim)
        self.u_layer = nn.Linear(dim, hidden_dim)
        self.hidden_dropout = nn.Dropout(att_dropout)
        self.w_layer = nn.Linear(hidden_dim, num_heads)

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor = None,
        ptr: torch.Tensor = None,
    ) -> torch.Tensor:
        """Score contexts with a multiplicative gate and normalize per protein."""
        a = torch.tanh(self.v_layer(x)) * torch.sigmoid(self.u_layer(x))
        a = self.hidden_dropout(a)
        a = self.w_layer(a)
        return normalize_attention(a, batch=batch, ptr=ptr)


class AdditiveContextAttention(nn.Module):
    """Score contexts with tanh(Vx + Ux)."""

    def __init__(self, dim: int, hidden_dim: int, num_heads: int = 1, att_dropout: float = 0.0):
        super().__init__()
        self.v_layer = nn.Linear(dim, hidden_dim)
        self.u_layer = nn.Linear(dim, hidden_dim)
        self.hidden_dropout = nn.Dropout(att_dropout)
        self.w_layer = nn.Linear(hidden_dim, num_heads)

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor = None,
        ptr: torch.Tensor = None,
    ) -> torch.Tensor:
        """Score contexts from summed linear projections and normalize per protein."""
        a = torch.tanh(self.v_layer(x) + self.u_layer(x))
        a = self.hidden_dropout(a)
        a = self.w_layer(a)
        return normalize_attention(a, batch=batch, ptr=ptr)


class ConjunctiveContextAttention(nn.Module):
    """Score contexts with tanh(Vx) + sigmoid(Ux)."""

    def __init__(self, dim: int, hidden_dim: int, num_heads: int = 1, att_dropout: float = 0.0):
        super().__init__()
        self.v_layer = nn.Linear(dim, hidden_dim)
        self.u_layer = nn.Linear(dim, hidden_dim)
        self.hidden_dropout = nn.Dropout(att_dropout)
        self.w_layer = nn.Linear(hidden_dim, num_heads)

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor = None,
        ptr: torch.Tensor = None,
    ) -> torch.Tensor:
        """Combine tanh and sigmoid context scores and normalize per protein."""
        a = torch.tanh(self.v_layer(x)) + torch.sigmoid(self.u_layer(x))
        a = self.hidden_dropout(a)
        a = self.w_layer(a)
        return normalize_attention(a, batch=batch, ptr=ptr)


def get_attention_module(
    attention_type: str,
    dim: int,
    hidden_dim: int,
    num_heads: int = 1,
    att_dropout: float = 0.0,
) -> nn.Module:
    """Assign attention modules."""
    attention_type = (attention_type or "").strip().lower()

    if attention_type == "gated":
        return GatedContextAttention(dim, hidden_dim, num_heads, att_dropout=att_dropout)
    if attention_type == "additive":
        return AdditiveContextAttention(dim, hidden_dim, num_heads, att_dropout=att_dropout)
    if attention_type == "conjunctive":
        return ConjunctiveContextAttention(dim, hidden_dim, num_heads, att_dropout=att_dropout)
