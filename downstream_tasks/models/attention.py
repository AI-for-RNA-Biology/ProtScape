# attention.py
# Attention modules for ABMIL models

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as pyg_softmax


def _normalize_attention(
    scores: torch.Tensor,
    batch: torch.Tensor = None,
    ptr: torch.Tensor = None,
) -> torch.Tensor:
    # single bag case
    if batch is None and ptr is None:
        return F.softmax(scores, dim=0)

    # multiple bags: use PyG softmax to normalize within each bag
    return pyg_softmax(scores, index=batch, ptr=ptr, dim=0)


class GatedContextAttention(nn.Module):
    """
    Gated (multiplicative) attention: tanh(Vx) * sigmoid(Ux)

    Uses a multiplicative gate to control the attention weights.
    """

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
        """
        Args:
            x: Input tensor of shape (n_instances, dim)

        Returns:
            Attention weights of shape (n_instances, num_heads)
        """
        a = torch.tanh(self.v_layer(x)) * torch.sigmoid(self.u_layer(x))
        a = self.hidden_dropout(a)
        a = self.w_layer(a)
        return _normalize_attention(a, batch=batch, ptr=ptr)


class AdditiveContextAttention(nn.Module):
    """
    Additive attention: tanh(Vx + Ux)

    Standard additive attention mechanism combining two transformations.
    """

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
        """
        Args:
            x: Input tensor of shape (n_instances, dim)

        Returns:
            Attention weights of shape (n_instances, num_heads)
        """
        a = torch.tanh(self.v_layer(x) + self.u_layer(x))
        a = self.hidden_dropout(a)
        a = self.w_layer(a)
        return _normalize_attention(a, batch=batch, ptr=ptr)


class ConjunctiveContextAttention(nn.Module):
    """
    Conjunctive attention: tanh(Vx) + sigmoid(Ux)

    Mixed additive-multiplicative attention that combines
    tanh and sigmoid non-linearities additively.
    """

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
        """
        Args:
            x: Input tensor of shape (n_instances, dim)

        Returns:
            Attention weights of shape (n_instances, num_heads)
        """
        a = torch.tanh(self.v_layer(x)) + torch.sigmoid(self.u_layer(x))
        a = self.hidden_dropout(a)
        a = self.w_layer(a)
        return _normalize_attention(a, batch=batch, ptr=ptr)


def get_attention_module(
    attention_type: str,
    dim: int,
    hidden_dim: int,
    num_heads: int = 1,
    att_dropout: float = 0.0,
) -> nn.Module:
    """
    Assign attention modules.

    Args:
        attention_type: "gated", "additive", or "conjunctive"
        dim: Input dimension
        hidden_dim: Hidden dimension for attention layers
        num_heads: Number of attention heads

    Returns:
        Attention module instance
    """
    attention_type = (attention_type or "").strip().lower()

    if attention_type == "gated":
        return GatedContextAttention(dim, hidden_dim, num_heads, att_dropout=att_dropout)
    if attention_type == "additive":
        return AdditiveContextAttention(dim, hidden_dim, num_heads, att_dropout=att_dropout)
    if attention_type == "conjunctive":
        return ConjunctiveContextAttention(dim, hidden_dim, num_heads, att_dropout=att_dropout)
