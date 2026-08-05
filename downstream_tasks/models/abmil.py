# Attention-Based Multiple Instance Learning models.

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from .attention import get_attention_module
from .pdl import PDropout
from .registry import ClassifierType


def prepare_context_bags(x_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert ctx into flat instances plus bag ids."""
    ctx = x_dict["ctx"]

    if isinstance(ctx, list):
        bag_sizes = torch.tensor(
            [bag.shape[0] for bag in ctx],
            device=ctx[0].device,
            dtype=torch.long,
        )
        ctx_ptr = torch.empty(len(ctx) + 1, device=ctx[0].device, dtype=torch.long)
        ctx_ptr[0] = 0
        ctx_ptr[1:] = torch.cumsum(bag_sizes, dim=0)
        ctx_batch = torch.repeat_interleave(
            torch.arange(len(ctx), device=ctx[0].device, dtype=torch.long),
            bag_sizes,
        )
        return torch.cat(ctx, dim=0), ctx_batch, ctx_ptr

    ctx_batch = x_dict.get("ctx_batch")
    ctx_ptr = x_dict.get("ctx_ptr")

    # Single unbatched bag: all rows belong to protein 0.
    if ctx_batch is None and ctx_ptr is None:
        ctx_batch = torch.zeros(ctx.shape[0], device=ctx.device, dtype=torch.long)
        ctx_ptr = torch.tensor([0, ctx.shape[0]], device=ctx.device, dtype=torch.long)
        return ctx, ctx_batch, ctx_ptr

    # PyG attention uses batch ids and ptr, for pooling and PDL to know how many bags are in the mini-batch.
    if ctx_batch is None:
        bag_sizes = ctx_ptr[1:] - ctx_ptr[:-1]
        ctx_batch = torch.repeat_interleave(
            torch.arange(bag_sizes.numel(), device=ctx.device, dtype=torch.long),
            bag_sizes,
        )
    elif ctx_ptr is None:
        counts = torch.bincount(ctx_batch)
        ctx_ptr = torch.empty(counts.numel() + 1, device=ctx.device, dtype=torch.long)
        ctx_ptr[0] = 0
        ctx_ptr[1:] = torch.cumsum(counts, dim=0)

    return ctx, ctx_batch, ctx_ptr


def pool_context_embeddings(
    x_ctx: torch.Tensor,
    att_w: torch.Tensor,
    ctx_batch: torch.Tensor,
    ctx_ptr: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """Pool context instances into one representation per bag.

    x_ctx has shape [num_instances, ctx_dim].
    att_w has shape [num_instances, num_heads] and is normalized within each bag.
    The output is [num_bags, ctx_dim] for one head, or [num_bags, num_heads * ctx_dim].
    """
    weighted_ctx = x_ctx.unsqueeze(1) * att_w.unsqueeze(-1)
    pooled = scatter(
        weighted_ctx,
        ctx_batch,
        dim=0,
        dim_size=ctx_ptr.numel() - 1,
        reduce="sum",
    )

    if num_heads == 1:
        return pooled.squeeze(1)
    return pooled.reshape(pooled.shape[0], -1)


class PDLProjector(nn.Sequential):
    """Sequential projector with PDropout layers."""

    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        ptr: Optional[torch.Tensor] = None,
    ):
        current = x
        current_batch = batch
        current_ptr = ptr

        for layer in self:
            if isinstance(layer, PDropout):
                output = layer(current, batch=current_batch, ptr=current_ptr)
                if current_batch is None and current_ptr is None:
                    current = output
                else:
                    current, current_batch, current_ptr = output
            else:
                current = layer(current)
        if current_batch is None and current_ptr is None:
            return current
        return current, current_batch, current_ptr


def build_pdl_projector(
    ctx_dim: int,
    mode: str = "identity",
    num_identity_layers: int = 2,
) -> PDLProjector:
    mode = (mode or "identity").strip().lower()
    if mode not in {"identity", "linear", "none"}:
        raise ValueError(f"Unsupported pdl_proj_mode: {mode}")

    if mode == "none":
        return PDLProjector(PDropout(0.0))

    if mode == "linear":
        return PDLProjector(
            nn.Linear(ctx_dim, ctx_dim),
            PDropout(0.0, importance_mode="abs_mean"),
        )

    if num_identity_layers < 1:
        raise ValueError(f"pdl_proj_layers must be >= 1, got {num_identity_layers}")

    layers = []
    for _ in range(num_identity_layers):
        layers.extend(
            [
                nn.Linear(ctx_dim, ctx_dim),
                nn.ReLU(),
                PDropout(0.0, importance_mode="mean"),
            ]
        )
    return PDLProjector(*layers)


class MLPClassifier(nn.Module):
    """
    Simple MLP classifier with one hidden layer.

    Architecture: Linear -> ReLU -> Dropout -> Linear
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(input_dim, 1024, num_classes)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def build_classifier(
    classifier_type: ClassifierType,
    input_dim: int,
    num_classes: int,
    hidden_dim: Optional[int] = None,
    dropout: float = 0.3,
) -> nn.Module:
    if classifier_type == ClassifierType.LINEAR:
        return nn.Linear(input_dim, num_classes)
    if classifier_type == ClassifierType.MLP:
        return MLPClassifier(input_dim, num_classes, hidden_dim, dropout)


class ABMIL_LateFusion(nn.Module):
    """
    ABMIL with late fusion of contextual and ESM embeddings.
    """

    def __init__(
        self,
        num_classes: int,
        ctx_dim: int,
        esm_dim: int,
        num_heads: int = 1,
        att_hidden: int = 32,
        dropout: float = 0.3,
        att_dropout: float = 0.0,
        attention_type: str = "gated",
        classifier_type: ClassifierType = ClassifierType.LINEAR,
        mlp_hidden_dim: Optional[int] = None,
        use_pdl: bool = False,
        pdl_proj_mode: str = "identity",
        pdl_proj_layers: int = 2,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.ctx_proj = (
            build_pdl_projector(ctx_dim, mode=pdl_proj_mode, num_identity_layers=pdl_proj_layers)
            if use_pdl
            else None
        )
        self.esm_proj = None #old

        self.attention = get_attention_module(
            attention_type,
            ctx_dim,
            att_hidden,
            num_heads,
            att_dropout=att_dropout,
        )

        pooled_ctx_dim = num_heads * ctx_dim
        self.fusion_dim = pooled_ctx_dim + esm_dim
        self.norm = nn.LayerNorm(self.fusion_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = build_classifier(
            classifier_type,
            self.fusion_dim,
            num_classes,
            mlp_hidden_dim,
            dropout,
        )

    def _pool_context(self, x_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_ctx, ctx_batch, ctx_ptr = prepare_context_bags(x_dict)
        if self.ctx_proj is not None:
            x_ctx, ctx_batch, ctx_ptr = self.ctx_proj(x_ctx, batch=ctx_batch, ptr=ctx_ptr)

        att_w = self.attention(x_ctx, batch=ctx_batch, ptr=ctx_ptr)
        pooled = pool_context_embeddings(x_ctx, att_w, ctx_batch, ctx_ptr, self.num_heads)
        return pooled, att_w, ctx_batch, ctx_ptr

    def forward(self, x_dict: Dict, return_attention: bool = False):
        ctx_pooled, att_w, ctx_batch, ctx_ptr = self._pool_context(x_dict)
        fused = torch.cat([ctx_pooled, x_dict["esm"]], dim=1)
        logits = self.classifier(self.dropout(self.norm(fused)))

        if return_attention:
            return logits, {"weights": att_w, "batch": ctx_batch, "ptr": ctx_ptr}
        return logits


class ABMIL_ContextOnly(nn.Module):
    """
    ABMIL with context-only (no ESM fusion).
    """

    def __init__(
        self,
        num_classes: int,
        ctx_dim: int,
        num_heads: int = 1,
        att_hidden: int = 32,
        dropout: float = 0.3,
        att_dropout: float = 0.0,
        attention_type: str = "gated",
        classifier_type: ClassifierType = ClassifierType.LINEAR,
        mlp_hidden_dim: Optional[int] = None,
        use_pdl: bool = False,
        pdl_proj_mode: str = "identity",
        pdl_proj_layers: int = 2,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.ctx_proj = (
            build_pdl_projector(ctx_dim, mode=pdl_proj_mode, num_identity_layers=pdl_proj_layers)
            if use_pdl
            else None
        )
        self.attention = get_attention_module(
            attention_type,
            ctx_dim,
            att_hidden,
            num_heads,
            att_dropout=att_dropout,
        )
        self.norm = nn.LayerNorm(num_heads * ctx_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = build_classifier(
            classifier_type,
            num_heads * ctx_dim,
            num_classes,
            mlp_hidden_dim,
            dropout,
        )

    def _pool_context(self, x_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_ctx, ctx_batch, ctx_ptr = prepare_context_bags(x_dict)
        if self.ctx_proj is not None:
            x_ctx, ctx_batch, ctx_ptr = self.ctx_proj(x_ctx, batch=ctx_batch, ptr=ctx_ptr)

        att_w = self.attention(x_ctx, batch=ctx_batch, ptr=ctx_ptr)
        pooled = pool_context_embeddings(x_ctx, att_w, ctx_batch, ctx_ptr, self.num_heads)
        return pooled, att_w, ctx_batch, ctx_ptr

    def forward(self, x_dict: Dict, return_attention: bool = False):
        pooled, att_w, ctx_batch, ctx_ptr = self._pool_context(x_dict)
        logits = self.classifier(self.dropout(self.norm(pooled)))

        if return_attention:
            return logits, {"weights": att_w, "batch": ctx_batch, "ptr": ctx_ptr}
        return logits
