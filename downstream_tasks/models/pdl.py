import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

LOG_E = 1.5
LOG_SPACE_MAX = np.power(10, LOG_E) - 1.0


class PDropout(nn.Module):
    """Progressive dropout layer (PDL)."""

    def __init__(self, p: float = 0.0, importance_mode: str = "mean"):
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError("Drop rate must be in range [0, 1]")
        if importance_mode not in {"mean", "abs_mean"}:
            raise ValueError(
                f"importance_mode must be 'mean' or 'abs_mean', got {importance_mode!r}"
            )
        self.p = float(p)
        self.importance_mode = importance_mode

    def forward(
        self,
        inputs: torch.Tensor,
        batch: torch.Tensor = None,
        ptr: torch.Tensor = None,
    ) -> torch.Tensor:
        single_bag = batch is None and ptr is None

        if ptr is None and batch is not None:
            counts = torch.bincount(batch)
            ptr = torch.empty(counts.numel() + 1, device=batch.device, dtype=torch.long)
            ptr[0] = 0
            ptr[1:] = torch.cumsum(counts, dim=0)
        elif batch is None and ptr is not None:
            bag_sizes = ptr[1:] - ptr[:-1]
            batch = torch.repeat_interleave(
                torch.arange(bag_sizes.numel(), device=ptr.device, dtype=torch.long),
                bag_sizes,
            )

        if single_bag and ((not self.training) or self.p <= 0.0 or inputs.shape[0] <= 1):
            return inputs

        if (not self.training) or self.p <= 0.0:
            if single_bag:
                return inputs
            return inputs, batch, ptr

        if single_bag:
            batch = torch.zeros(inputs.shape[0], device=inputs.device, dtype=torch.long)
            ptr = torch.tensor([0, inputs.shape[0]], device=inputs.device, dtype=torch.long)

        keep = self.sample_keep_mask(self.compute_importance(inputs), batch)
        compacted_inputs = inputs[keep]

        if single_bag:
            return compacted_inputs

        compacted_batch = batch[keep]
        compacted_counts = torch.bincount(compacted_batch, minlength=ptr.numel() - 1)
        compacted_ptr = torch.empty(compacted_counts.numel() + 1, device=ptr.device, dtype=torch.long)
        compacted_ptr[0] = 0
        compacted_ptr[1:] = torch.cumsum(compacted_counts, dim=0)
        return compacted_inputs, compacted_batch, compacted_ptr

    def compute_importance(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.importance_mode == "abs_mean":
            importances = torch.mean(torch.abs(inputs), dim=1, keepdim=True)
        else:
            importances = torch.mean(inputs, dim=1, keepdim=True)
        return torch.sigmoid(importances)

    def sample_keep_mask(self, importance: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # use dense tensor for easier indexing
        dense_importance, mask = to_dense_batch(importance.view(-1), batch=batch)
        counts = mask.sum(dim=1)
        drop_probs = self.get_drop_probs(
            counts,
            dense_importance.size(1),
            device=dense_importance.device,
            dtype=dense_importance.dtype,
        ).to(dense_importance.dtype)

        sort_order = dense_importance.masked_fill(~mask, float("inf")).argsort(dim=1)
        sampling_probs = torch.zeros_like(dense_importance)
        sampling_probs.scatter_(1, sort_order, drop_probs)

        keep = (torch.rand_like(sampling_probs) >= sampling_probs) & mask
        empty_rows = keep.sum(dim=1) == 0
        if empty_rows.any():
            fallback = dense_importance.masked_fill(~mask, float("-inf")).argmax(dim=1)
            keep[empty_rows, fallback[empty_rows]] = True
        return keep[mask]

    def get_drop_probs(
        self,
        counts: torch.Tensor,
        max_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        positions = torch.arange(max_len, device=device, dtype=dtype).unsqueeze(0)
        counts_f = counts.unsqueeze(1).to(dtype)
        denom = torch.clamp(counts_f - 1, min=1)
        scale = torch.as_tensor(LOG_SPACE_MAX, device=device, dtype=dtype)
        max_drop = torch.as_tensor(self.p / LOG_E, device=device, dtype=dtype)
        values = max_drop * torch.log10(scale * positions / denom + 1)
        return values * (positions < counts_f)


class LinearScheduler:
    """Progressively increase all PDropout layers to a target value across epochs."""

    def __init__(self, model: nn.Module, start_value: float, stop_value: float, nr_steps: int):
        self.i = 0
        self.dropout_layers = [
            layer for _, layer in model.named_modules() if isinstance(layer, PDropout)
        ]
        self.drop_values = (stop_value - start_value) / LOG_E * np.log10(
            np.linspace(0, LOG_SPACE_MAX, int(nr_steps)) + 1
        ) + start_value

    def step(self) -> None:
        if self.i < len(self.drop_values):
            value = float(self.drop_values[self.i])
            for layer in self.dropout_layers:
                layer.p = value
        self.i += 1
