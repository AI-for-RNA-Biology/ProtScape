"""Shared four-query residue pooling with one optional graph-guided re-read.

PMA/Perceiver-inspired building blocks, not reproductions of either full model.
The caller owns ACM and supplies the same visible adjacency to both passes.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .residue_pooling import ResiduePooler

RECYCLING_MODES = {"slots4", "recycle_memory", "recycle_residues", "recycle_no_graph"}


class RecyclingResiduePooler(ResiduePooler):
    graph_conditioned = True  # enforce pair masking for the matched A–D panel
    recycling = True

    def __init__(self, cache_root, mode, graph_dim, hidden_dim=128,
                 residue_budget=32768, manifest_sha256=None, normalization=None):
        if mode not in RECYCLING_MODES:
            raise ValueError(f"Unknown recycling mode: {mode}")
        super().__init__(cache_root, "attention", hidden_dim, residue_budget,
                         manifest_sha256, normalization)
        del self.V, self.U, self.output
        self.variant = mode
        self.input_projection = nn.Linear(self.input_dim, hidden_dim, bias=False)
        self.queries = nn.Parameter(torch.randn(4, hidden_dim) / math.sqrt(hidden_dim))
        self.output_projection = nn.Linear(4 * hidden_dim, self.input_dim, bias=False)
        nn.init.zeros_(self.output_projection.weight)
        if mode != "slots4":
            self.graph_projection = nn.Linear(graph_dim, hidden_dim, bias=False)
            self.query_update = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            nn.init.zeros_(self.query_update[-1].weight)
            nn.init.zeros_(self.query_update[-1].bias)

    @staticmethod
    def attend(queries, memory, valid=None):
        scores = torch.bmm(queries, memory.transpose(1, 2)) / math.sqrt(memory.shape[-1])
        if valid is not None:
            scores = scores.masked_fill(~valid[:, None], -torch.inf)
        return torch.bmm(scores.softmax(-1), memory)

    def _read_batch(self, ids, queries):
        indices = ids.detach().cpu().numpy()
        starts = self.offsets[indices]
        lengths = self.offsets[indices + 1] - starts
        positions = starts[:, None] + np.arange(int(lengths.max()))
        valid = np.arange(positions.shape[1])[None] < lengths[:, None]
        positions = np.minimum(positions, len(self._residues) - 1)
        device = queries.device
        if self._device_residues is not None:
            values = self._device_residues[torch.as_tensor(positions, device=device)].float()
        else:
            values = torch.from_numpy(np.asarray(self._residues[positions])).to(device).float()
        values = (values - self.feature_mean) / self.feature_std
        projected = self.input_projection(values)
        return self.attend(queries, projected, torch.as_tensor(valid, device=device))

    def read_residues(self, ids, queries=None):
        """Every residue is read; sorted padded batches bound temporary memory.

        Checkpointing recomputes projections on backward. Only frozen residue
        inputs persist across forwards; no graph-dependent result is cached.
        """
        self._load_residues()
        lengths = np.diff(self.offsets)[ids.detach().cpu().numpy()]
        order = np.argsort(lengths, kind="stable")
        if queries is None:
            queries = self.queries[None].expand(len(ids), -1, -1)
        batches, start = [], 0
        for end in range(len(order)):
            if end > start and int(lengths[order[end]]) * (end - start + 1) > self.residue_budget:
                batches.append(order[start:end])
                start = end
        if start < len(order):
            batches.append(order[start:])
        parts = []
        for batch in batches:
            local = torch.as_tensor(batch, device=ids.device)
            args = (ids[local], queries[local])
            if self.training and torch.is_grad_enabled():
                parts.append(checkpoint(self._read_batch, *args, use_reentrant=False, preserve_rng_state=False))
            else:
                parts.append(self._read_batch(*args))
        return torch.cat(parts)[torch.as_tensor(np.argsort(order), device=ids.device)]

    def protein_features(self, slots, sequence_means):
        return sequence_means + self.output_projection(slots.flatten(1))

    def refine(self, ids, first_slots, graph_embeddings):
        context = self.graph_projection(graph_embeddings)
        if self.variant == "recycle_no_graph":
            context = torch.zeros_like(context)
        context = context[:, None].expand(-1, 4, -1)
        queries = self.queries[None] + self.query_update(torch.cat([first_slots, context], -1))
        if self.variant == "recycle_memory":
            return self.attend(queries, first_slots)
        return self.read_residues(ids, queries)
