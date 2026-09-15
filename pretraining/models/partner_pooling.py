"""Observed-partner queries over frozen focal residues, before the CF ACM.

No candidate-edge inputs, RNA, cell vectors, or graph-derived persistent caches.
All graph information comes from the exact message graph supplied to encode().
"""
import math

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .residue_pooling import ResiduePooler

PARTNER_MODES = {
    "partner", "self_query", "post_fusion", "partner_dispersion",
    "partner_mean_mlp", "partner_slots", "pma4",
}


def neighbour_rows(edges, n_nodes, cap=None, seed=0):
    """Unique visible, non-self neighbours; uniform without-replacement cap.

    A private generator keeps sampling out of the masking/dropout RNG stream.
    The caller passes a symmetric pair-masked graph. Never symmetrise a hidden
    candidate edge here, and never look at the original reference adjacency.
    """
    pairs = edges.detach().cpu().numpy().T
    pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)
    if cap is not None and len(pairs):
        rng = np.random.default_rng(seed)
        order = np.lexsort((rng.random(len(pairs)), pairs[:, 0]))
        pairs = pairs[order]
        counts = np.bincount(pairs[:, 0], minlength=n_nodes)
        starts = np.r_[0, counts.cumsum()[:-1]]
        ranks = np.arange(len(pairs)) - np.repeat(starts, counts)
        pairs = pairs[ranks < cap]
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    counts = np.bincount(pairs[:, 0], minlength=n_nodes) if len(pairs) else np.zeros(n_nodes, dtype=int)
    return np.split(pairs[:, 1], counts.cumsum()[:-1])


class PartnerResiduePooler(ResiduePooler):
    """One-head additive gated/query attention, preserving D-dimensional values.

    Four-slot queries are a PMA-inspired adaptation, not a reproduction of the
    complete Set Transformer. Dispersion is population SD across partner views;
    partner_mean_mlp supplies its exactly parameter-matched capacity control.
    """
    graph_conditioned = True

    def __init__(self, cache_root, mode, hidden_dim=128, residue_budget=32768,
                 manifest_sha256=None, normalization=None, max_partners=16,
                 query_chunk_size=32):
        if mode not in PARTNER_MODES:
            raise ValueError(f"Unsupported partner pooler: {mode}")
        super().__init__(cache_root, "attention", hidden_dim, residue_budget,
                         manifest_sha256, normalization)
        self.variant = mode
        self.max_partners = int(max_partners)
        self.query_chunk_size = int(query_chunk_size)
        if self.max_partners < 1 or self.query_chunk_size < 1:
            raise ValueError("Partner and query chunk sizes must be positive")
        self.norm = nn.LayerNorm(self.input_dim, elementwise_affine=False)
        self.register_buffer("sampling_step", torch.zeros((), dtype=torch.long))
        if mode != "post_fusion":
            self.key = nn.Linear(self.input_dim, hidden_dim, bias=False)
            query_dim = hidden_dim if mode == "partner_slots" else self.input_dim
            self.query = nn.Linear(query_dim, hidden_dim, bias=False)
            nn.init.zeros_(self.query.weight)  # keys remain random: nonzero initial query gradient
        if mode in {"partner_slots", "pma4"}:
            self.slot_seeds = nn.Parameter(torch.randn(4, hidden_dim) / math.sqrt(hidden_dim))
            self.slot_values = nn.Linear(self.input_dim, hidden_dim, bias=False)
        if mode == "pma4":
            del self.V, self.U, self.output, self.query
            self.slot_output = nn.Linear(4 * hidden_dim, self.input_dim)
            nn.init.zeros_(self.slot_output.weight)
            nn.init.zeros_(self.slot_output.bias)
        if mode in {"post_fusion", "partner_dispersion", "partner_mean_mlp"}:
            # Fusion's extra parameter budget approximately matches Q/K;
            # dispersion and its mean-input control match exactly.
            width = max(1, 2 * hidden_dim // 3) if mode == "post_fusion" else hidden_dim
            dim = 2 * self.input_dim if mode == "post_fusion" else self.input_dim
            self.residual = nn.Sequential(nn.Linear(dim, width), nn.GELU(), nn.Linear(width, self.input_dim))
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

    def _batches(self, ids, query_counts=None):
        lengths = np.diff(self.offsets)[ids.detach().cpu().numpy()]
        buckets = np.zeros(len(ids), dtype=int) if query_counts is None else np.ceil(np.log2(np.maximum(1, query_counts))).astype(int)
        order = np.lexsort((lengths, buckets))
        batches, start = [], 0
        for end in range(len(order)):
            if end > start and (buckets[order[end]] != buckets[order[start]] or int(lengths[order[end]]) * (end - start + 1) > self.residue_budget):
                batches.append(order[start:end])
                start = end
        if start < len(order):
            batches.append(order[start:])
        return batches

    def _values(self, ids):
        indices = ids.detach().cpu().numpy()
        starts = self.offsets[indices]
        lengths = self.offsets[indices + 1] - starts
        positions = starts[:, None] + np.arange(int(lengths.max()))
        valid = np.arange(positions.shape[1])[None, :] < lengths[:, None]
        positions = np.minimum(positions, len(self._residues) - 1)
        device = self.feature_mean.device
        if getattr(self, "_device_residues", None) is not None:
            values = self._device_residues[torch.as_tensor(positions, device=device)].float()
        else:
            values = torch.from_numpy(np.asarray(self._residues[positions])).to(device).float()
        values = (values - self.feature_mean) / self.feature_std
        return values, torch.as_tensor(valid, device=device)

    def _slot_batch(self, ids):
        values, valid = self._values(ids)
        normalized = self.norm(values)
        scores = torch.einsum("qh,blh->bql", self.slot_seeds, self.key(normalized)) / math.sqrt(self.key.out_features)
        weights = scores.masked_fill(~valid[:, None, :], -torch.inf).softmax(-1)
        return torch.bmm(weights, self.slot_values(normalized))

    def _compute_batch(self, fn, *args):
        if self.training and torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=False)
        return fn(*args)

    def _pool_batch(self, ids, queries, query_valid, has_partners):
        values, valid = self._values(ids)
        scores = self.output(torch.tanh(self.V(values)) * torch.sigmoid(self.U(values))).squeeze(-1)
        keys = self.key(self.norm(values))
        # Bound eval memory even for high-degree proteins. Every permitted
        # neighbour is evaluated; chunks change memory use, not the estimator.
        total = values.new_zeros((len(ids), self.input_dim))
        running_mean = torch.zeros_like(total)
        centered_squares = torch.zeros_like(total)
        seen = values.new_zeros((len(ids), 1))
        count = query_valid.sum(1).clamp_min(1).to(values.dtype)[:, None]
        for start in range(0, queries.shape[1], self.query_chunk_size):
            q = queries[:, start:start + self.query_chunk_size]
            mask = query_valid[:, start:start + self.query_chunk_size]
            logits = scores[:, None, :] + torch.bmm(q, keys.transpose(1, 2)) / math.sqrt(keys.shape[-1])
            weights = logits.masked_fill(~valid[:, None, :], -torch.inf).softmax(-1)
            views = torch.bmm(weights, values) * mask[:, :, None]
            # Four slot views are averaged WITHIN each partner before computing
            # between-partner statistics (slots mode itself reports mean only).
            total = total + views.sum(1)
            if self.variant == "partner_dispersion":
                # Merge centred moments: E[x²]-E[x]² is unstable when partner
                # views coincide (including the gated baseline at initialization).
                added = mask.sum(1)[:, None].to(values.dtype)
                chunk_mean = views.sum(1) / added.clamp_min(1)
                chunk_m2 = ((views - chunk_mean[:, None]).square() * mask[:, :, None]).sum(1)
                delta = chunk_mean - running_mean
                updated = seen + added
                centered_squares = centered_squares + chunk_m2 + delta.square() * seen * added / updated.clamp_min(1)
                running_mean = running_mean + delta * added / updated.clamp_min(1)
                seen = updated
        mean = total / count
        if self.variant == "partner_dispersion":
            variance = (centered_squares / count).clamp_min(0)
            sd = (variance + 1e-8).sqrt() - 1e-4
            sd = torch.where(count > 1, sd, torch.zeros_like(sd))
            return mean + self.residual(sd) * has_partners[:, None]
        if self.variant == "partner_mean_mlp":
            return mean + self.residual(mean) * has_partners[:, None]
        return mean

    def _gated_batch(self, ids):
        values, valid = self._values(ids)
        scores = self.output(torch.tanh(self.V(values)) * torch.sigmoid(self.U(values))).squeeze(-1)
        weights = scores.masked_fill(~valid, -torch.inf).softmax(-1)
        return torch.bmm(weights[:, None], values).squeeze(1)

    def forward(self, protein_ids, message_edges, sequence_means):
        if len(torch.unique(protein_ids)) != len(protein_ids):
            raise ValueError("CF node IDs must be unique within a message graph")
        self._load_residues()
        batches = self._batches(protein_ids)
        device = protein_ids.device
        slot_bank = None
        if self.variant in {"partner_slots", "pma4"}:
            slot_parts = [self._compute_batch(self._slot_batch, protein_ids[torch.as_tensor(b, device=device)]) for b in batches]
            order = np.concatenate(batches)
            slot_bank = torch.cat(slot_parts)[torch.as_tensor(np.argsort(order), device=device)]
            if self.variant == "pma4":
                return sequence_means + self.slot_output(slot_bank.flatten(1))
        rows = None
        if self.variant != "self_query":
            step = int(self.sampling_step.item())
            rows = neighbour_rows(message_edges, len(protein_ids), self.max_partners if self.training else None, seed=step)
            if self.training:
                self.sampling_step.add_(1)
            # Degree buckets bound padding overhead for full-neighbour evaluation.
            batches = self._batches(protein_ids, [len(row) for row in rows])
        parts = []
        for batch in batches:
            local = torch.as_tensor(batch, device=device)
            ids = protein_ids[local]
            if self.variant == "self_query":
                queries = self.query(self.norm(sequence_means[local]))[:, None]
                query_valid = torch.ones((len(batch), 1), dtype=torch.bool, device=device)
                has_partners = torch.ones(len(batch), dtype=torch.bool, device=device)
            else:
                width = max(1, max(len(rows[i]) for i in batch))
                neighbours = np.full((len(batch), width), -1, dtype=np.int64)
                for i, node in enumerate(batch):
                    neighbours[i, :len(rows[node])] = rows[node]
                present = torch.as_tensor(neighbours >= 0, device=device)
                has_partners = present.any(1)
                indices = torch.as_tensor(neighbours.clip(min=0), device=device)
                if self.variant == "post_fusion":
                    means = (sequence_means[indices] * present[:, :, None]).sum(1) / present.sum(1).clamp_min(1)[:, None]
                    gated = self._compute_batch(self._gated_batch, ids)
                    fusion = self.residual(torch.cat([gated, means], 1))
                    parts.append(gated + fusion * present.any(1)[:, None])
                    continue
                if self.variant == "partner_slots":
                    queries = self.query(slot_bank[indices]).flatten(1, 2)
                    query_valid = present.repeat_interleave(4, dim=1)
                else:
                    queries = self.query(self.norm(sequence_means[indices]))
                    query_valid = present
                queries = queries * query_valid[:, :, None]
                # A zero query recovers the gated pooler for isolated nodes.
                query_valid = query_valid.clone()
                query_valid[~present.any(1), 0] = True
            parts.append(self._compute_batch(self._pool_batch, ids, queries, query_valid, has_partners))
        order = np.concatenate(batches)
        return torch.cat(parts)[torch.as_tensor(np.argsort(order), device=device)]
