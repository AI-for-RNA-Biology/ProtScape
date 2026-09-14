"""Context-independent pooling of frozen ESM2 residues before the existing GNN."""
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def attach_residue_ids(ppi_data, ppi_layers, celltype_map, cache_root):
    """Bind sampler node IDs to the cache without relying on local graph numbering."""
    metadata = json.loads((Path(cache_root) / "manifest.json").read_text())
    lookup = {name: index for index, name in enumerate(metadata["genes"])}
    for name, cell in celltype_map.items():
        if cell in ppi_data:
            names = list(ppi_layers[name].nodes())
            if len(names) != ppi_data[cell].num_nodes:
                raise ValueError(f"Residue node order/size mismatch for {name}")
            ppi_data[cell].residue_id = torch.tensor([lookup[gene] for gene in names])


class ResiduePooler(nn.Module):
    """Pool only sampled proteins; deduplicate them across contexts in each step.

    The MLP is a residue-wise residual bottleneck initialized to identity.
    Attention has one gated head, initialized to uniform weights. Both therefore
    start at the frozen residue-mean representation. Standardization is fixed
    from the mean vectors, not recomputed as learned pooling changes.
    """

    def __init__(self, cache_root, mode, hidden_dim=128, residue_budget=32768, manifest_sha256=None):
        super().__init__()
        if mode not in {"mlp", "attention"}:
            raise ValueError(f"Unsupported learned residue pooling: {mode}")
        self.cache_root, self.mode = str(cache_root), mode
        self.residue_budget = int(residue_budget)
        root = Path(cache_root)
        if manifest_sha256 and hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest() != manifest_sha256:
            raise ValueError("Residue cache fingerprint differs from the checkpoint/config")
        manifest = json.loads((root / "manifest.json").read_text())
        if not (root / "COMPLETE.json").is_file():
            raise ValueError("Residue cache is not complete")
        self.offsets = np.load(root / "offsets.npy")
        self.input_dim = int(manifest["embedding_dim"])
        self.n_proteins = len(manifest["genes"])
        stats = np.load(root / "normalization.npz")
        self.register_buffer("feature_mean", torch.from_numpy(stats["mean"]))
        self.register_buffer("feature_std", torch.from_numpy(stats["std"]))
        self.V = nn.Linear(self.input_dim, hidden_dim)
        if mode == "attention":
            self.U = nn.Linear(self.input_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, 1, bias=False)
        else:
            self.output = nn.Linear(hidden_dim, self.input_dim)
            nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.output.weight)
        self._residues = None
        self._evaluation_values = None
        self._evaluated = None

    def __getstate__(self):
        state = super().__getstate__().copy()
        # Full-object restart checkpoints must not copy the frozen residue bank.
        for key in ("_residues", "_evaluation_values", "_evaluated"):
            state[key] = None
        return state

    def train(self, mode=True):
        self._evaluation_values = None
        self._evaluated = None
        return super().train(mode)

    def _load_from_state_dict(self, *args, **kwargs):
        self._evaluation_values = None
        self._evaluated = None
        return super()._load_from_state_dict(*args, **kwargs)

    def pool(self, residues, lengths):
        """Residues contain no BOS/EOS/padding tokens; lengths delimit proteins."""
        values = (residues.float() - self.feature_mean) / self.feature_std
        if self.mode == "mlp":
            values = values + self.output(torch.nn.functional.gelu(self.V(values)))
            return torch.segment_reduce(values, "mean", lengths=lengths)
        scores = self.output(torch.tanh(self.V(values)) * torch.sigmoid(self.U(values))).squeeze(-1)
        maxima = torch.segment_reduce(scores, "max", lengths=lengths)
        weights = torch.exp(scores - torch.repeat_interleave(maxima, lengths))
        totals = torch.segment_reduce(weights, "sum", lengths=lengths)
        weights = weights / torch.repeat_interleave(totals, lengths)
        return torch.segment_reduce(values * weights[:, None], "sum", lengths=lengths)

    def _pool_ids(self, ids):
        if self._residues is None:
            self._residues = np.load(Path(self.cache_root) / "residues.npy", mmap_mode="r")
        protein_ids = ids.detach().cpu().tolist()
        arrays = [self._residues[self.offsets[i]:self.offsets[i + 1]] for i in protein_ids]
        residues = torch.from_numpy(np.concatenate(arrays)).to(self.feature_mean.device)
        lengths = torch.tensor([len(array) for array in arrays], device=residues.device)
        return self.pool(residues, lengths)

    def _pool_chunks(self, ids):
        lengths = np.diff(self.offsets)[ids.detach().cpu().numpy()]
        outputs, start, total = [], 0, 0
        for index, length in enumerate(lengths):
            if index > start and total + length > self.residue_budget:
                outputs.append(self._compute(ids[start:index]))
                start, total = index, 0
            total += int(length)
        if start < len(ids):
            outputs.append(self._compute(ids[start:]))
        return torch.cat(outputs)

    def _compute(self, ids):
        if self.training and torch.is_grad_enabled():
            # Reload frozen residues on backward instead of retaining every bag.
            return checkpoint(self._pool_ids, ids, use_reentrant=False, preserve_rng_state=False)
        return self._pool_ids(ids)

    def forward(self, protein_ids):
        if protein_ids is None:
            raise ValueError("Residue pooling needs sampler-aligned residue_id values")
        ids, inverse = torch.unique(protein_ids.long(), sorted=True, return_inverse=True)
        if self.training:
            return self._pool_chunks(ids)[inverse]
        if self._evaluation_values is None:
            self._evaluation_values = self.feature_mean.new_empty((self.n_proteins, self.input_dim))
            self._evaluated = torch.zeros(self.n_proteins, dtype=torch.bool, device=ids.device)
        missing = ids[~self._evaluated[ids]]
        if len(missing):
            with torch.no_grad():
                self._evaluation_values[missing] = self._pool_chunks(missing)
                self._evaluated[missing] = True
        return self._evaluation_values[protein_ids.long()]
