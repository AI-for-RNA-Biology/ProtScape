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
    lookup = {}
    for index, name in enumerate(metadata["genes"]):
        lookup.setdefault(name, index)  # same first-record policy as read_data
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
    start at the frozen residue-mean representation. SWE-Simple instead keeps
    random unit slicers and a uniform reference frozen. Standardization is fixed
    from the mean vectors, not recomputed as learned pooling changes.
    """

    def __init__(self, cache_root, mode, hidden_dim=128, residue_budget=32768, manifest_sha256=None,
                 normalization=None, num_ref_points=100):
        super().__init__()
        if mode not in {"mlp", "attention", "swe"}:
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
        # Independent downstream poolers supply training-fold statistics only.
        stats = np.load(root / "normalization.npz") if normalization is None else normalization
        self.register_buffer("feature_mean", torch.from_numpy(stats["mean"]))
        self.register_buffer("feature_std", torch.from_numpy(stats["std"]))
        if mode == "swe":
            self.swe = SlicedWassersteinPooler(self.input_dim, num_ref_points)
        else:
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
        if self.mode == "swe":
            return self.swe(values, lengths)
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


class SlicedWassersteinPooler(nn.Module):
    """SWE-Simple, NaderiAlizadeh & Singh (2025), doi:10.1093/bioadv/vbaf060.

    Freeze unit-length slicers and the projected reference; learn only the shared
    combination over reference points. L=d keeps the output dimension unchanged. Interpolate
    interior quantiles as in PLM_SWE; use the inverse reference permutation from
    Eq. 4 (not the forward permutation in the authors' current implementation).
    Work on unpadded proteins, including length-one sequences, using native
    differentiable Torch operations rather than a custom interpolation backward.
    """

    def __init__(self, input_dim, num_ref_points=100):
        super().__init__()
        self.register_buffer("directions", torch.nn.functional.normalize(torch.randn(input_dim, input_dim), dim=1))
        self.register_buffer("reference", torch.linspace(-1, 1, num_ref_points)[:, None].repeat(1, input_dim))
        self.combination = nn.Parameter(torch.empty(num_ref_points))
        nn.init.uniform_(self.combination, -num_ref_points ** -.5, num_ref_points ** -.5)

    def forward(self, values, lengths):
        slices = torch.nn.functional.linear(values, self.directions)
        reference_ranks = self.reference.argsort(dim=0).argsort(dim=0)
        m = len(self.reference)
        quantile_grid = torch.arange(1, m + 1, device=values.device, dtype=values.dtype) / (m + 1)
        outputs = []
        for protein in slices.split(lengths.tolist()):
            ordered = protein.sort(dim=0).values
            n = len(ordered)
            if n == m:
                quantiles = ordered
            elif n == 1:
                quantiles = ordered.expand(m, -1)
            else:
                position = quantile_grid * (n + 1) - 1
                lower = position.floor().long().clamp(0, n - 2)
                fraction = (position - lower)[:, None]
                # Linear extrapolation at the tails matches PLM_SWE's grid.
                quantiles = ordered[lower] + fraction * (ordered[lower + 1] - ordered[lower])
            displacement = quantiles.gather(0, reference_ranks) - self.reference
            outputs.append((displacement * self.combination[:, None]).sum(0))
        return torch.stack(outputs)
