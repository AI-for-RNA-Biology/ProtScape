# stratification.py
# Stratified sampling and fold building utilities

from typing import Iterator, List, Optional, Tuple
import numpy as np
from torch.utils.data import Sampler
from sklearn.cluster import KMeans, MiniBatchKMeans


class MultilabelStratifiedSampler(Sampler[int]):
    """
    Stratified batch sampler for multilabel classification.

    Clusters samples based on their label patterns (or precomputed clusters)
    and ensures each batch contains representative samples from each cluster.
    This helps maintain balanced distributions across training batches.
    """

    def __init__(
        self,
        labels: Optional[np.ndarray],
        batch_size: int,
        n_clusters: int = 10,
        shuffle: bool = True,
        seed: int = 42,
        cluster_labels: Optional[np.ndarray] = None,
    ):
        """
        Args:
            labels: Binary label matrix of shape (n_samples, n_labels)
            batch_size: Number of samples per batch
            n_clusters: Number of label clusters (default 10)
            shuffle: Whether to shuffle within clusters each epoch
            seed: Random seed for deterministic splitting
            cluster_labels: Optional precomputed cluster ids for each sample
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._iter_count = 0

        if cluster_labels is not None:
            self.cluster_labels = np.asarray(cluster_labels)
            self.n_samples = len(self.cluster_labels)
            if self.n_samples == 0:
                self.n_clusters = 0
                self.cluster_indices = []
                return
            unique = np.unique(self.cluster_labels)
            remap = {int(old): int(new) for new, old in enumerate(unique)}
            self.cluster_labels = np.array([remap[int(x)] for x in self.cluster_labels], dtype=int)
            self.n_clusters = len(unique)
        else:
            if labels is None:
                raise ValueError("labels or cluster_labels must be provided")
            self.labels = np.asarray(labels)
            self.n_samples = len(self.labels)
            n_clusters = min(n_clusters, self.n_samples)
            if n_clusters < 2 or self.n_samples < n_clusters:
                self.cluster_labels = np.zeros(self.n_samples, dtype=int)
                self.n_clusters = 1
            else:
                kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
                self.cluster_labels = kmeans.fit_predict(self.labels)
                self.n_clusters = n_clusters

        # Group sample indices by cluster
        self.cluster_indices = [
            np.where(self.cluster_labels == c)[0] for c in range(self.n_clusters)
        ]

    def __iter__(self) -> Iterator[int]:
        if self.n_samples == 0 or self.n_clusters == 0:
            return iter([])
        rng = np.random.default_rng(self.seed + self._iter_count)
        self._iter_count += 1

        # Shuffle within each cluster
        cluster_indices = [
            rng.permutation(idx) if self.shuffle else idx.copy()
            for idx in self.cluster_indices
        ]

        # Build batches by interleaving from clusters
        indices = []
        pointers = [0] * self.n_clusters

        while True:
            batch = []
            # Try to get samples from each cluster
            for c in range(self.n_clusters):
                if pointers[c] < len(cluster_indices[c]):
                    n_from_cluster = max(1, self.batch_size // self.n_clusters)
                    end_idx = min(pointers[c] + n_from_cluster, len(cluster_indices[c]))
                    batch.extend(cluster_indices[c][pointers[c]:end_idx].tolist())
                    pointers[c] = end_idx

            if not batch:
                break

            # Shuffle the batch to mix clusters
            if self.shuffle:
                rng.shuffle(batch)
            indices.extend(batch)

        return iter(indices)

    def __len__(self) -> int:
        return self.n_samples


def _build_context_matrix(cell_ids_per_bag: List[List[str]]) -> Optional[np.ndarray]:
    """
    Build a binary context matrix from cell IDs per sample.

    Args:
        cell_ids_per_bag: List of cell ID lists, one per sample

    Returns:
        Binary matrix of shape (n_samples, n_unique_contexts) or None if empty
    """
    all_ctx = sorted({str(cid) for bag in cell_ids_per_bag for cid in bag})
    if not all_ctx:
        return None
    ctx_to_col = {c: j for j, c in enumerate(all_ctx)}
    X = np.zeros((len(cell_ids_per_bag), len(all_ctx)), dtype=np.float32)
    for i, bag in enumerate(cell_ids_per_bag):
        for cid in set(bag):
            j = ctx_to_col.get(str(cid))
            if j is not None:
                X[i, j] = 1.0
    return X


def _count_unique_contexts(X_ctx: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Count unique context patterns per cluster."""
    counts = np.zeros(n_clusters, dtype=int)
    for i in range(n_clusters):
        members = np.where(labels == i)[0]
        if members.size:
            counts[i] = np.unique(X_ctx[members], axis=0).shape[0]
    return counts


def _merge_small_label_clusters(
    X_labels: np.ndarray,
    labels: np.ndarray,
    min_size: int,
    X_ctx: Optional[np.ndarray] = None,
    min_ctx: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge clusters that are too small or have too few contexts."""
    labels = labels.copy()
    while True:
        kept = sorted(set(labels.tolist()))
        remap = {old: new for new, old in enumerate(kept)}
        labels = np.array([remap[int(x)] for x in labels], dtype=int)
        counts = np.bincount(labels)
        if len(counts) <= 1:
            break
        if X_ctx is not None and min_ctx is not None:
            ctx_counts = _count_unique_contexts(X_ctx, labels, len(counts))
            small = np.where((counts < min_size) | (ctx_counts < min_ctx))[0]
        else:
            small = np.where(counts < min_size)[0]
        if len(small) == 0:
            break
        centers = np.vstack([X_labels[labels == i].mean(axis=0) for i in range(len(counts))])
        if X_ctx is not None and min_ctx is not None:
            order = np.lexsort((counts[small], ctx_counts[small]))
            c = small[int(order[0])]
        else:
            c = small[int(np.argmin(counts[small]))]
        candidates = [i for i in range(len(counts)) if i != c]
        if not candidates:
            break
        d = np.linalg.norm(centers[candidates] - centers[c], axis=1)
        tgt = candidates[int(d.argmin())]
        labels[labels == c] = tgt
    kept = sorted(set(labels.tolist()))
    remap = {old: new for new, old in enumerate(kept)}
    labels = np.array([remap[int(x)] for x in labels], dtype=int)
    centers = np.vstack([X_labels[labels == i].mean(axis=0) for i in range(len(kept))])
    return labels, centers


def _compute_label_clusters(Y_ref: np.ndarray, k_label: int, seed: int) -> np.ndarray:
    """
    Cluster samples by label patterns.

    For single-label binary targets (shape n x 1), use the true class labels
    directly so context stratification remains label-aware (0 vs 1).
    """
    Y_arr = np.asarray(Y_ref)
    if Y_arr.ndim != 2:
        raise ValueError(f"Expected Y_ref with shape (n_samples, n_labels), got {Y_arr.shape}")

    # Binary case: cluster by class label directly.
    if Y_arr.shape[1] == 1:
        return (Y_arr[:, 0] > 0).astype(int, copy=False)

    km = MiniBatchKMeans(n_clusters=k_label, random_state=seed, batch_size=1024, n_init=10)
    return km.fit_predict(Y_arr)


def build_context_stratified_folds(
    Y_ref: np.ndarray,
    cell_ids_per_bag: List[List[str]],
    *,
    seed: int,
    k_label: Optional[int],
    n_splits: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Build context-stratified folds ensuring each fold has samples from diverse contexts.

    The algorithm:
    1. Cluster samples by label patterns
    2. Merge small clusters to ensure min samples/contexts per cluster
    3. Within each label cluster, sub-cluster by context patterns
    4. Distribute samples across folds using round-robin to ensure all folds populated

    Args:
        Y_ref: Label matrix of shape (n_samples, n_labels)
        cell_ids_per_bag: List of cell ID lists for each sample
        seed: Random seed
        k_label: Number of label clusters (None for auto)
        n_splits: Number of folds

    Returns:
        Tuple of (folds, label_clusters) where folds is a list of index arrays
    """
    rng = np.random.RandomState(seed)
    n = Y_ref.shape[0]
    if n_splits < 2:
        raise ValueError("Need at least 2 samples for CV.")
    k_label = int(max(1, min(k_label if k_label is not None else Y_ref.shape[1], n)))

    X_ctx = _build_context_matrix(cell_ids_per_bag)
    if X_ctx is None:
        raise ValueError("No context IDs available for context stratification.")

    # Precompute exact row keys once to avoid repeated hash/tuple conversion.
    row_keys = [X_ctx[i].tobytes() for i in range(X_ctx.shape[0])]
    n_unique_ctx = len(set(row_keys))
    if n_unique_ctx < n_splits:
        raise ValueError(f"Need at least {n_splits} unique contexts, found {n_unique_ctx}.")

    label_clusters = _compute_label_clusters(Y_ref, k_label, seed)
    label_clusters, _ = _merge_small_label_clusters(Y_ref, label_clusters, n_splits, X_ctx, n_splits)
    ctx_counts = _count_unique_contexts(X_ctx, label_clusters, len(np.unique(label_clusters)))
    if np.any(ctx_counts < n_splits):
        raise ValueError("Context stratification failed to ensure enough context diversity per label cluster.")

    folds: List[List[int]] = [[] for _ in range(n_splits)]
    unique_lcs = np.unique(label_clusters)
    for lc_idx, lc in enumerate(unique_lcs):
        members = np.where(label_clusters == lc)[0]
        if len(members) == 0:
            continue
        if len(members) < n_splits:
            raise ValueError("Context stratification requires at least n_splits samples per label cluster.")

        # Count unique context patterns inside this label cluster.
        ctx_unique = len({row_keys[m] for m in members})
        if ctx_unique < n_splits:
            raise ValueError(
                "Context stratification requires at least n_splits unique contexts per label cluster. "
                "Merge small label clusters or reduce n_splits."
            )

        # Cap k_ctx to avoid empty clusters
        k_ctx = min(int(ctx_unique), len(members))
        if k_ctx > 1 and len(members) // k_ctx < 1:
            k_ctx = max(1, len(members))

        # When k_ctx equals the number of unique context patterns, we can group
        # directly by exact pattern and skip KMeans.
        if k_ctx == ctx_unique:
            groups_dict = {}
            for m in members:
                groups_dict.setdefault(row_keys[m], []).append(int(m))
            groups = [np.asarray(g, dtype=int) for g in groups_dict.values()]
        else:
            kmc = KMeans(n_clusters=k_ctx, random_state=seed, n_init=10)
            sub = kmc.fit_predict(X_ctx[members])
            groups = [members[sub == j] for j in range(k_ctx) if np.sum(sub == j) > 0]

        if not groups:
            groups = [members]

        # Collect all samples and distribute round-robin
        all_samples_lc = []
        for g in groups:
            g_arr = np.array(g).copy()
            rng.shuffle(g_arr)
            all_samples_lc.extend(g_arr.tolist())

        # Use a randomized start fold per label cluster.
        # Without this, every cluster's remainder is always assigned to fold 0,
        # which systematically inflates early fold sizes.
        start_fold = int(rng.randint(0, n_splits))
        for i, sample_idx in enumerate(all_samples_lc):
            fold_idx = (start_fold + i) % n_splits
            folds[fold_idx].append(sample_idx)

    out_folds = []
    for fold_idx in range(n_splits):
        test_idx = np.array(sorted(set(folds[fold_idx])), dtype=int)
        if test_idx.size == 0:
            raise ValueError("Context-stratified split produced empty folds.")
        out_folds.append(test_idx)
    return out_folds, label_clusters


class MultilabelStratifiedKFold:
    """
    Fallback implementation of MultilabelStratifiedKFold.

    Attempts to stratify splits based on multi-label distribution.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: Optional[int] = None):
        n_splits = int(n_splits)
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.shuffle = bool(shuffle)
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y, groups=None):
        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            raise ValueError("MultilabelStratifiedKFold expects multilabel y with shape (n_samples, n_labels)")
        if y_arr.ndim != 2:
            raise ValueError(f"Expected y with 2 dimensions, got {y_arr.ndim}")

        y_bin = (y_arr > 0).astype(np.int8, copy=False)
        n_samples, n_labels = y_bin.shape
        if n_samples < self.n_splits:
            raise ValueError(f"Cannot have n_splits={self.n_splits} greater than n_samples={n_samples}.")

        rng = np.random.default_rng(self.random_state) if self.shuffle else None

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        fold_counts = np.zeros((self.n_splits, n_labels), dtype=int)
        fold_ns = np.zeros(self.n_splits, dtype=int)
        desired = y_bin.sum(axis=0).astype(np.float64) / self.n_splits

        sample_pos = [np.flatnonzero(y_bin[i]) for i in range(n_samples)]
        label_freq = y_bin.sum(axis=0)
        max_freq = int(label_freq.max()) if label_freq.size else 0
        rarity = np.array([int(label_freq[p].min()) if p.size else (max_freq + 1) for p in sample_pos], dtype=int)
        n_pos = np.array([p.size for p in sample_pos], dtype=int)

        if rng is not None:
            perm = rng.permutation(n_samples)
            order = perm[np.lexsort((-n_pos[perm], rarity[perm]))]
        else:
            order = np.arange(n_samples)[np.lexsort((-n_pos, rarity))]

        folds = [[] for _ in range(self.n_splits)]
        for idx in order:
            pos = sample_pos[idx]
            candidates = np.flatnonzero(fold_ns < fold_sizes)
            if candidates.size == 0:
                candidates = np.arange(self.n_splits)

            if pos.size:
                need = desired[pos] - fold_counts[:, pos]
                scores = need.sum(axis=1)
            else:
                scores = np.zeros(self.n_splits, dtype=np.float64)

            score_masked = np.full(self.n_splits, -np.inf, dtype=np.float64)
            score_masked[candidates] = scores[candidates]
            score_masked[candidates] += (fold_sizes[candidates] - fold_ns[candidates]) * 1e-3

            best_score = float(np.max(score_masked))
            best = np.flatnonzero(score_masked == best_score)
            if best.size > 1:
                cap = (fold_sizes[best] - fold_ns[best])
                best = best[cap == cap.max()]
            chosen = int(rng.choice(best) if (rng is not None and best.size > 1) else best[0])

            folds[chosen].append(int(idx))
            fold_ns[chosen] += 1
            if pos.size:
                fold_counts[chosen, pos] += 1

        all_idx = np.arange(n_samples)
        for fold in range(self.n_splits):
            test_idx = np.array(folds[fold], dtype=int)
            mask = np.zeros(n_samples, dtype=bool)
            mask[test_idx] = True
            train_idx = all_idx[~mask]
            yield train_idx, test_idx
