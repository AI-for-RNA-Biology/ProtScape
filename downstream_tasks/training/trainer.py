"""Train linear probes and attention models on the same five validation folds."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import Config
from ..models.registry import ModelVariant, ModelType, ClassifierType
from ..models.abmil import ABMIL_ContextOnly, ABMIL_LateFusion
from ..models.pdl import LinearScheduler
from ..models.linear import LinearProbe
from ..data.loaders import EmbeddingLoader
from ..data.datasets import ABMILDataset, collate_abmil
from ..data.preprocessing import (
    zscore_normalize_bags,
    compute_class_weights,
)
from ..utils.stratification import MultilabelStratifiedSampler
from ..utils.io_utils import save_model_weights, load_model_weights, model_weights_exist
from .cv_utils import SplitPlan, build_cv_splits, get_test_indices, get_cv_train_val_indices
from .metrics import compute_all_metrics, summarize_cv_metrics
from .history import TrainingHistory, plot_training_curve, plot_training_curves, save_histories_csv


def json_compatible(value: Any) -> Any:
    """Convert NumPy values to ordinary numbers and lists for saved metrics."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_compatible(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_compatible(v) for v in value]
    return value


class Trainer:
    """Build features, fit five fold models, and evaluate the held-out proteins."""

    def __init__(
        self,
        config: Config,
        embedding_loader: EmbeddingLoader,
        output_dir: Path,
        device: Optional[torch.device] = None,
        force: bool = False,
    ):
        self.config = config
        self.loader = embedding_loader
        self.output_dir = output_dir
        self.force = force
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Enable GPU optimizations
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.loader_num_workers = min(4, os.cpu_count() or 1) if self.device.type == "cuda" else 0
        self.loader_pin_memory = self.device.type == "cuda"

    def train_and_evaluate(
        self,
        variant: ModelVariant,
        genes: List[str],
        Y: np.ndarray,
        split_plan: Optional[SplitPlan] = None,
        strict_gene_universe: bool = False,
    ) -> Dict:
        """Train and evaluate a model variant with cross-validation."""
        print(f"\n{'='*60}")
        print(f"Training: {variant.name}")
        print(f"{'='*60}")

        try:
            features = self.build_features(variant, genes)
        finally:
            self.loader.clear_cache()

        # Keep labels in the same order when a feature source omits proteins.
        filtered_genes = features["genes"]
        if len(filtered_genes) < len(genes):
            if strict_gene_universe:
                filtered_set = set(g.upper() for g in filtered_genes)
                missing = [g for g in genes if g.upper() not in filtered_set]
                preview = ", ".join(missing[:10])
                raise ValueError(
                    "Shared gene universe mismatch: feature builder dropped genes "
                    f"for variant {variant.name}. Missing examples: {preview}"
                )
            gene_to_orig_idx = {g.upper(): i for i, g in enumerate(genes)}
            filtered_idx = [gene_to_orig_idx[g.upper()] for g in filtered_genes]
            Y = Y[filtered_idx]
            genes = filtered_genes
            print(f"[INFO] Filtered Y to {len(genes)} samples to match available embeddings")

        if split_plan is None:
            # Determine if context stratification should be used
            use_context_strat = variant.use_context_stratification and features.get("cell_ids") is not None

            split_plan = build_cv_splits(
                Y,
                cell_ids_per_bag=features.get("cell_ids") if use_context_strat else None,
                use_context_split=use_context_strat,
                k_label=None,
                n_splits=self.config.n_folds,
                seed=self.config.seed,
            )
        else:
            # Validate compatibility of shared split plan with current features.
            n_samples = len(genes)
            max_fold_idx = max((int(np.max(f)) for f in split_plan.folds if len(f) > 0), default=-1)
            if max_fold_idx >= n_samples:
                raise ValueError(
                    f"Shared split plan expects >= {max_fold_idx + 1} samples, but got {n_samples}."
                )
            if split_plan.label_clusters is not None and len(split_plan.label_clusters) != n_samples:
                raise ValueError(
                    "Shared split plan label clusters are misaligned with current sample count."
                )
            print("[INFO] Using precomputed shared split plan")
            
        # Check for existing weights (use n_folds - 1 since fold 0 is held out for test)
        n_cv_folds = self.config.n_folds - 1  # 5 CV folds from 6 total
        if not self.force and model_weights_exist(self.output_dir, n_cv_folds):
            print(f"[SKIP] Weights exist for {variant.name}, running inference only")
            return self.evaluate_saved_models(variant, features, Y, split_plan)

        if variant.model_type == ModelType.LR:
            return self.train_linear_folds(variant, features, Y, split_plan)
        else:
            return self.train_attention_folds(variant, features, Y, split_plan)

    def binary_loss(self, Y_train: np.ndarray) -> nn.Module:
        """Create the configured binary classification loss."""
        if self.config.use_class_weights:
            pos_weight = compute_class_weights(Y_train)
            pos_weight_t = torch.from_numpy(pos_weight).to(self.device)
        else:
            pos_weight_t = None

        return nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

    def abmil_hidden_dim(self, variant: ModelVariant) -> Optional[int]:
        """Resolve MLP hidden size for ABMIL models."""
        if variant.classifier_type != ClassifierType.MLP:
            return None
        if variant.mlp_hidden_dim is not None:
            return int(variant.mlp_hidden_dim)
        return int(self.config.mil_mlp_dim)

    def selection_metric_key(self) -> str:
        """Choose the validation score used for early stopping."""
        metric = str(getattr(self.config, "train_selection_metric", "auprc")).strip().lower()
        if metric == "f1":
            return "f1_macro"
        return "auprc_macro"

    def pdl_projection_mode(self, variant: ModelVariant) -> str:
        """Use the model variant's PDL projection setting when specified."""
        return variant.resolve_pdl_proj_mode(self.config.pdl_proj_mode)

    def pdl_projection_layers(self, variant: ModelVariant) -> int:
        """Count identity projection layers before attention pooling."""
        if self.pdl_projection_mode(variant) != "identity":
            return 0
        return self.config.pdl_proj_layers

    def build_abmil_model(self, model_params: Dict) -> nn.Module:
        """Restore an attention model from its saved architecture settings."""
        params = {
            key: model_params[key]
            for key in (
                "num_classes", "ctx_dim", "num_heads", "att_hidden", "dropout",
                "attention_type", "classifier_type", "mlp_hidden_dim",
            )
        }
        params.update(
            att_dropout=model_params.get("att_dropout", 0.0),
            use_pdl=model_params.get("use_pdl", False),
            pdl_proj_mode=model_params.get("pdl_proj_mode", "identity"),
            pdl_proj_layers=model_params.get("pdl_proj_layers", 2),
        )
        if model_params.get("use_ext_embed", True):
            model = ABMIL_LateFusion(esm_dim=model_params["esm_dim"], **params)
        else:
            model = ABMIL_ContextOnly(**params)
        return model.to(self.device)

    def abmil_loader_options(self) -> Dict:
        """Collate variable-length context bags in the training process."""
        return {
            "batch_size": self.config.batch_size,
            "collate_fn": collate_abmil,
            "num_workers": 0,
            "pin_memory": False,
        }

    def build_features(
        self,
        variant: ModelVariant,
        genes: List[str],
    ) -> Dict:
        """Align proteins with sequence vectors and their available cell contexts.

        Linear probes receive pooled features; ABMIL receives one context bag per protein.
        """
        sources = variant.embedding_sources
        gene_set = set(g.upper() for g in genes)
        result = {}

        # Sequence (external) features
        X_esm = None
        esm_dict = None
        if "ext_embed" in sources:
            esm_dict = self.loader.load_esm()
            valid_genes = [g for g in genes if g.upper() in esm_dict]
            if len(valid_genes) < len(genes):
                print(
                    f"[INFO] Filtering to {len(valid_genes)}/{len(genes)} genes with sequence embeddings"
                )
                genes = valid_genes
                gene_set = set(g.upper() for g in genes)
            X_esm = np.stack([esm_dict[g.upper()] for g in genes]).astype(np.float32)
            result["esm"] = X_esm

        # Context features (HC, Cell)
        ctx_vecs = None
        cell_ids = None
        ctx_means = None

        if "hc" in sources and "cell" in sources:
            # HC with cell embeddings concatenated
            if variant.model_type == ModelType.LR:
                gene_to_means, gene_to_cells = self.loader.load_hc_with_cell_mean(gene_set)
                valid_genes = [g for g in genes if g.upper() in gene_to_means]
            else:
                gene_to_vecs, gene_to_cells = self.loader.load_hc_with_cell(gene_set)
                valid_genes = [g for g in genes if g.upper() in gene_to_vecs]
            if len(valid_genes) < len(genes):
                print(f"[INFO] Filtering to {len(valid_genes)}/{len(genes)} genes with HC+Cell embeddings")
                genes = valid_genes
            cell_ids = [gene_to_cells.get(g.upper(), ["unknown"]) for g in genes]
            if variant.model_type == ModelType.LR:
                ctx_means = gene_to_means
            else:
                ctx_vecs = gene_to_vecs

        elif "hc" in sources:
            # HC without cell
            if variant.model_type == ModelType.LR:
                gene_to_means, gene_to_cells = self.loader.load_hc_mean(gene_set)
                valid_genes = [g for g in genes if g.upper() in gene_to_means]
            else:
                gene_to_vecs, gene_to_cells = self.loader.load_hc(gene_set)
                valid_genes = [g for g in genes if g.upper() in gene_to_vecs]
            if len(valid_genes) < len(genes):
                print(f"[INFO] Filtering to {len(valid_genes)}/{len(genes)} genes with HC embeddings")
                genes = valid_genes
            cell_ids = [gene_to_cells.get(g.upper(), ["unknown"]) for g in genes]

            if variant.model_type == ModelType.LR:
                ctx_means = gene_to_means
            else:
                ctx_vecs = gene_to_vecs

        if cell_ids is not None:
            result["cell_ids"] = cell_ids

        # Keep ESM aligned with the final filtered gene order.
        if X_esm is not None and len(X_esm) != len(genes):
            X_esm = np.stack([esm_dict[g.upper()] for g in genes]).astype(np.float32)
            result["esm"] = X_esm

        # Build context bags for ABMIL
        if ctx_vecs is not None and variant.model_type == ModelType.ABMIL:
            ctx_bags = []
            for g in genes:
                vecs = ctx_vecs[g.upper()]
                ctx_bags.append(np.stack(vecs, axis=0).astype(np.float32))
            result["ctx_bags"] = ctx_bags

        # Pool for LR models
        if variant.model_type == ModelType.LR:
            if ctx_means is not None:
                result["X_ctx"] = np.stack([ctx_means[g.upper()] for g in genes]).astype(np.float32)

        # Build final feature matrix for LR only
        if variant.model_type == ModelType.LR:
            X_parts = []
            if X_esm is not None:
                X_parts.append(X_esm)
                result["esm"] = X_esm

            if "X_ctx" in result:
                X_parts.append(result["X_ctx"])

            if X_parts:
                result["X"] = np.concatenate(X_parts, axis=1) if len(X_parts) > 1 else X_parts[0]
        elif X_esm is not None:
            # Keep ESM for ABMIL variants that use it
            result["esm"] = X_esm

        result["genes"] = genes
        return result

    def evaluate_linear_folds(
        self,
        X,
        Y,
        split_plan,
        input_dim,
        output_dim,
        n_cv_folds,
        fold_states=None,
    ):
        """Evaluate each saved linear fold model on the shared held-out proteins."""
        test_idx = get_test_indices(split_plan)
        Y_test = Y[test_idx]
        
        test_fold_metrics = []
        test_fold_probs = []
        for cv_fold in range(n_cv_folds):
            tr_idx, _, _ = get_cv_train_val_indices(split_plan, cv_fold)
            X_train = X[tr_idx]
            mean = X_train.mean(axis=0)
            std = np.maximum(X_train.std(axis=0), 1e-8)
            X_test = ((X[test_idx] - mean) / std).astype(np.float32)
            X_test_t = torch.from_numpy(X_test).to(self.device)

            model = LinearProbe(input_dim, output_dim).to(self.device)
            state_dict = None
            if fold_states is not None and cv_fold < len(fold_states):
                state_dict = fold_states[cv_fold]
            if state_dict is not None:
                model.load_state_dict(state_dict)
            else:
                model.load_state_dict(
                    torch.load(
                        self.output_dir / "models" / f"fold_{cv_fold}.pt",
                        map_location=self.device,
                        weights_only=True,
                    )
                )
            model.eval()

            with torch.no_grad():
                logits = model(X_test_t)
                probs = torch.sigmoid(logits).cpu().numpy()
            local_metrics = compute_all_metrics(Y_test, probs)
            local_metrics['fold'] = cv_fold 
            test_fold_metrics.append(local_metrics)
            test_fold_probs.append(probs.astype(np.float32))

        # Aggregate CV validation results on test set
        summary_test = summarize_cv_metrics(test_fold_metrics, split='test')
        serializable_fold_metrics = json_compatible(test_fold_metrics)
        summary_test["test_fold_metrics_json"] = json.dumps(serializable_fold_metrics, sort_keys=True)
        with open(self.output_dir / "test_fold_metrics.json", "w") as f:
            json.dump(serializable_fold_metrics, f, indent=2, sort_keys=True)
        np.savez_compressed(
            self.output_dir / "test_predictions.npz",
            y_true=Y_test.astype(np.float32),
            test_idx=np.asarray(test_idx, dtype=np.int64),
            fold_probs=np.stack(test_fold_probs, axis=0),
        )

        print(
            "  Test (mean over folds): "
            f"AUROC={summary_test['test_auroc_macro_mean']:.4f} +/- {summary_test['test_auroc_macro_std']:.4f}, "
            f"AUPRC={summary_test['test_auprc_macro_mean']:.4f} +/- {summary_test['test_auprc_macro_std']:.4f}, "
            f"F1={summary_test['test_f1_macro_mean']:.4f} +/- {summary_test['test_f1_macro_std']:.4f}"
        )

        return summary_test
    
    
    def train_linear_folds(
        self,
        variant: ModelVariant,
        features: Dict,
        Y: np.ndarray,
        split_plan,
    ) -> Dict:
        """Fit five linear models and report their validation and held-out scores."""
        X = features["X"]
        genes = features["genes"]

        test_idx = get_test_indices(split_plan)
        n_cv_folds = split_plan.n_cv_folds  # Should be 5

        print(f"  Using {n_cv_folds}-fold CV with held-out test set ({len(test_idx)} samples)")

        fold_metrics = []
        histories = []
        selection_key = self.selection_metric_key()
        best_val_score = -np.inf
        best_model_state = None
        best_fold = 0
        fold_states = []

        for cv_fold in range(n_cv_folds):
            print(f"\n--- CV Fold {cv_fold + 1}/{n_cv_folds} ---")

            tr_idx, val_idx, _ = get_cv_train_val_indices(split_plan, cv_fold)

            # Fold-local normalization (train-only) to avoid validation leakage.
            X_train_raw = X[tr_idx]
            mean = X_train_raw.mean(axis=0)
            std = np.maximum(X_train_raw.std(axis=0), 1e-8)

            X_train = ((X[tr_idx] - mean) / std).astype(np.float32)
            X_val = ((X[val_idx] - mean) / std).astype(np.float32)

            Y_train = Y[tr_idx]
            Y_val = Y[val_idx]

            model = LinearProbe(X.shape[1], Y.shape[1]).to(self.device)

            history = TrainingHistory(variant.name, cv_fold)
            model = self.train_linear_fold(
                model, X_train, Y_train, X_val, Y_val,
                history, split_plan.label_clusters, tr_idx
            )
            histories.append(history)

            model.eval()
            with torch.no_grad():
                X_val_t = torch.from_numpy(X_val).to(self.device)
                logits = model(X_val_t)
                probs = torch.sigmoid(logits).cpu().numpy()

            metrics = compute_all_metrics(Y_val, probs)
            metrics["fold"] = cv_fold
            fold_metrics.append(metrics)
            print(f"  Val: AUROC={metrics['auroc_macro']:.4f}, AUPRC={metrics['auprc_macro']:.4f}, F1={metrics['f1_macro']:.4f}")

            # Track best model for final test evaluation
            score = float(metrics.get(selection_key, np.nan))
            if np.isfinite(score) and score > best_val_score:
                best_val_score = score
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_fold = cv_fold

            fold_states.append({k: v.cpu().clone() for k, v in model.state_dict().items()})

            save_model_weights(model, self.output_dir, cv_fold)
            plot_training_curve(
                history,
                self.output_dir / "training_curves" / f"{variant.name.lower().replace(' ', '_')}_fold_{cv_fold}.png",
                variant.name,
            )

        # Save best model as "best" weights
        if best_model_state is not None:
            best_model = LinearProbe(X.shape[1], Y.shape[1])
            best_model.load_state_dict(best_model_state)
            save_model_weights(best_model, self.output_dir, "best")
        selection_label = "F1" if selection_key == "f1_macro" else "AUPRC"
        print(f"  Selected best fold by validation {selection_label}: fold={best_fold + 1}")

        save_histories_csv(histories, self.output_dir / "training_histories.csv")
        plot_training_curves(
            histories,
            self.output_dir / "training_curves" / f"{variant.name.lower().replace(' ', '_')}.png",
            variant.name,
        )

        # Aggregate CV validation results on validation set
        summary = summarize_cv_metrics(fold_metrics, split='val')
        summary["model"] = variant.name
        summary["n_samples"] = len(genes)
        summary["n_classes"] = Y.shape[1]
        summary["n_test_samples"] = len(test_idx)
        
        
        # Final test evaluation - evaluate all fold models on test set and summarize mean/std
        print(f"\n--- Final Test Evaluation (mean over {n_cv_folds} fold metrics) ---")

        summary_test = self.evaluate_linear_folds(
            X,
            Y,
            split_plan,
            X.shape[1],
            Y.shape[1],
            n_cv_folds,
            fold_states=fold_states,
            )
        summary.update(summary_test)
        
        return summary


    def train_linear_fold(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        history: TrainingHistory,
        label_clusters: Optional[np.ndarray],
        train_idx: np.ndarray,
    ) -> nn.Module:
        """Train a single fold of LR model."""
        criterion = self.binary_loss(Y_train)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        if label_clusters is not None:
            train_clusters = label_clusters[train_idx]
            sampler = MultilabelStratifiedSampler(
                labels=None,
                batch_size=self.config.batch_size,
                shuffle=True,
                seed=self.config.seed,
                cluster_labels=train_clusters,
            )
        else:
            sampler = None

        n_train = len(X_train)
        X_train_t = torch.from_numpy(X_train).to(self.device)
        Y_train_t = torch.from_numpy(Y_train).to(self.device)
        X_val_t = torch.from_numpy(X_val).to(self.device)
        Y_val_t = torch.from_numpy(Y_val).to(self.device)

        selection_key = self.selection_metric_key()
        best_score = -np.inf
        best_state = None
        patience_counter = 0

        epoch_rng = np.random.default_rng(self.config.seed)
        for epoch in range(self.config.epochs):
            # Reshuffle training indices each epoch
            if sampler is not None:
                train_indices = list(sampler)
            else:
                train_indices = list(range(n_train))
                epoch_rng.shuffle(train_indices)

            # Training
            model.train()
            total_loss = 0.0
            n_batches = 0

            for i in range(0, len(train_indices), self.config.batch_size):
                batch_idx = train_indices[i:i + self.config.batch_size]
                X_batch = X_train_t[batch_idx]
                Y_batch = Y_train_t[batch_idx]

                optimizer.zero_grad()

                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = model(X_batch)
                    loss = criterion(logits, Y_batch)

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                total_loss += loss.item()
                n_batches += 1

            avg_train_loss = total_loss / max(n_batches, 1)

            # Validation
            model.eval()
            with torch.no_grad():
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    val_logits = model(X_val_t)
                    val_loss = criterion(val_logits, Y_val_t).item()
                    val_probs = torch.sigmoid(val_logits).cpu().numpy()

            val_metrics = compute_all_metrics(Y_val, val_probs)
            val_auprc = val_metrics["auprc_macro"]
            val_auroc = val_metrics["auroc_macro"]
            val_score = float(val_metrics.get(selection_key, np.nan))

            history.add_epoch(epoch + 1, avg_train_loss, val_loss, val_auprc, val_auroc)

            # Early stopping
            if np.isfinite(val_score) and val_score > best_score:
                best_score = val_score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 50 == 0:
                print(
                    f"  Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, "
                    f"val_auprc={val_auprc:.4f}, val_f1={val_metrics['f1_macro']:.4f}"
                )

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)

        return model


    def evaluate_attention_folds(
        self,
        ctx_bags,
        esm,
        Y,
        split_plan,
        n_cv_folds:int,
        model_params:dict,
        fold_states=None,
    ):
        """Evaluate each saved attention fold model on the shared held-out proteins."""
        test_fold_metrics = []
        test_fold_probs = []
        test_idx = get_test_indices(split_plan)
        Y_test = Y[test_idx]
        
        models_path = self.output_dir / "models"

        for cv_fold in range(n_cv_folds):
            tr_idx, _, _ = get_cv_train_val_indices(split_plan, cv_fold)
            train_bags = [ctx_bags[i] for i in tr_idx]
            _, bag_mean, bag_std = zscore_normalize_bags(train_bags)
            test_bags = [(ctx_bags[i] - bag_mean) / bag_std for i in test_idx]

            if esm is not None:
                esm_train = esm[tr_idx]
                esm_mean = esm_train.mean(axis=0)
                esm_std = np.maximum(esm_train.std(axis=0), 1e-8)
                test_esm = ((esm[test_idx] - esm_mean) / esm_std).astype(np.float32)
            else:
                test_esm = None

            test_dataset = ABMILDataset(test_bags, test_esm, Y_test)
            test_loader = DataLoader(test_dataset, shuffle=False, **self.abmil_loader_options())

            model = self.build_abmil_model(model_params)
            state_dict = None
            if fold_states is not None and cv_fold < len(fold_states):
                state_dict = fold_states[cv_fold]
            if state_dict is not None:
                model.load_state_dict(state_dict)
            else:
                model.load_state_dict(
                    torch.load(
                        models_path / f"fold_{cv_fold}.pt",
                        map_location=self.device,
                        weights_only=True,
                    )
                )
            model.eval()

            fold_probs = []
            with torch.no_grad():
                for batch_x, _ in test_loader:
                    batch_x = self.move_batch_to_device(batch_x)
                    with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                        logits = model(batch_x)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    fold_probs.append(probs)
            fold_probs = np.concatenate(fold_probs, axis=0)

            local_metrics = compute_all_metrics(Y_test, fold_probs)
            local_metrics['fold'] = cv_fold
            test_fold_metrics.append(local_metrics)
            test_fold_probs.append(fold_probs.astype(np.float32))

        # Aggregate CV validation results on test set
        summary_test = summarize_cv_metrics(test_fold_metrics, split='test')
        serializable_fold_metrics = json_compatible(test_fold_metrics)
        summary_test["test_fold_metrics_json"] = json.dumps(serializable_fold_metrics, sort_keys=True)
        with open(self.output_dir / "test_fold_metrics.json", "w") as f:
            json.dump(serializable_fold_metrics, f, indent=2, sort_keys=True)
        np.savez_compressed(
            self.output_dir / "test_predictions.npz",
            y_true=Y_test.astype(np.float32),
            test_idx=np.asarray(test_idx, dtype=np.int64),
            fold_probs=np.stack(test_fold_probs, axis=0),
        )

        print(
            "  Test (mean over folds): "
            f"AUROC={summary_test['test_auroc_macro_mean']:.4f} +/- {summary_test['test_auroc_macro_std']:.4f}, "
            f"AUPRC={summary_test['test_auprc_macro_mean']:.4f} +/- {summary_test['test_auprc_macro_std']:.4f}, "
            f"F1={summary_test['test_f1_macro_mean']:.4f} +/- {summary_test['test_f1_macro_std']:.4f}"
        )

        return summary_test


    def train_attention_folds(
        self,
        variant: ModelVariant,
        features: Dict,
        Y: np.ndarray,
        split_plan,
    ) -> Dict:
        """Fit five attention models and report their validation and held-out scores."""
        # Get hyperparams from config (can be overridden via CLI --att-dim and --dropout)
        dropout = self.config.dropout
        att_hidden_dim = self.config.att_hidden_dim
        num_heads = int(self.config.num_heads)
        mlp_hidden_dim = self.abmil_hidden_dim(variant)

        ctx_bags = features["ctx_bags"]
        esm = features.get("esm")
        genes = features["genes"]

        ctx_dim = ctx_bags[0].shape[1] if ctx_bags else 0
        esm_dim = esm.shape[1] if esm is not None else 0

        test_idx = get_test_indices(split_plan)
        n_cv_folds = split_plan.n_cv_folds

        print(f"  Using {n_cv_folds}-fold CV with held-out test set ({len(test_idx)} samples)")
        pdl_info = ""
        pdl_proj_mode = self.pdl_projection_mode(variant)
        pdl_proj_layers = self.pdl_projection_layers(variant)
        if variant.use_pdl and self.config.pdl_pmax > 0:
            pdl_info = f", pdl_pmax={self.config.pdl_pmax}, pdl_proj={pdl_proj_mode}"
            if pdl_proj_layers > 0:
                pdl_info += f", pdl_layers={pdl_proj_layers}"
        att_dropout = self.config.att_dropout
        attdo_info = f", att_dropout={att_dropout}" if att_dropout > 0 else ""
        print(f"  Hyperparams: dropout={dropout}, att_hidden={att_hidden_dim}{attdo_info}{pdl_info}")

        fold_metrics = []
        histories = []
        selection_key = self.selection_metric_key()
        best_val_score = -np.inf
        best_model_state = None
        best_fold = 0
        fold_states = []

        model_params = {
            'num_classes': Y.shape[1],
            'ctx_dim': ctx_dim,
            'esm_dim': esm_dim,
            'use_ext_embed': "ext_embed" in variant.embedding_sources,
            'num_heads': num_heads,
            'att_hidden': att_hidden_dim,
            'dropout': dropout,
            'att_dropout': att_dropout,
            'attention_type': variant.attention_type or "gated",
            'classifier_type': variant.classifier_type,
            'mlp_hidden_dim': mlp_hidden_dim,
            'use_pdl': variant.use_pdl,
            'pdl_proj_mode': pdl_proj_mode,
            'pdl_proj_layers': pdl_proj_layers,
        }

        for cv_fold in range(n_cv_folds):
            print(f"\n--- CV Fold {cv_fold + 1}/{n_cv_folds} ---")

            tr_idx, val_idx, _ = get_cv_train_val_indices(split_plan, cv_fold)

            # Fold-local normalization (train-only) to avoid validation leakage.
            train_bags_raw = [ctx_bags[i] for i in tr_idx]
            train_bags, bag_mean, bag_std = zscore_normalize_bags(train_bags_raw)
            val_bags = [(ctx_bags[i] - bag_mean) / bag_std for i in val_idx]

            # Apply fold-local ESM normalization if present
            if esm is not None:
                esm_train = esm[tr_idx]
                esm_mean = esm_train.mean(axis=0)
                esm_std = np.maximum(esm_train.std(axis=0), 1e-8)
                train_esm = ((esm[tr_idx] - esm_mean) / esm_std).astype(np.float32)
                val_esm = ((esm[val_idx] - esm_mean) / esm_std).astype(np.float32)
            else:
                train_esm = val_esm = None

            Y_train = Y[tr_idx]
            Y_val = Y[val_idx]

            model = self.build_abmil_model(model_params)

            history = TrainingHistory(variant.name, cv_fold)
            model = self.train_attention_fold(
                variant,
                model, train_bags, train_esm, Y_train,
                val_bags, val_esm, Y_val,
                history, split_plan.label_clusters, tr_idx,
            )
            histories.append(history)

            model.eval()
            val_dataset = ABMILDataset(val_bags, val_esm, Y_val)
            val_loader = DataLoader(val_dataset, shuffle=False, **self.abmil_loader_options())

            all_probs = []
            with torch.no_grad():
                for batch_x, _ in val_loader:
                    batch_x = self.move_batch_to_device(batch_x)
                    with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                        logits = model(batch_x)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_probs.append(probs)

            all_probs = np.concatenate(all_probs, axis=0)
            metrics = compute_all_metrics(Y_val, all_probs)
            metrics["fold"] = cv_fold
            fold_metrics.append(metrics)
            print(f"  Val: AUROC={metrics['auroc_macro']:.4f}, AUPRC={metrics['auprc_macro']:.4f}, F1={metrics['f1_macro']:.4f}")

            # Track best model for final test evaluation
            score = float(metrics.get(selection_key, np.nan))
            if np.isfinite(score) and score > best_val_score:
                best_val_score = score
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_fold = cv_fold

            fold_states.append({k: v.cpu().clone() for k, v in model.state_dict().items()})

            save_model_weights(model, self.output_dir, cv_fold)
            plot_training_curve(
                history,
                self.output_dir / "training_curves" / f"{variant.name.lower().replace(' ', '_')}_fold_{cv_fold}.png",
                variant.name,
            )

        
        save_histories_csv(histories, self.output_dir / "training_histories.csv")
        plot_training_curves(
            histories,
            self.output_dir / "training_curves" / f"{variant.name.lower().replace(' ', '_')}.png",
            variant.name,
        )

        # Save best model as "best" weights
        if best_model_state is not None:
            best_model = model
            best_model.load_state_dict(best_model_state)
            save_model_weights(best_model, self.output_dir, "best")
        selection_label = "F1" if selection_key == "f1_macro" else "AUPRC"
        print(f"  Selected best fold by validation {selection_label}: fold={best_fold + 1}")


        # Aggregate CV validation results on validation set
        summary = summarize_cv_metrics(fold_metrics, split='val')
        summary["model"] = variant.name
        summary["n_samples"] = len(genes)
        summary["n_classes"] = Y.shape[1]
        summary["n_test_samples"] = len(test_idx)
        # Add validated hyperparameters used
        summary["dropout"] = dropout
        summary["att_dropout"] = att_dropout
        summary["att_hidden_dim"] = att_hidden_dim
        summary["num_heads"] = num_heads
        summary["mil_mlp_dim"] = mlp_hidden_dim if mlp_hidden_dim is not None else float("nan")
        summary["pdl_pmax"] = self.config.pdl_pmax if variant.use_pdl else 0.0
        summary["pdl_proj_mode"] = pdl_proj_mode if variant.use_pdl else ""
        summary["pdl_proj_layers"] = pdl_proj_layers if variant.use_pdl else 0
        summary["abmil_arch"] = "late_fusion" if "ext_embed" in variant.embedding_sources else "context_only"
        summary["lr"] = self.config.lr
        summary["weight_decay"] = self.config.weight_decay

        # Final test evaluation - evaluate all fold models on test set and summarize mean/std
        print(f"\n--- Final Test Evaluation (mean over {n_cv_folds} fold metrics) ---")
        summary_test = self.evaluate_attention_folds(
            ctx_bags,
            esm,
            Y,
            split_plan,
            n_cv_folds,
            model_params,
            fold_states=fold_states,
            )
        summary.update(summary_test)
        
        return summary

    def train_attention_fold(
        self,
        variant: ModelVariant,
        model: nn.Module,
        train_bags: List[np.ndarray],
        train_esm: Optional[np.ndarray],
        Y_train: np.ndarray,
        val_bags: List[np.ndarray],
        val_esm: Optional[np.ndarray],
        Y_val: np.ndarray,
        history: TrainingHistory,
        label_clusters: Optional[np.ndarray],
        train_idx: np.ndarray,
    ) -> nn.Module:
        """Train a single fold of ABMIL model."""
        criterion = self.binary_loss(Y_train)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        use_pdl = variant.use_pdl and self.config.pdl_pmax > 0
        pdl_scheduler = None
        if use_pdl:
            pdl_scheduler = LinearScheduler(
                model,
                start_value=0.0,
                stop_value=float(self.config.pdl_pmax),
                # Use a fixed PDL ramp horizon rather than the global epoch budget.
                nr_steps=40,
            )

        train_dataset = ABMILDataset(train_bags, train_esm, Y_train)
        val_dataset = ABMILDataset(val_bags, val_esm, Y_val)

        # Use stratified sampler if available
        if label_clusters is not None:
            train_clusters = label_clusters[train_idx]
            sampler = MultilabelStratifiedSampler(
                labels=None,
                batch_size=self.config.batch_size,
                shuffle=True,
                seed=self.config.seed,
                cluster_labels=train_clusters,
            )
            train_loader = DataLoader(train_dataset, sampler=sampler, **self.abmil_loader_options())
        else:
            train_loader = DataLoader(train_dataset, shuffle=True, **self.abmil_loader_options())

        val_loader = DataLoader(val_dataset, shuffle=False, **self.abmil_loader_options())

        selection_key = self.selection_metric_key()
        best_score = -np.inf
        best_state = None
        patience_counter = 0

        for epoch in tqdm(range(self.config.epochs), desc='epochs'):
            if pdl_scheduler is not None:
                pdl_scheduler.step()
            # Training
            model.train()
            total_loss = 0.0
            n_batches = 0

            for batch_x, batch_y in train_loader:
                batch_x = self.move_batch_to_device(batch_x)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    loss = criterion(model(batch_x), batch_y)

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                total_loss += loss.item()
                n_batches += 1

            avg_train_loss = total_loss / max(n_batches, 1)

            # Validation
            model.eval()
            val_loss = 0.0
            val_probs = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = self.move_batch_to_device(batch_x)
                    batch_y = batch_y.to(self.device)

                    with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                        logits = model(batch_x)
                        loss = criterion(logits, batch_y)

                    val_loss += loss.item()
                    val_probs.append(torch.sigmoid(logits).cpu().numpy())

            val_loss /= max(len(val_loader), 1)
            val_probs = np.concatenate(val_probs, axis=0)

            val_metrics = compute_all_metrics(Y_val, val_probs)
            val_auprc = val_metrics["auprc_macro"]
            val_auroc = val_metrics["auroc_macro"]
            val_score = float(val_metrics.get(selection_key, np.nan))

            history.add_epoch(epoch + 1, avg_train_loss, val_loss, val_auprc, val_auroc)

            # Early stopping
            if np.isfinite(val_score) and val_score > best_score:
                best_score = val_score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 50 == 0:
                print(
                    f"  Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, "
                    f"val_auprc={val_auprc:.4f}, val_f1={val_metrics['f1_macro']:.4f}"
                )

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)

        return model

    def move_batch_to_device(self, batch_x: Dict) -> Dict:
        """Move batch to device."""
        out = {
            "ctx": batch_x["ctx"].to(self.device),
            "ctx_batch": batch_x["ctx_batch"].to(self.device),
            "ctx_ptr": batch_x["ctx_ptr"].to(self.device),
            "esm": (
                batch_x["esm"].to(self.device)
                if batch_x["esm"] is not None
                else None
            ),
        }
        return out

    def evaluate_saved_models(
        self,
        variant: ModelVariant,
        features: Dict,
        Y: np.ndarray,
        split_plan,
    ) -> Dict:
        """Run inference only using saved weights."""
        print(f"\n[INFERENCE] Running inference for {variant.name}")

        if variant.model_type == ModelType.LR:
            return self.evaluate_saved_linear_models(variant, features, Y, split_plan)
        else:
            return self.evaluate_saved_attention_models(variant, features, Y, split_plan)

    def evaluate_saved_linear_models(
        self,
        variant: ModelVariant,
        features: Dict,
        Y: np.ndarray,
        split_plan,
    ) -> Dict:
        """Run inference for LR model using saved weights."""
        X = features["X"]
        genes = features["genes"]

        test_idx = get_test_indices(split_plan)
        n_cv_folds = split_plan.n_cv_folds

        fold_metrics = []

        for cv_fold in range(n_cv_folds):
            print(f"\n--- CV Fold {cv_fold + 1}/{n_cv_folds} (inference) ---")

            tr_idx, val_idx, _ = get_cv_train_val_indices(split_plan, cv_fold)

            X_train_raw = X[tr_idx]
            mean = X_train_raw.mean(axis=0)
            std = np.maximum(X_train_raw.std(axis=0), 1e-8)

            X_val = ((X[val_idx] - mean) / std).astype(np.float32)
            Y_val = Y[val_idx]

            model = LinearProbe(X.shape[1], Y.shape[1]).to(self.device)
            loaded = load_model_weights(model, self.output_dir, cv_fold)
            if not loaded:
                raise FileNotFoundError(
                    f"Missing weights for fold {cv_fold}: {self.output_dir / 'models' / f'fold_{cv_fold}.pt'}"
                )
            model.eval()

            # Run inference
            with torch.no_grad():
                X_val_t = torch.from_numpy(X_val).to(self.device)
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = model(X_val_t)
                probs = torch.sigmoid(logits).cpu().numpy()

            metrics = compute_all_metrics(Y_val, probs)
            metrics["fold"] = cv_fold
            fold_metrics.append(metrics)
            print(f"  Val: AUROC={metrics['auroc_macro']:.4f}, AUPRC={metrics['auprc_macro']:.4f}, F1={metrics['f1_macro']:.4f}")

        # Aggregate results on validation sets
        
        summary = summarize_cv_metrics(fold_metrics, split='val')
        summary["model"] = variant.name
        summary["n_samples"] = len(genes)
        summary["n_classes"] = Y.shape[1]
        summary["n_test_samples"] = len(test_idx)
        
        # Test evaluation - evaluate all fold models and summarize mean/std
        print(f"\n--- Test Evaluation (mean over {n_cv_folds} fold metrics) ---")

        summary_test = self.evaluate_linear_folds(
            X,
            Y,
            split_plan,
            X.shape[1],
            Y.shape[1],
            n_cv_folds,
            )
        summary.update(summary_test)
        
        return summary

    def evaluate_saved_attention_models(
        self,
        variant: ModelVariant,
        features: Dict,
        Y: np.ndarray,
        split_plan,
    ) -> Dict:
        """Run inference for ABMIL model using saved weights."""
        # Hyperparams used to rebuild the model architecture during inference
        dropout = self.config.dropout
        att_hidden_dim = self.config.att_hidden_dim
        num_heads = int(self.config.num_heads)
        mlp_hidden_dim = self.abmil_hidden_dim(variant)

        ctx_bags = features["ctx_bags"]
        esm = features.get("esm")
        genes = features["genes"]

        ctx_dim = ctx_bags[0].shape[1] if ctx_bags else 0
        esm_dim = esm.shape[1] if esm is not None else 0

        test_idx = get_test_indices(split_plan)
        n_cv_folds = split_plan.n_cv_folds

        dropout = self.config.dropout
        att_hidden_dim = self.config.att_hidden_dim
        att_dropout = self.config.att_dropout
        pdl_proj_mode = self.pdl_projection_mode(variant)
        pdl_proj_layers = self.pdl_projection_layers(variant)

        model_params = {
            'num_classes': Y.shape[1],
            'ctx_dim': ctx_dim,
            'esm_dim': esm_dim,
            'use_ext_embed': "ext_embed" in variant.embedding_sources,
            'num_heads': num_heads,
            'att_hidden': att_hidden_dim,
            'dropout': dropout,
            'att_dropout': att_dropout,
            'attention_type': variant.attention_type or "gated",
            'classifier_type': variant.classifier_type,
            'mlp_hidden_dim': mlp_hidden_dim,
            'use_pdl': variant.use_pdl,
            'pdl_proj_mode': pdl_proj_mode,
            'pdl_proj_layers': pdl_proj_layers,
        }

        fold_metrics = []

        for cv_fold in range(n_cv_folds):
            print(f"\n--- CV Fold {cv_fold + 1}/{n_cv_folds} (inference) ---")

            tr_idx, val_idx, _ = get_cv_train_val_indices(split_plan, cv_fold)

            train_bags = [ctx_bags[i] for i in tr_idx]
            _, bag_mean, bag_std = zscore_normalize_bags(train_bags)

            val_bags = [(ctx_bags[i] - bag_mean) / bag_std for i in val_idx]
            if esm is not None:
                esm_train = esm[tr_idx]
                esm_mean = esm_train.mean(axis=0)
                esm_std = np.maximum(esm_train.std(axis=0), 1e-8)
                val_esm = ((esm[val_idx] - esm_mean) / esm_std).astype(np.float32)
            else:
                val_esm = None
            Y_val = Y[val_idx]

            model = self.build_abmil_model(model_params)
            loaded = load_model_weights(model, self.output_dir, cv_fold)
            if not loaded:
                raise FileNotFoundError(
                    f"Missing weights for fold {cv_fold}: {self.output_dir / 'models' / f'fold_{cv_fold}.pt'}"
                )
            model.eval()

            # Run inference
            val_dataset = ABMILDataset(val_bags, val_esm, Y_val)
            val_loader = DataLoader(val_dataset, shuffle=False, **self.abmil_loader_options())

            all_probs = []
            with torch.no_grad():
                for batch_x, _ in val_loader:
                    batch_x = self.move_batch_to_device(batch_x)
                    with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                        logits = model(batch_x)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_probs.append(probs)

            all_probs = np.concatenate(all_probs, axis=0)
            metrics = compute_all_metrics(Y_val, all_probs)
            metrics["fold"] = cv_fold
            fold_metrics.append(metrics)
            print(f"  Val: AUROC={metrics['auroc_macro']:.4f}, AUPRC={metrics['auprc_macro']:.4f}, F1={metrics['f1_macro']:.4f}")

        # compute validation metrics
        summary = summarize_cv_metrics(fold_metrics, split='val')
        summary["model"] = variant.name
        summary["n_samples"] = len(genes)
        summary["n_classes"] = Y.shape[1]
        summary["n_test_samples"] = len(test_idx)
        # Add validated hyperparameters used
        summary["dropout"] = dropout
        summary["att_dropout"] = att_dropout
        summary["att_hidden_dim"] = att_hidden_dim
        summary["num_heads"] = num_heads
        summary["mil_mlp_dim"] = mlp_hidden_dim if mlp_hidden_dim is not None else float("nan")
        summary["pdl_pmax"] = self.config.pdl_pmax if variant.use_pdl else 0.0
        summary["pdl_proj_mode"] = pdl_proj_mode if variant.use_pdl else ""
        summary["pdl_proj_layers"] = pdl_proj_layers if variant.use_pdl else 0
        summary["abmil_arch"] = "late_fusion" if "ext_embed" in variant.embedding_sources else "context_only"
        summary["lr"] = self.config.lr
        summary["weight_decay"] = self.config.weight_decay

        # Test evaluation - evaluate all fold models and summarize mean/std
        print(f"\n--- Test Evaluation (mean over {n_cv_folds} fold metrics) ---")

        summary_test = self.evaluate_attention_folds(
            ctx_bags,
            esm,
            Y,
            split_plan,
            n_cv_folds,
            model_params,
            )
        summary.update(summary_test)
        
        return summary
