"""Verify and plot saved topology correction-head results; never fit a model."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score

from pretraining.fit_topology_decoder import CorrectionHead, VARIANTS, file_hash
from pretraining.investigate_global_topology import BANKS, K_VALUES, pair_fold

LABELS = {
    "mlp_global": "CF + global MLP",
    "mlp_context": "CF + global/local MLP",
    "mlp_context_distilled": "CF + global/local MLP (distilled)",
}
COLORS = {"CF global": "#d7191c", "CF Cell-PPI": "#7b3294",
          "ProtScape Cell-PPI": "#ff6600", "CF + global MLP": "#2171b5",
          "CF + global/local MLP": "#008837",
          "CF + global/local MLP (distilled)": "#333333"}


def paired_differences(context):
    wide = context.query("k == 500").pivot(index=["cell_id", "bank"], columns="variant", values="auprc")
    comparisons = [(variant, "linear_context") for variant in LABELS]
    comparisons += [("mlp_context", "mlp_global"), ("mlp_context_distilled", "mlp_context")]
    rows = []
    for variant, reference in comparisons:
        for bank, values in ((wide[variant] - wide[reference]) * 100).groupby("bank"):
            rows.append({"variant": variant, "reference": reference, "bank": bank,
                         "mean_delta_pp": values.mean(), "median_delta_pp": values.median(),
                         "min_delta_pp": values.min(), "max_delta_pp": values.max(),
                         "contexts_improved": int((values > 1e-8).sum()), "contexts": len(values)})
    return pd.DataFrame(rows)


def verify(output):
    protocol = json.loads((output / "protocol.json").read_text())
    source = Path(protocol["source_dir"])
    for cell, digest in protocol["source_artifact_hashes"].items():
        assert file_hash(source / f"cell_{cell}.npz") == digest
    dataset = torch.load(output / "dataset.pt", weights_only=True, mmap=True)
    ap_error, f1_error, prediction_error = 0., 0., 0.
    for variant in VARIANTS:
        for fold in range(3):
            path = output / variant / f"fold_{fold}"
            mask = dataset["folds"].numpy() == fold
            expected_keys = dataset["keys"][:, mask].numpy()
            cells = dataset["cells"][mask].numpy()
            values = dataset["features"][:, mask]
            with np.load(path / "predictions.npz") as artifact:
                assert np.array_equal(artifact["keys"], expected_keys)
                assert np.array_equal(artifact["cells"], cells)
                assert np.all(pair_fold(artifact["keys"]) == fold)
                scores = artifact["scores"]
            assert np.isfinite(scores).all()
            if variant == "linear_context":
                assert np.array_equal(scores, values[..., 0].numpy())
            saved = torch.load(path / "head.pt", weights_only=True)
            model = CorrectionHead(variant, np.zeros(11), np.ones(11))
            model.load_state_dict(saved["state_dict"])
            # Independent CPU check of saved weights against GPU predictions.
            indices = np.linspace(0, scores.size - 1, 256, dtype=int)
            with torch.no_grad():
                predicted = model(values.reshape(-1, 11)[indices]).numpy()
            prediction_error = max(prediction_error, float(np.abs(predicted - scores.ravel()[indices]).max()))
            recorded = pd.read_csv(path / "metrics.csv").query("k == 500").set_index(["bank", "cell_id"])
            for bank_id, bank in enumerate(BANKS):
                for cell in np.unique(cells):
                    current = scores[bank_id, cells == cell]
                    labels = np.broadcast_to(np.arange(501) == 0, current.shape).ravel()
                    expected = recorded.loc[bank, cell]
                    ap_error = max(ap_error, abs(average_precision_score(labels, current.ravel()) - expected.auprc))
                    f1_error = max(f1_error, abs(f1_score(labels, current.ravel() >= 0, average="macro") - expected.macro_f1))
    assert ap_error < 1e-12 and f1_error < 1e-12 and prediction_error < 1e-4
    # Fold/context means must also agree with the original frozen CF baseline.
    original = pd.read_csv(source / "metrics_by_context.csv").query("model == 'CF global'")
    current = pd.read_csv(output / "metrics_by_context.csv").query("variant == 'linear_context'")
    paired = original.merge(current, on=["cell_id", "bank", "k"], validate="one_to_one")
    baseline_error = float((paired.auprc_x - paired.auprc_y).abs().max())
    assert len(paired) == 16 * 3 * 5 and baseline_error < 1e-12
    checks = {"source_artifact_hashes_verified": len(protocol["source_artifact_hashes"]),
              "head_folds_verified": 12, "all_saved_pair_ids_and_folds_match": True,
              "zero_update_linear_scores_exactly_cf": True,
              "max_ap500_error": float(ap_error), "max_macro_f1_500_error": float(f1_error),
              "max_sampled_cpu_head_prediction_error": prediction_error,
              "max_original_cf_context_ap_error": baseline_error,
              "status": "exploratory validation only; not a new test result"}
    (output / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


def save_figure(fig, output, stem):
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_curves(output, source):
    heads = pd.read_csv(output / "summary.csv")
    for metric, filename, ylabel in (("auprc", "summary.csv", "Validation AUPRC (%)"),
                                      ("macro_f1", "f1_summary.csv", "Validation binary-class macro-F1 (%)")):
        frozen = pd.read_csv(source / filename)
        frozen = frozen[frozen.model.isin(list(COLORS)[:3])][["model", "bank", "k", "mean"]]
        added = heads[heads.variant.isin(LABELS)].rename(columns={"variant": "model", metric: "mean"})
        added["model"] = added.model.map(LABELS)
        data = pd.concat([frozen, added[["model", "bank", "k", "mean"]]])
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
        for ax, bank in zip(axes, BANKS):
            for name, color in COLORS.items():
                subset = data[(data.bank == bank) & (data.model == name)].sort_values("k")
                ax.plot(subset.k, subset["mean"] * 100, marker="o", color=color, label=name,
                        markersize=4, linestyle="--" if name in LABELS.values() else "-")
            ax.set_xscale("log")
            ax.set_xticks(K_VALUES, [str(k) for k in K_VALUES])
            ax.set_xlabel("Negatives per positive")
            ax.set_title(bank.replace("_", " ").title())
            ax.set_ylim(0, 100)
        axes[0].set_ylabel(ylabel)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
                   bbox_to_anchor=(.5, -.10), frameon=False)
        save_figure(fig, output, f"single_backbone_{metric}")


def plot_learning_curves(output):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3), sharey=True)
    labels = {"linear_context": "Linear global/local", **LABELS}
    for ax, variant in zip(axes, VARIANTS):
        for fold in range(3):
            history = pd.read_csv(output / variant / f"fold_{fold}" / "inner_history.csv")
            delta = (history.selection_ap - history.selection_ap.iloc[0]) * 100
            line, = ax.plot(history["update"], delta, label=f"Inner split {fold}")
            selected = history.selection_ap.idxmax()
            ax.plot(history.loc[selected, "update"], delta.loc[selected], "o", color=line.get_color(), ms=4)
        ax.axhline(0, color="black", linewidth=.7)
        ax.axvline(100, color="grey", linestyle=":", linewidth=1)
        ax.set_title(labels[variant].replace("CF + ", ""), fontsize=9)
        ax.set_xlabel("Inner fitting updates")
    axes[0].set_ylabel("Inner selector change vs CF (pp)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               bbox_to_anchor=(.5, -.12), frameon=False)
    save_figure(fig, output, "inner_learning_curves")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    verify(args.output_dir)
    context = pd.read_csv(args.output_dir / "metrics_by_context.csv")
    paired_differences(context).to_csv(args.output_dir / "paired_ap500_differences.csv", index=False)
    source = Path(json.loads((args.output_dir / "protocol.json").read_text())["source_dir"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    plot_curves(args.output_dir, source)
    plot_learning_curves(args.output_dir)


if __name__ == "__main__":
    main()
