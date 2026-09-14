"""Measure same-pair context variation in saved, paired Cell-PPI predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = {
    "global_context_free": "CF / global graph",
    "cell_context_free": "CF / Cell-PPI",
    "protscape": "ProtScape / Cell-PPI",
}


def analyze(scores, min_contexts=5):
    if min_contexts < 2:
        raise ValueError("At least two contexts are required for variation")
    if scores.duplicated(["pair_key", "cell_id"]).any():
        raise ValueError("Duplicate pair/context observations")
    columns = [f"score_{key}" for key in MODELS]
    values = scores[columns].to_numpy()
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Expected finite probability scores in [0, 1]")
    sizes = scores.groupby("pair_key").size()
    eligible = sizes.index[sizes >= min_contexts]
    scores = scores[scores.pair_key.isin(eligible)].copy()
    if scores.empty:
        raise ValueError("No pairs meet the context coverage requirement")
    grouped = scores.groupby("pair_key", sort=True)
    pairs = grouped[["source_protein", "target_protein"]].first()
    pairs["n_contexts"] = grouped.size()
    cells = scores.groupby(["cell_id", "cell"], sort=True).size().rename("n_pairs").to_frame()
    summary = []
    for key, name in MODELS.items():
        column = f"score_{key}"
        sd = grouped[column].std(ddof=0)
        pairs[f"sd_{key}_pp"] = 100 * sd
        centered = scores[column] - grouped[column].transform("mean")
        scores[f"squared_deviation_{key}"] = centered ** 2
        cells[f"rms_context_deviation_{key}_pp"] = 100 * np.sqrt(
            scores.groupby(["cell_id", "cell"])[f"squared_deviation_{key}"].mean()
        )
        summary.append({"model": name, "model_key": key, "n_pairs": len(pairs),
                        "mean_pair_sd_pp": float(100 * sd.mean()),
                        "median_pair_sd_pp": float(100 * sd.median())})

    # Within-pair Spearman: compare context rankings, not protein-pair identity.
    # This is invariant to monotonic recalibration of either model's scores.
    rank_columns = ["score_cell_context_free", "score_protscape"]
    ranks = grouped[rank_columns].rank(method="average")
    rank_means = ranks.groupby(scores.pair_key).transform("mean")
    centered_ranks = ranks - rank_means
    x, y = (centered_ranks[column] for column in rank_columns)
    numerator = (x * y).groupby(scores.pair_key).sum()
    denominator = np.sqrt((x * x).groupby(scores.pair_key).sum() * (y * y).groupby(scores.pair_key).sum())
    rho = numerator / denominator.replace(0, np.nan)
    pairs["context_rank_spearman"] = rho
    defined = rho.dropna()
    agreement = {"median_within_pair_spearman": float(defined.median()) if len(defined) else float("nan"),
                 "mean_within_pair_spearman": float(defined.mean()) if len(defined) else float("nan"),
                 "n_defined_pairs": int(rho.notna().sum()),
                 "n_constant_score_pairs": int(rho.isna().sum())}
    counts = {"min_contexts": min_contexts, "n_pairs": len(pairs),
              "n_pair_context_observations": len(scores), "n_contexts": len(cells)}
    return pd.DataFrame(summary), pairs.reset_index(), cells.reset_index(), agreement, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    source = args.input_dir / "test_positive_pair_context_scores.csv.gz"
    protocol = json.loads((args.input_dir / "summary.json").read_text())
    selected = json.loads(args.selection.read_text())
    if protocol["global_checkpoint_sha256"] != selected["checkpoint_sha256"]:
        raise ValueError("Saved scores do not use the final selected CF checkpoint")
    if protocol["split_policy"]["test"] != "test targets scored on train+validation topology":
        raise ValueError("Unexpected message graph / test split policy")
    scores = pd.read_csv(source)
    if scores.cell_id.nunique() != protocol["n_cells"]:
        raise ValueError("Context coverage differs from the evaluation manifest")
    summary, pairs, cells, agreement, counts = analyze(scores)
    if counts["n_contexts"] != protocol["n_cells"]:
        raise ValueError("Coverage filter removed an entire context")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pairs.to_csv(args.output_dir / "per_pair.csv", index=False)
    cells.to_csv(args.output_dir / "per_cell.csv", index=False)
    metadata = {**counts, "agreement": agreement, "source": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "global_checkpoint_sha256": protocol["global_checkpoint_sha256"],
                "protscape_checkpoint_sha256": protocol["protscape_checkpoint_sha256"],
                "population": "held-out known interaction pairs shared across >=5 Cell-PPIs",
                "aggregation": "each protein pair receives equal weight; SD uses ddof=0",
                "per_cell_metric": "RMS deviation from each pair's own across-context mean, in probability percentage points"}
    (args.output_dir / "protocol.json").write_text(json.dumps(metadata, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9,
                         "axes.linewidth": 0.6, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(4.5, 4.1))
    x = cells.rms_context_deviation_cell_context_free_pp
    y = cells.rms_context_deviation_protscape_pp
    limit = 1.06 * max(x.max(), y.max())
    ax.plot([0, limit], [0, limit], "--", color="0.6", lw=1)
    ax.scatter(x, y, s=17, color="#e6550d", alpha=0.6, edgecolors="none")
    ax.set(xlabel="CF / Cell-PPI context deviation (pp)",
           ylabel="ProtScape context deviation (pp)",
           title=f"Same-pair context dependence: {len(cells)} Cell-PPIs",
           xlim=(0, limit), ylim=(0, limit))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "per_cell_variation.png", dpi=180)
    fig.savefig(args.output_dir / "per_cell_variation.pdf")
    plt.close(fig)
    print(summary.to_string(index=False))
    print(json.dumps({**counts, **agreement}, indent=2))


if __name__ == "__main__":
    main()
