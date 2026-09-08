"""Generate Parkinson candidates and support from saved ensemble predictions."""

from exploration.analysis.parkinson_inference import load_predictions
from exploration.analysis.parkinson_target_analysis import (
    ANALYSIS_DIR,
    build_candidates,
    build_external_support,
)


def main():
    scores, _ = load_predictions()
    candidates = build_candidates(scores)
    support, summary = build_external_support(candidates)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, table in {
        "parkinson_candidate_set.csv": candidates,
        "parkinson_candidate_external_support_rows.csv": support,
        "parkinson_candidate_external_support.csv": summary,
    }.items():
        table.to_csv(ANALYSIS_DIR / filename, index=False)


if __name__ == "__main__":
    main()
