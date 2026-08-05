"""Write a CellPhoneDB-v3-compatible AnnData file from Step 4's pickle."""

import argparse
import pickle
from pathlib import Path

import anndata
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Pickle written by Step 4.")
    parser.add_argument("output", type=Path, help="CellPhoneDB-compatible h5ad path.")
    args = parser.parse_args()

    with args.input.open("rb") as handle:
        data = pickle.load(handle)

    adata = anndata.AnnData(
        X=data["X"],
        obs=pd.DataFrame(index=data["obs_index"]),
        var=pd.DataFrame(index=data["var_index"]),
    )
    adata.write_h5ad(args.output)
    args.input.unlink()


if __name__ == "__main__":
    main()
