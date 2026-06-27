"""Fix final submission header to match the evaluator-required format: ID,Caption."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--zip_name", default="submission_fixed_header.zip")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if "caption" in df.columns:
        df = df.rename(columns={"caption": "Caption"})

    if list(df.columns) != ["ID", "Caption"]:
        raise ValueError(f"Expected columns ['ID', 'Caption'], got {df.columns.tolist()}")

    print("Rows:", len(df))
    print("Duplicate IDs:", df["ID"].astype(str).duplicated().sum())
    print("Empty captions:", df["Caption"].fillna("").astype(str).str.strip().eq("").sum())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed_csv = args.output_dir / "submission.csv"
    fixed_zip = args.output_dir / args.zip_name

    df.to_csv(fixed_csv, index=False)
    with zipfile.ZipFile(fixed_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(fixed_csv, arcname="submission.csv")

    print("Saved fixed CSV:", fixed_csv)
    print("Saved fixed ZIP:", fixed_zip)


if __name__ == "__main__":
    main()
