"""Merge rank-level Qwen3 caption CSVs and validate the final submission.

Example:
    python caption_inference/merge_and_validate_submission.py \
        --rank_dir /kaggle/working/qwen3_full_6gpu \
        --test_image_root "/kaggle/input/.../TEST DATASET/images" \
        --output_dir /kaggle/working/qwen3_final_submission
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def numeric_suffix_value(x: str) -> int:
    s = str(x)
    m = re.search(r"(\d+)(?!.*\d)", s)
    return int(m.group(1)) if m else 10**18


def numeric_suffix_key(series: pd.Series) -> pd.Series:
    return series.map(numeric_suffix_value)


def collect_expected_ids(test_image_root: Path) -> list[str]:
    image_paths = [p for p in test_image_root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    df = pd.DataFrame({"ID": [p.stem for p in image_paths]})
    df["ID"] = df["ID"].astype(str)
    df = df.sort_values("ID", key=numeric_suffix_key).reset_index(drop=True)
    return df["ID"].tolist()


def read_rank_file(rank_dir: Path, rank: int) -> pd.DataFrame:
    path = rank_dir / f"rank{rank}_of_6" / f"qwen3_full_candidates_rank{rank}_of_6.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing rank file: {path}")
    df = pd.read_csv(path)
    if "ID" not in df.columns:
        raise ValueError(f"Rank file lacks ID column: {path}")
    if "caption_qwen3" not in df.columns:
        raise ValueError(f"Rank file lacks caption_qwen3 column: {path}")
    df["ID"] = df["ID"].astype(str)
    df["caption_qwen3"] = df["caption_qwen3"].fillna("").astype(str)
    return df[["ID", "caption_qwen3"]].drop_duplicates("ID", keep="first")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank_dir", type=Path, required=True)
    parser.add_argument("--test_image_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--zip_name", default="submission.zip")
    args = parser.parse_args()

    expected_ids = collect_expected_ids(args.test_image_root)
    expected_set = set(expected_ids)

    rank_frames = [read_rank_file(args.rank_dir, r) for r in range(6)]
    df = pd.concat(rank_frames, ignore_index=True)
    df["ID"] = df["ID"].astype(str)
    df = df.drop_duplicates("ID", keep="first")

    duplicate_count = df["ID"].duplicated().sum()
    empty_count = df["caption_qwen3"].fillna("").astype(str).str.strip().eq("").sum()
    missing_ids = expected_set - set(df["ID"])
    extra_ids = set(df["ID"]) - expected_set

    print("Expected rows:", len(expected_ids))
    print("Merged rows:", len(df))
    print("Duplicate IDs:", duplicate_count)
    print("Empty captions:", empty_count)
    print("Missing IDs:", len(missing_ids))
    print("Extra IDs:", len(extra_ids))

    if len(df) != len(expected_ids):
        raise ValueError("Merged row count does not match expected test image count.")
    if duplicate_count:
        raise ValueError("Duplicate IDs detected.")
    if empty_count:
        raise ValueError("Empty captions detected.")
    if missing_ids:
        raise ValueError(f"Missing IDs detected, first examples: {list(sorted(missing_ids))[:10]}")
    if extra_ids:
        raise ValueError(f"Extra IDs detected, first examples: {list(sorted(extra_ids))[:10]}")

    order_df = pd.DataFrame({"ID": expected_ids, "_order": range(len(expected_ids))})
    sub = df.rename(columns={"caption_qwen3": "Caption"}).merge(order_df, on="ID", how="left")
    sub = sub.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    sub = sub[["ID", "Caption"]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "submission.csv"
    zip_path = args.output_dir / args.zip_name
    sub.to_csv(csv_path, index=False)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="submission.csv")

    print("Saved CSV:", csv_path)
    print("Saved ZIP:", zip_path)


if __name__ == "__main__":
    main()
