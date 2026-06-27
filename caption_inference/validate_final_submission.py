"""Validate an ImageCLEFmedical Caption Prediction submission file.

This script performs internal format checks used by the UIT_HighDefinition
reproducibility workflow. It is not a replacement for the official
ImageCLEFmedical checker.

Example:
    python caption_inference/validate_final_submission.py \
      --submission /path/to/submission.csv \
      --expected_ids /path/to/expected_ids.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def load_expected_ids(path: Path) -> list[str]:
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
        if 'ID' not in df.columns:
            raise ValueError('Expected-IDs CSV must contain an ID column.')
        return df['ID'].astype(str).tolist()
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission', type=Path, required=True, help='submission.csv with ID,Caption columns')
    parser.add_argument('--expected_ids', type=Path, default=None, help='Optional txt/csv file with expected IDs')
    args = parser.parse_args()

    df = pd.read_csv(args.submission)
    print('Columns:', df.columns.tolist())
    if df.columns.tolist() != ['ID', 'Caption']:
        raise ValueError('Submission must have exactly two columns: ID,Caption')

    df['ID'] = df['ID'].astype(str)
    df['Caption'] = df['Caption'].fillna('').astype(str)

    duplicate_ids = int(df['ID'].duplicated().sum())
    empty_captions = int(df['Caption'].str.strip().eq('').sum())
    unique_captions = int(df['Caption'].nunique())

    print('Rows:', len(df))
    print('Duplicate IDs:', duplicate_ids)
    print('Empty captions:', empty_captions)
    print('Unique captions:', unique_captions)

    if duplicate_ids:
        raise ValueError('Duplicate IDs detected.')
    if empty_captions:
        raise ValueError('Empty captions detected.')

    if args.expected_ids is not None:
        expected = load_expected_ids(args.expected_ids)
        expected_set = set(expected)
        actual_set = set(df['ID'])
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        print('Expected rows:', len(expected))
        print('Missing IDs:', len(missing))
        print('Extra IDs:', len(extra))
        if len(df) != len(expected) or missing or extra:
            raise ValueError('Submission IDs do not match expected IDs.')

    print('Internal validation passed.')


if __name__ == '__main__':
    main()
