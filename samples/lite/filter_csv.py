"""filter_csv.py — Filter a WaterNet metadata CSV to only rows whose image files exist.

Usage
-----
    python filter_csv.py <csv_file> [<image_folder>] [--output <out.csv>]

Arguments
---------
    csv_file        Path to the input CSV (columns: nome, distancia, motion, gauss).
    image_folder    Directory to look for images in. Defaults to a folder with the
                    same stem as the CSV, in the same directory.
    --output / -o   Output CSV path. Defaults to <stem>_filtered.csv next to the input.

Examples
--------
    # Auto-discover folder (session_01.csv → session_01/)
    python filter_csv.py datasets/session_01.csv

    # Explicit folder and output
    python filter_csv.py session_01.csv ./images/ --output session_01_clean.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


_REQUIRED_COLUMNS = {"nome", "distancia", "motion", "gauss"}


def filter_csv(
    csv_path: Path,
    image_folder: Path,
    output_path: Path,
) -> None:
    # ── Load ──────────────────────────────────────────────────────────────
    if not csv_path.is_file():
        sys.exit(f"[ERROR] CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing_cols = _REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        sys.exit(
            f"[ERROR] CSV is missing required column(s): {sorted(missing_cols)}\n"
            f"        Expected: {sorted(_REQUIRED_COLUMNS)}"
        )

    if not image_folder.is_dir():
        sys.exit(f"[ERROR] Image folder not found: {image_folder}")

    # ── Filter ────────────────────────────────────────────────────────────
    total = len(df)
    mask = df["nome"].apply(lambda name: (image_folder / name).is_file())
    df_filtered = df[mask].reset_index(drop=True)
    kept = len(df_filtered)
    dropped = total - kept

    # ── Save ──────────────────────────────────────────────────────────────
    df_filtered.to_csv(output_path, index=False)

    print(f"[filter_csv] Input  : {csv_path}  ({total:,} rows)")
    print(f"[filter_csv] Folder : {image_folder}")
    print(f"[filter_csv] Kept   : {kept:,} rows  ({dropped:,} dropped — files not found)")
    print(f"[filter_csv] Output : {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy only the rows whose image files exist to a new CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "csv_file",
        type=Path,
        help="Input CSV with columns: nome, distancia, motion, gauss.",
    )
    parser.add_argument(
        "image_folder",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Folder containing the images. "
            "Defaults to a directory with the same stem as the CSV, "
            "in the same parent directory."
        ),
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help=(
            "Output CSV path. "
            "Defaults to <csv_stem>_filtered.csv next to the input file."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    csv_path: Path = args.csv_file.resolve()

    image_folder: Path = (
        args.image_folder.resolve()
        if args.image_folder is not None
        else csv_path.parent / csv_path.stem
    )

    output_path: Path = (
        args.output.resolve()
        if args.output is not None
        else csv_path.parent / f"{csv_path.stem}_filtered.csv"
    )

    filter_csv(csv_path, image_folder, output_path)
