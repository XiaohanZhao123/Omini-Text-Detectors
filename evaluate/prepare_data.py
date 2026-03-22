#!/usr/bin/env python3
"""Prepare downloaded data into standard JSONL format.

Transforms raw CSV/Excel/JSON files into the format expected by data_loader.py.

Usage:
    python prepare_data.py data/external/sondos                     # prepare all files
    python prepare_data.py data/external/sondos --inspect           # just inspect
    python prepare_data.py data/external/sondos --human_col text --ai_col text_eval

Output format (one JSONL per dataset):
    {"human_text": "...", "ai_text": "...", "domain": "essay", "ai_model": "gpt", ...}
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd


# Column mappings: adjust these when new data arrives
COLUMN_DEFAULTS = {
    "essays": {"human": "text", "ai": "text_eval", "domain": "essay", "model": "gpt"},
    "abstracts": {"human": "abstract", "ai": "generated_abstract", "domain": "academic", "model": "gpt"},
}


def detect_type(filepath: Path) -> str:
    """Guess dataset type from filename."""
    name = filepath.stem.lower()
    if "essay" in name:
        return "essays"
    if "abstract" in name:
        return "abstracts"
    return "unknown"


def read_file(filepath: Path) -> pd.DataFrame:
    """Read any supported file format into a DataFrame."""
    s = filepath.suffix.lower()
    if s == ".csv":
        return pd.read_csv(filepath)
    elif s == ".parquet":
        return pd.read_parquet(filepath)
    elif s == ".jsonl":
        return pd.read_json(filepath, lines=True)
    elif s == ".json":
        return pd.read_json(filepath)
    elif s in (".xlsx", ".xls"):
        return pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported format: {s}")


def inspect(input_dir: str):
    """Print summary of all data files in a directory."""
    root = Path(input_dir)
    exts = {".csv", ".jsonl", ".json", ".parquet", ".xlsx", ".xls"}
    files = [f for f in sorted(root.rglob("*")) if f.is_file() and f.suffix.lower() in exts]

    if not files:
        print(f"No data files in {input_dir}")
        return

    for fp in files:
        mb = fp.stat().st_size / 1024 / 1024
        print(f"\n{'='*60}")
        print(f"{fp.relative_to(root)} ({mb:.1f} MB)")
        try:
            df = read_file(fp)
            print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} cols")
            print(f"  Columns: {list(df.columns)}")
            for col in df.columns:
                if df[col].dtype == "object":
                    lengths = df[col].dropna().str.len()
                    if len(lengths) > 0:
                        print(f"    {col}: avg_len={lengths.mean():.0f}, "
                              f"nulls={df[col].isnull().sum()}")
        except Exception as e:
            print(f"  Error: {e}")


def prepare(
    filepath: Path,
    output_dir: Path,
    human_col: str,
    ai_col: str,
    domain: str,
    model: str,
):
    """Transform a data file into standard JSONL format."""
    df = read_file(filepath)
    print(f"\nPreparing {filepath.name}: {len(df)} rows")
    print(f"  Mapping: human={human_col}, ai={ai_col}")

    # Validate columns exist
    if human_col not in df.columns:
        print(f"  ERROR: Column '{human_col}' not found. Available: {list(df.columns)}")
        return
    if ai_col not in df.columns:
        print(f"  ERROR: Column '{ai_col}' not found. Available: {list(df.columns)}")
        return

    # Filter empty rows
    mask = df[human_col].notna() & df[ai_col].notna()
    mask &= df[human_col].astype(str).str.strip().str.len() > 0
    mask &= df[ai_col].astype(str).str.strip().str.len() > 0
    df = df[mask]
    print(f"  After filtering: {len(df)} rows")

    # Write JSONL — use source filename to avoid overwriting
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{filepath.stem}_prepared.jsonl"

    # Preserve all other columns as "extra"
    extra_cols = [c for c in df.columns if c not in {human_col, ai_col}]

    with open(out_path, "w") as f:
        for idx, row in df.iterrows():
            rec = {
                "human_text": str(row[human_col]),
                "ai_text": str(row[ai_col]),
                "domain": domain,
                "ai_model": model,
                "source_file": filepath.name,
                "line_index": int(idx),
            }
            # Save extra columns (version info, labels, etc.)
            extra = {}
            for c in extra_cols:
                val = row.get(c)
                if pd.notna(val):
                    if isinstance(val, float) and math.isfinite(val) and val == int(val):
                        extra[c] = int(val)
                    else:
                        extra[c] = val
            if extra:
                rec["extra"] = extra

            f.write(json.dumps(rec, default=str) + "\n")

    print(f"  Saved: {out_path} ({len(df)} records)")
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_dir", help="Directory with downloaded data files")
    p.add_argument("--output_dir", default=None,
                   help="Output dir (default: input_dir/prepared)")
    p.add_argument("--inspect", action="store_true", help="Only inspect, don't prepare")
    p.add_argument("--human_col", default=None, help="Override human text column name")
    p.add_argument("--ai_col", default=None, help="Override AI text column name")
    args = p.parse_args()

    if args.inspect:
        inspect(args.input_dir)
        return

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir) if args.output_dir else input_path / "prepared"

    exts = {".csv", ".jsonl", ".json", ".parquet", ".xlsx", ".xls"}
    files = [f for f in sorted(input_path.rglob("*"))
             if f.is_file() and f.suffix.lower() in exts and "prepared" not in f.stem]

    if not files:
        print(f"No data files in {args.input_dir}")
        sys.exit(1)

    for fp in files:
        ds_type = detect_type(fp)
        defaults = COLUMN_DEFAULTS.get(ds_type, COLUMN_DEFAULTS["essays"])

        human_col = args.human_col or defaults["human"]
        ai_col = args.ai_col or defaults["ai"]

        prepare(
            filepath=fp,
            output_dir=output_path,
            human_col=human_col,
            ai_col=ai_col,
            domain=defaults["domain"],
            model=defaults["model"],
        )

    print(f"\nDone. Prepared files in: {output_path}")


if __name__ == "__main__":
    main()
