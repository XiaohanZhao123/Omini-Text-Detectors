#!/usr/bin/env python3
"""Prepare Sondos's external dataset for evaluation pipeline integration.

This script inspects and transforms the downloaded essays/abstracts data into
a standardized JSONL format that the evaluation data_loader can consume.

The input data (from Sondos) contains:
- Essay data (~4,200 essays) with GPT-generated text in `text_eval` column
- Abstract data (format TBD, added as models finish)

This script is designed to be **adaptable** — you can modify the column
mappings and transformation logic as new data arrives.

Usage:
    # Inspect raw data (see columns, shapes, sample rows)
    python prepare_sondos_data.py --inspect --input_dir data/external/sondos

    # Prepare essays dataset
    python prepare_sondos_data.py \
        --input_dir data/external/sondos \
        --output_dir data/external/sondos/prepared

    # Prepare with custom column mappings (if columns differ from expected)
    python prepare_sondos_data.py \
        --input_dir data/external/sondos \
        --human_text_col "original_text" \
        --ai_text_col "text_eval" \
        --domain_col "domain" \
        --model_col "model"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# ============================================================================
# COLUMN MAPPING CONFIGURATION
# Adjust these if the data columns change when new models are added
# ============================================================================

DEFAULT_COLUMN_MAP = {
    # --- Essays dataset (essays_v0_v8_spans_finall_eval.csv) ---
    "essays": {
        # Column containing the original human-written text
        "human_text": "text",
        # Column containing AI-generated text
        "ai_text": "text_eval",
        # Optional: column for domain/category
        "domain": None,  # Set to column name if available, e.g., "essay_type"
        # Optional: column for AI model name
        "ai_model": None,  # Set to column name if available, e.g., "model"
        # Optional: column for task type
        "task": None,  # Set to column name if available
        # Optional: ID column
        "id": None,  # Set to column name if available, e.g., "essay_id"
        # Fixed values when column not available
        "default_domain": "essay",
        "default_task": "generation",
        "default_ai_model": "gpt",  # Currently GPT-only, update as models added
    },
    # --- Abstracts dataset (format TBD) ---
    "abstracts": {
        "human_text": "abstract",
        "ai_text": "generated_abstract",
        "domain": None,
        "ai_model": None,
        "task": None,
        "id": None,
        "default_domain": "academic",
        "default_task": "abstract_generation",
        "default_ai_model": "gpt",
    },
}


def inspect_file(filepath: Path) -> Dict:
    """Inspect a single data file and return its metadata.

    Returns dict with: columns, shape, dtypes, sample_values, null_counts
    """
    suffix = filepath.suffix.lower()
    info = {"path": str(filepath), "suffix": suffix}

    try:
        if suffix == ".csv":
            df = pd.read_csv(filepath)
        elif suffix == ".parquet":
            df = pd.read_parquet(filepath)
        elif suffix == ".jsonl":
            df = pd.read_json(filepath, lines=True)
        elif suffix == ".json":
            df = pd.read_json(filepath)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(filepath)
        else:
            info["error"] = f"Unsupported format: {suffix}"
            return info

        info["shape"] = df.shape
        info["columns"] = list(df.columns)
        info["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        info["null_counts"] = {col: int(df[col].isnull().sum()) for col in df.columns}

        # Sample values for each column
        info["sample_values"] = {}
        for col in df.columns:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                sample = non_null.iloc[0]
                # Truncate long strings
                if isinstance(sample, str) and len(sample) > 100:
                    sample = sample[:100] + "..."
                info["sample_values"][col] = str(sample)

        # Text length statistics for string columns
        info["text_lengths"] = {}
        for col in df.columns:
            if df[col].dtype == "object":
                lengths = df[col].dropna().str.len()
                if len(lengths) > 0:
                    info["text_lengths"][col] = {
                        "min": int(lengths.min()),
                        "max": int(lengths.max()),
                        "mean": round(float(lengths.mean()), 1),
                        "median": round(float(lengths.median()), 1),
                    }

        # Label distribution if there are obvious label columns
        info["value_counts"] = {}
        for col in df.columns:
            nunique = df[col].nunique()
            if 2 <= nunique <= 20:  # Likely a categorical/label column
                info["value_counts"][col] = df[col].value_counts().to_dict()

    except Exception as e:
        info["error"] = str(e)

    return info


def inspect_directory(input_dir: str):
    """Inspect all data files in a directory."""
    data_path = Path(input_dir)
    if not data_path.exists():
        print(f"Directory not found: {input_dir}")
        print("Run download_gdrive.py first to download the data.")
        return

    print(f"Inspecting: {data_path.resolve()}")
    print("=" * 70)

    supported = {".csv", ".jsonl", ".json", ".parquet", ".xlsx", ".xls"}
    files = [f for f in sorted(data_path.rglob("*")) if f.is_file() and f.suffix.lower() in supported]

    if not files:
        print("No supported data files found.")
        print(f"Looked for: {supported}")
        return

    for filepath in files:
        info = inspect_file(filepath)
        rel = filepath.relative_to(data_path)
        size_mb = filepath.stat().st_size / (1024 * 1024)

        print(f"\n{'='*70}")
        print(f"FILE: {rel} ({size_mb:.1f} MB)")
        print(f"{'='*70}")

        if "error" in info:
            print(f"  ERROR: {info['error']}")
            continue

        rows, cols = info["shape"]
        print(f"  Shape: {rows} rows x {cols} columns")
        print(f"\n  Columns:")
        for col in info["columns"]:
            dtype = info["dtypes"].get(col, "?")
            nulls = info["null_counts"].get(col, 0)
            sample = info["sample_values"].get(col, "N/A")
            print(f"    {col:<30} {dtype:<15} nulls={nulls:<5} sample={sample}")

        if info.get("text_lengths"):
            print(f"\n  Text Length Statistics:")
            for col, stats in info["text_lengths"].items():
                print(f"    {col}: min={stats['min']}, max={stats['max']}, "
                      f"mean={stats['mean']}, median={stats['median']}")

        if info.get("value_counts"):
            print(f"\n  Categorical Distributions:")
            for col, counts in info["value_counts"].items():
                print(f"    {col}: {counts}")


def detect_dataset_type(filepath: Path) -> Optional[str]:
    """Auto-detect dataset type from filename."""
    name = filepath.stem.lower()
    if "essay" in name:
        return "essays"
    elif "abstract" in name:
        return "abstracts"
    return None


def prepare_dataset(
    filepath: Path,
    dataset_type: str,
    column_map: Dict,
    output_dir: Path,
    max_samples: int = None,
) -> Path:
    """Transform a raw data file into standardized JSONL format.

    Output format (per line):
    {
        "human_text": "...",
        "ai_text": "...",
        "domain": "essay",
        "task": "generation",
        "ai_model": "gpt",
        "source_file": "essays_v0_v8_spans_finall_eval.csv",
        "line_index": 0,
        "extra": { ... any additional columns ... }
    }

    Args:
        filepath: Path to input file
        dataset_type: "essays" or "abstracts"
        column_map: Column mapping configuration
        output_dir: Directory for output files
        max_samples: Maximum samples to include (None = all)

    Returns:
        Path to output JSONL file
    """
    suffix = filepath.suffix.lower()

    # Load data
    if suffix == ".csv":
        df = pd.read_csv(filepath)
    elif suffix == ".parquet":
        df = pd.read_parquet(filepath)
    elif suffix == ".jsonl":
        df = pd.read_json(filepath, lines=True)
    elif suffix == ".json":
        df = pd.read_json(filepath)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    print(f"Loaded {len(df)} rows from {filepath.name}")
    print(f"Columns: {list(df.columns)}")

    # Validate required columns exist
    human_col = column_map["human_text"]
    ai_col = column_map["ai_text"]

    if human_col not in df.columns:
        # Try to find a close match
        candidates = [c for c in df.columns if "human" in c.lower() or "original" in c.lower() or c.lower() == "text"]
        raise ValueError(
            f"Human text column '{human_col}' not found in data. "
            f"Available columns: {list(df.columns)}. "
            f"Candidates: {candidates}"
        )
    if ai_col not in df.columns:
        candidates = [c for c in df.columns if "ai" in c.lower() or "eval" in c.lower() or "generated" in c.lower()]
        raise ValueError(
            f"AI text column '{ai_col}' not found in data. "
            f"Available columns: {list(df.columns)}. "
            f"Candidates: {candidates}"
        )

    # Apply max_samples
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        print(f"Sampled {max_samples} rows")

    # Filter out rows with missing text
    initial_count = len(df)
    df = df.dropna(subset=[human_col, ai_col])
    df = df[df[human_col].str.strip().str.len() > 0]
    df = df[df[ai_col].str.strip().str.len() > 0]
    if len(df) < initial_count:
        print(f"Filtered {initial_count - len(df)} rows with missing/empty text")

    # Prepare output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_type}_prepared.jsonl"

    # Build metadata column extractors
    domain_col = column_map.get("domain")
    model_col = column_map.get("ai_model")
    task_col = column_map.get("task")
    id_col = column_map.get("id")

    # Columns to preserve as "extra" metadata
    mapped_cols = {human_col, ai_col}
    if domain_col:
        mapped_cols.add(domain_col)
    if model_col:
        mapped_cols.add(model_col)
    if task_col:
        mapped_cols.add(task_col)
    if id_col:
        mapped_cols.add(id_col)
    extra_cols = [c for c in df.columns if c not in mapped_cols]

    records_written = 0
    with open(output_path, "w") as f:
        for idx, row in df.iterrows():
            record = {
                "human_text": str(row[human_col]),
                "ai_text": str(row[ai_col]),
                "domain": str(row[domain_col]) if domain_col and pd.notna(row.get(domain_col)) else column_map.get("default_domain", "unknown"),
                "task": str(row[task_col]) if task_col and pd.notna(row.get(task_col)) else column_map.get("default_task", "unknown"),
                "ai_model": str(row[model_col]) if model_col and pd.notna(row.get(model_col)) else column_map.get("default_ai_model", "unknown"),
                "source_file": filepath.name,
                "line_index": int(idx),
            }

            # Preserve extra columns
            if extra_cols:
                extra = {}
                for col in extra_cols:
                    val = row.get(col)
                    if pd.notna(val):
                        extra[col] = val if not isinstance(val, float) else (int(val) if val == int(val) else val)
                if extra:
                    record["extra"] = extra

            f.write(json.dumps(record, default=str) + "\n")
            records_written += 1

    print(f"Written {records_written} records to {output_path}")

    # Print summary statistics
    print(f"\nSummary for {dataset_type}:")
    print(f"  Total paired records: {records_written}")
    print(f"  Human text avg length: {df[human_col].str.len().mean():.0f} chars")
    print(f"  AI text avg length: {df[ai_col].str.len().mean():.0f} chars")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Sondos's external data for evaluation pipeline"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/external/sondos",
        help="Directory containing downloaded data files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/external/sondos/prepared",
        help="Output directory for prepared JSONL files",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Only inspect data files (no transformation)",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["essays", "abstracts", "auto"],
        default="auto",
        help="Dataset type (auto-detect from filename if 'auto')",
    )
    # Column override arguments
    parser.add_argument("--human_text_col", type=str, default=None,
                        help="Override: column name for human text")
    parser.add_argument("--ai_text_col", type=str, default=None,
                        help="Override: column name for AI-generated text")
    parser.add_argument("--domain_col", type=str, default=None,
                        help="Override: column name for domain/category")
    parser.add_argument("--model_col", type=str, default=None,
                        help="Override: column name for AI model identifier")
    parser.add_argument("--task_col", type=str, default=None,
                        help="Override: column name for task type")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples to prepare (default: all)")

    args = parser.parse_args()

    if args.inspect:
        inspect_directory(args.input_dir)
        return

    # Find data files
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Input directory not found: {args.input_dir}")
        print("Run download_gdrive.py first to download the data.")
        sys.exit(1)

    supported = {".csv", ".jsonl", ".json", ".parquet", ".xlsx", ".xls"}
    files = [
        f for f in sorted(input_path.rglob("*"))
        if f.is_file() and f.suffix.lower() in supported
        and "prepared" not in f.stem  # Skip already-prepared files
    ]

    if not files:
        print(f"No data files found in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(files)} data files:")
    for f in files:
        print(f"  {f.relative_to(input_path)} ({f.stat().st_size / 1024 / 1024:.1f} MB)")

    output_path = Path(args.output_dir)

    for filepath in files:
        # Determine dataset type
        if args.dataset_type == "auto":
            ds_type = detect_dataset_type(filepath)
            if ds_type is None:
                print(f"\nSkipping {filepath.name} - cannot auto-detect type. "
                      "Use --dataset_type to specify.")
                continue
        else:
            ds_type = args.dataset_type

        # Build column map with overrides
        column_map = DEFAULT_COLUMN_MAP.get(ds_type, DEFAULT_COLUMN_MAP["essays"]).copy()
        if args.human_text_col:
            column_map["human_text"] = args.human_text_col
        if args.ai_text_col:
            column_map["ai_text"] = args.ai_text_col
        if args.domain_col:
            column_map["domain"] = args.domain_col
        if args.model_col:
            column_map["ai_model"] = args.model_col
        if args.task_col:
            column_map["task"] = args.task_col

        print(f"\n{'='*70}")
        print(f"Preparing: {filepath.name} (type={ds_type})")
        print(f"Column map: human={column_map['human_text']}, ai={column_map['ai_text']}")
        print(f"{'='*70}")

        try:
            prepare_dataset(
                filepath=filepath,
                dataset_type=ds_type,
                column_map=column_map,
                output_dir=output_path,
                max_samples=args.max_samples,
            )
        except Exception as e:
            print(f"ERROR preparing {filepath.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nAll prepared files saved to: {output_path}")
    print("\nNext steps:")
    print("  1. Verify prepared files: python prepare_sondos_data.py --inspect --input_dir data/external/sondos/prepared")
    print("  2. Run evaluation: python run_flexible_eval.py --datasets sondos_essays --detectors e5-small binoculars")


if __name__ == "__main__":
    main()
