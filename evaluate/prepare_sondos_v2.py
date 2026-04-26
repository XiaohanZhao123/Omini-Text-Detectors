#!/usr/bin/env python3
"""Standardize Sondos v2 data for the unified training and evaluation pipelines.

Reads the raw v2 CSVs (4 domains x 3 AI models) and produces:
  1. Unified CSV   — drop-in replacement for finetune/dataset.py
  2. AES-chains JSONL — drop-in replacement for evaluate/prepare_sft_data.py
                        and evaluate/data_loader.py (aes_chains loader)

The two outputs share identical data; only the format differs so that
every downstream pipeline can consume the data without modification.

Directory layout expected:
    data_local/external/sondos/v2/
        Abstract/Abstract/*.csv
        Essays/Essays/*.csv
        News/News/*.csv
        Reports/Reports/*.csv

Usage:
    # Inspect all files first
    python evaluate/prepare_sondos_v2.py --inspect

    # Prepare all data (default output: data_local/external/sondos/v2/prepared/)
    python evaluate/prepare_sondos_v2.py

    # Prepare specific domains
    python evaluate/prepare_sondos_v2.py --domains abstract essays

    # Custom output directory
    python evaluate/prepare_sondos_v2.py --out data/sondos_v2/
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


# ============================================================================
# Constants
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data_local" / "external" / "sondos" / "v2"
DEFAULT_OUTPUT = DEFAULT_INPUT / "prepared"

# Map subdirectory name to domain label
DOMAIN_MAP = {
    "Abstract": "abstract",
    "Essays": "essay",
    "News": "news",
    "Reports": "report",
}

# Columns that finetune/dataset.py relies on
REQUIRED_COLS = [
    "essay_id", "version", "split", "model_used", "operation",
    "AI_token_ratio", "text_clean", "tokens", "tok_labels",
]

# Columns we always carry through (superset across old/new data).
# Sentence-level columns (`sentences`, `sent_labels`, …) are provided by Sondos in
# the April 14 v2 `_clean.csv` files and are preserved verbatim so downstream
# pipelines can use them without re-deriving from token-level labels.
STANDARD_COLS = [
    "essay_id", "version", "split", "domain", "ai_model",
    "model_used", "operation",
    "num_paragraphs", "num_sentences", "essay_length",
    "C_para_meaningful", "C_sent_target",
    "AI_sent_ratio", "Avg_sent_ai_frac", "Avg_sent_ai_frac_touched",
    "AI_token_ratio", "AI_char_ratio",
    "C_para_measured", "Avg_para_ai_frac",
    "text_clean", "text_tagged",
    "ai_spans_char", "ai_spans_tok",
    "num_ai_spans_tok", "avg_ai_span_len_tok",
    "tokens", "tok_labels", "boundary_pattern",
    # Sentence-level fields from Sondos's v2 _clean CSVs (April 14, 2026+)
    "sentences", "sent_labels",
    "sentences_number", "v0_sentences_number", "sentence_count_match_v0",
]


# ============================================================================
# Helpers
# ============================================================================

def discover_csvs(input_dir: Path, domains: list[str] | None = None):
    """Find all v2 CSV files, grouped by domain.

    Returns list of (domain, ai_model, path) tuples.
    """
    results = []
    for subdir_name, domain in DOMAIN_MAP.items():
        if domains and domain not in domains:
            continue
        # Handle the nested SubDir/SubDir/*.csv layout from unzip
        candidates = sorted(input_dir.glob(f"{subdir_name}/**/*.csv"))
        # Skip macOS metadata
        candidates = [c for c in candidates if "__MACOSX" not in str(c)]
        for csv_path in candidates:
            # Extract ai_model from filename
            ai_model = _extract_model(csv_path.stem)
            results.append((domain, ai_model, csv_path))
    return results


def _extract_model(stem: str) -> str:
    """Extract a short model identifier from the CSV filename.

    Examples:
        abstracts_v0_v8_spans_gpt-5.4-2026-03-05   -> gpt-5.4
        essays_v0_v8_spans_gpt-5.4-nano-2026-03-17 -> gpt-5.4-nano
        news_v0_v8_spans_gemini-2.5-flash          -> gemini-2.5-flash
        news_v0_v8_spans_qwen3-8b                  -> qwen3-8b
        gove_report_summries_parahLevel_ai         -> unknown
    """
    stem_lower = stem.lower()
    if "gpt-5.4-nano" in stem_lower:
        return "gpt-5.4-nano"
    elif "gpt-5.4" in stem_lower:
        return "gpt-5.4"
    elif "gemini-2.5-flash" in stem_lower:
        return "gemini-2.5-flash"
    elif "qwen3-8b" in stem_lower:
        return "qwen3-8b"
    else:
        return "unknown"


def load_and_standardize(domain: str, ai_model: str, csv_path: Path) -> pd.DataFrame:
    """Load a single CSV and standardize its columns.

    Deduplicates on (essay_id, version) to protect against known upstream
    duplication in Sondos's Abstract v2 _clean CSVs (≈30–40% duplicate keys —
    some identical, some with minor generation variants for the same slot).
    The first occurrence is kept for deterministic, reproducible output.
    """
    df = pd.read_csv(csv_path)
    n_in = len(df)

    # Normalize ID column: News uses 'id' instead of 'essay_id'
    if "id" in df.columns and "essay_id" not in df.columns:
        df = df.rename(columns={"id": "essay_id"})

    # Add domain and ai_model columns
    df["domain"] = domain
    df["ai_model"] = ai_model

    # Ensure essay_id is string
    df["essay_id"] = df["essay_id"].astype(str)

    # --- Deduplication -----------------------------------------------------
    # Step 1: drop fully-identical rows (cheap, unambiguous).
    df = df.drop_duplicates()
    # Step 2: drop (essay_id, version) key duplicates, keeping the first row.
    # Sondos's data model is one row per (essay_id, version); extra rows with
    # the same key are treated as redundant variants regardless of text diffs.
    if {"essay_id", "version"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["essay_id", "version"], keep="first")
    n_out = len(df)
    if n_in != n_out:
        print(f"    [dedup] {csv_path.name}: {n_in} -> {n_out} rows "
              f"({n_in - n_out} duplicates removed)")

    # Keep only standard columns that exist in this file
    keep = [c for c in STANDARD_COLS if c in df.columns]
    df = df[keep]

    return df


# ============================================================================
# Output: Unified CSV
# ============================================================================

def write_unified_csv(
    csv_files: list[tuple[str, str, Path]],
    output_dir: Path,
):
    """Write standardized per-domain CSVs.

    Processes one source CSV at a time and appends to per-domain output
    files to keep memory usage low.  Each domain CSV contains all AI
    models for that domain with standardized columns.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    domain_counts = Counter()
    header_written = set()

    for domain, ai_model, csv_path in csv_files:
        print(f"  Loading {domain}/{ai_model}: {csv_path.name}")
        df = load_and_standardize(domain, ai_model, csv_path)
        print(f"    -> {len(df)} rows")

        out_path = output_dir / f"{domain}.csv"
        write_header = domain not in header_written
        df.to_csv(out_path, index=False, mode="a" if not write_header else "w",
                  header=write_header)
        header_written.add(domain)
        domain_counts[domain] += len(df)
        del df

    for domain, count in sorted(domain_counts.items()):
        print(f"  CSV ({domain}): {output_dir / f'{domain}.csv'}  ({count} rows)")


# ============================================================================
# Output: AES-chains JSONL
# ============================================================================

def write_aes_chains_jsonl(csv_files: list[tuple[str, str, Path]], output_dir: Path):
    """Convert CSV files to AES-chains JSONL format, one file at a time.

    Processes each CSV independently to avoid loading all data into memory.
    Groups rows by (essay_id) within each CSV and builds a history array
    with one entry per version.  Token labels are converted from 0/1 ints
    to "human"/"ai" strings to match the existing pipeline convention.

    Produces:
      - Per-domain JONSLs (appended across models)
      - One combined JSONL (all data)
    """
    import csv as csv_mod

    output_dir.mkdir(parents=True, exist_ok=True)

    # Track per-domain doc counts
    domain_counts = Counter()

    # Open per-domain and combined output files
    combined_path = output_dir / "all_domains.jsonl"
    domain_files = {}

    with open(combined_path, "w") as f_combined:
        for domain, ai_model, csv_path in csv_files:
            print(f"  Processing {csv_path.name} ({domain}/{ai_model})...")

            # Open per-domain file in append mode
            domain_path = output_dir / f"{domain}.jsonl"
            if domain not in domain_files:
                domain_files[domain] = open(domain_path, "w")

            # Stream CSV and group by essay_id. Dedupe on (essay_id, version)
            # keeping the first row; rows with duplicate keys are counted and
            # reported so we notice if upstream duplication regresses.
            docs = defaultdict(dict)     # {eid: {version: row}}
            id_col = None
            csv_row_dups = 0

            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv_mod.DictReader(f)
                # Detect ID column
                if "essay_id" in reader.fieldnames:
                    id_col = "essay_id"
                elif "id" in reader.fieldnames:
                    id_col = "id"
                else:
                    print(f"    WARNING: no ID column found, skipping")
                    continue

                for row in reader:
                    eid = row[id_col]
                    ver = row.get("version", "")
                    if ver in docs[eid]:
                        csv_row_dups += 1
                        continue  # keep first occurrence, drop later duplicates
                    docs[eid][ver] = row

            if csv_row_dups:
                print(f"    [dedup] {csv_path.name}: dropped {csv_row_dups} "
                      f"duplicate (essay_id, version) rows from JSONL stream")

            # Convert each document group to a record
            for eid, version_map in docs.items():
                rows = sorted(version_map.values(),
                              key=lambda r: r.get("version", ""))
                history = []

                for row in rows:
                    tok_labels_raw = _parse_json_field(row.get("tok_labels", "[]"))
                    tokens_raw = _parse_json_field(row.get("tokens", "[]"))

                    token_labels_str = [
                        "ai" if lab == 1 else "human" for lab in tok_labels_raw
                    ]

                    text = row.get("text_clean", "")
                    if not text or not text.strip():
                        text = " ".join(str(t) for t in tokens_raw)

                    # Prefer Sondos's explicit sentence-level labels when the
                    # CSV provides them (v2 _clean files, April 14+). Fall back
                    # to deriving from token labels for older CSV layouts.
                    sentences_raw = _parse_json_field(row.get("sentences", "[]"))
                    sent_labels_raw = _parse_json_field(row.get("sent_labels", "[]"))
                    if sent_labels_raw:
                        sentence_labels = [
                            "ai" if lab == 1 else "human"
                            for lab in sent_labels_raw
                        ]
                    else:
                        sentence_labels = _derive_sentence_labels(
                            tokens_raw, token_labels_str,
                        )

                    ai_ratio = row.get("AI_token_ratio", "0.0")
                    try:
                        ai_ratio = float(ai_ratio)
                    except (ValueError, TypeError):
                        ai_ratio = 0.0

                    op = row.get("operation")
                    if op in ("", "nan", "None"):
                        op = None

                    entry = {
                        "version_id": row.get("version", ""),
                        "text": text,
                        "token_labels": token_labels_str,
                        "sentence_labels": sentence_labels,
                        "operation": op,
                        "ai_ratio": ai_ratio,
                    }
                    # Surface Sondos's raw sentence list when available so
                    # sentence-level evaluators don't have to re-split text.
                    if sentences_raw:
                        entry["sentences"] = sentences_raw
                    history.append(entry)

                record = {
                    "q_id": str(eid),
                    "domain": domain,
                    "ai_model": ai_model,
                    "history": history,
                }

                line = json.dumps(record) + "\n"
                f_combined.write(line)
                domain_files[domain].write(line)
                domain_counts[domain] += 1

            del docs  # free memory before next file

    # Close per-domain files
    for f in domain_files.values():
        f.close()

    print(f"  JSONL (combined): {combined_path}  ({sum(domain_counts.values())} docs)")
    for domain, count in sorted(domain_counts.items()):
        print(f"  JSONL ({domain}): {output_dir / f'{domain}.jsonl'}  ({count} docs)")


def _parse_json_field(val) -> list:
    """Parse a JSON-encoded field (tokens or tok_labels)."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def _derive_sentence_labels(tokens: list, token_labels: list[str]) -> list[str]:
    """Derive sentence-level labels from token-level labels.

    Uses a simple sentence splitter: split on sentence-ending punctuation
    followed by a capitalized word.  For each sentence, if any token is
    "ai", the sentence is labeled "ai".
    """
    if not tokens or not token_labels or len(tokens) != len(token_labels):
        return []

    sentence_ends = {'.', '!', '?'}
    sentences = []
    current_labels = []

    for i, (tok, lab) in enumerate(zip(tokens, token_labels)):
        current_labels.append(lab)
        tok_str = str(tok)
        # Check for sentence boundary
        if (tok_str and tok_str[-1] in sentence_ends
                and i + 1 < len(tokens)
                and str(tokens[i + 1])[:1].isupper()):
            has_ai = "ai" in current_labels
            sentences.append("ai" if has_ai else "human")
            current_labels = []

    # Last sentence
    if current_labels:
        has_ai = "ai" in current_labels
        sentences.append("ai" if has_ai else "human")

    return sentences


def _write_jsonl(records: list[dict], path: Path):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ============================================================================
# Inspect
# ============================================================================

def inspect(input_dir: Path, domains: list[str] | None = None):
    """Print summary of all discovered CSVs."""
    csvs = discover_csvs(input_dir, domains)
    if not csvs:
        print(f"No CSV files found under {input_dir}")
        return

    print(f"\nFound {len(csvs)} CSV files under {input_dir}\n")
    print(f"{'Domain':<10} {'AI Model':<20} {'Rows':>8} {'Docs':>7} "
          f"{'Versions':<25} {'Splits':<18} {'File'}")
    print("-" * 120)

    total_rows = 0
    for domain, ai_model, csv_path in csvs:
        try:
            df = pd.read_csv(csv_path, usecols=lambda c: c in {
                "essay_id", "id", "version", "split",
            })
            id_col = "id" if "id" in df.columns else "essay_id"
            n_docs = df[id_col].nunique()
            versions = sorted(df["version"].unique())
            splits = sorted(df["split"].unique())
            n_rows = len(df)
            total_rows += n_rows
            print(f"{domain:<10} {ai_model:<20} {n_rows:>8} {n_docs:>7} "
                  f"{','.join(versions):<25} {','.join(splits):<18} "
                  f"{csv_path.name}")
        except Exception as e:
            print(f"{domain:<10} {ai_model:<20} {'ERROR':>8}  {csv_path.name}: {e}")

    print(f"\nTotal: {total_rows} rows across {len(csvs)} files")


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", default=str(DEFAULT_INPUT),
                   help="Root directory containing Abstract/, Essays/, etc.")
    p.add_argument("--out", default=str(DEFAULT_OUTPUT),
                   help="Output directory for prepared files")
    p.add_argument("--domains", nargs="+", default=None,
                   choices=list(DOMAIN_MAP.values()),
                   help="Only process these domains (default: all)")
    p.add_argument("--inspect", action="store_true",
                   help="Only inspect data, don't write output")
    p.add_argument("--no-csv", action="store_true",
                   help="Skip CSV output (only produce JSONL)")
    p.add_argument("--no-jsonl", action="store_true",
                   help="Skip JSONL output (only produce CSV)")
    args = p.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.out)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)

    if args.inspect:
        inspect(input_dir, args.domains)
        return

    # Discover CSV files
    csv_files = discover_csvs(input_dir, args.domains)
    if not csv_files:
        print(f"No CSV files found under {input_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files\n")

    # Write CSV outputs (one source file at a time)
    if not args.no_csv:
        print("Writing per-domain CSVs...")
        write_unified_csv(csv_files, output_dir / "csv")
        print()

    # Write JSONL outputs (streams one CSV at a time, low memory)
    if not args.no_jsonl:
        print("Writing AES-chains JSONL...")
        write_aes_chains_jsonl(csv_files, output_dir / "jsonl")
        print()

    print(f"Done. Output: {output_dir}")
    print()
    print("Next steps:")
    print(f"  # Fine-tune on a single domain:")
    print(f"  python finetune/train.py --data {output_dir}/csv/essay.csv ...")
    print(f"  # Prepare SFT data:")
    print(f"  python evaluate/prepare_sft_data.py {output_dir}/jsonl/all_domains.jsonl")
    print(f"  # Evaluate:")
    print(f"  python evaluate/run_eval.py e5-small custom:{output_dir}/jsonl/essay.jsonl")


if __name__ == "__main__":
    main()