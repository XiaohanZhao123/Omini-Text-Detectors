#!/usr/bin/env python3
"""Extract SeqXGPT per-word log-likelihood features from 4 LLMs on OpAI-Bench.

SeqXGPT is a "forward-pass only" detector: each of 4 LLMs scores every word's
log-likelihood under that LLM, and a tiny downstream CRF+Transformer classifier
is trained on the resulting (num_words, 4) features. No backward pass through
the 4 LLMs is ever needed.

This script runs the forward pass. Later, a small classifier is fine-tuned on
the saved features alone.

Design:
  - Load ONE LLM at a time (keeps VRAM tight; doesn't race with running jobs).
  - For each LLM, stream through all docs (train/val/test across 4 domains),
    writing one jsonl entry per doc.
  - Resumable: if the output jsonl already has a doc_id we skip re-extraction.
  - Uses the EXACT feature extraction stack from the SeqXGPT codebase
    (BBPETokenizerPPLCalc / SPLlamaTokenizerPPLCalc from baseline/seqxgpt/...).

Output layout:
  data_local/external/opai_bench/v2/prepared/seqxgpt_features/
    {split}/
      {llm_name}.jsonl       one entry per doc: {doc_id, ll_tokens, begin_idx}
      meta.jsonl             one entry per doc: {doc_id, split, domain, essay_id,
                                                version, ai_model, AI_sent_ratio,
                                                text_clean, words, tok_labels}
      progress.json          per-(split,llm) counts (for monitoring)

Usage (one LLM per invocation; launch 4 copies on 4 GPUs OR serial on 1 GPU):
  uv run python analysis/extract_seqxgpt_features.py \\
      --llm gpt2-xl \\
      --gpu 0 \\
      --splits train dev test \\
      --domains essay abstract news report
"""
from __future__ import annotations
import argparse, ast, hashlib, json, os, sys, time
from pathlib import Path
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SEQ_ROOT = _REPO_ROOT / "baseline" / "seqxgpt" / "SeqXGPT"
_SEQ_CLASSIFIER = _SEQ_ROOT / "SeqXGPT"
# Put repo root first so `omini_text.*` resolves; SeqXGPT paths after.
for p in (_REPO_ROOT, _SEQ_ROOT, _SEQ_CLASSIFIER):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

PREPARED_CSV_DIR = Path(
    "data_local/external/opai_bench/v2/prepared/csv"
)
OUT_DIR = Path(
    "data_local/external/opai_bench/v2/prepared/seqxgpt_features"
)


# ---------------------------------------------------------------------------
# Doc id / meta
# ---------------------------------------------------------------------------
def doc_id(row) -> str:
    key = f"{row['essay_id']}|{row['version']}|{row['ai_model']}|{row.get('operation', '')}|{row.get('domain', '')}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def load_rows(domains, split, csv_dir=None):
    """Load (doc_id, text_clean, words, tok_labels, meta) for each row."""
    csv_root = Path(csv_dir) if csv_dir else PREPARED_CSV_DIR
    rows = []
    for dom in domains:
        csv = csv_root / f"{dom}.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if "split" in df.columns:
            df = df[df["split"].str.lower().str.strip() == split]
        for _, r in df.iterrows():
            text = str(r.get("text_clean", "")).strip()
            if not text:
                continue
            try:
                words = ast.literal_eval(r["tokens"]) if isinstance(r["tokens"], str) else r["tokens"]
                tlabs = ast.literal_eval(r["tok_labels"]) if isinstance(r["tok_labels"], str) else r["tok_labels"]
            except Exception:
                words, tlabs = [], []
            rows.append({
                "doc_id": doc_id(r),
                "split": split,
                "domain": dom,
                "essay_id": r.get("essay_id", ""),
                "version": r.get("version", ""),
                "ai_model": r.get("ai_model", ""),
                "operation": r.get("operation", ""),
                "AI_sent_ratio": float(r.get("AI_sent_ratio", 0.0) or 0.0),
                "text_clean": text,
                "words": words,
                "tok_labels": tlabs,
            })
    return rows


def write_meta_once(split, rows, out_dir):
    """Write the per-doc meta exactly once per split (idempotent)."""
    mpath = out_dir / split / "meta.jsonl"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    if mpath.exists():
        # Trust existing file if its row count matches
        n = sum(1 for _ in mpath.open())
        if n == len(rows):
            return
    with mpath.open("w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in
                ["doc_id", "split", "domain", "essay_id", "version",
                 "ai_model", "operation", "AI_sent_ratio",
                 "text_clean", "words", "tok_labels"]},
                ensure_ascii=False) + "\n")


def load_existing_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    with path.open() as f:
        for line in f:
            try:
                ids.add(json.loads(line)["doc_id"])
            except Exception:
                pass
    return ids


# ---------------------------------------------------------------------------
# LLM feature extractor -- uses the already-written FeatureExtractor in the
# unified detector wrapper. That wrapper's FeatureExtractor takes a list of
# model names and loads each with the same 8-bit / fp32 settings as the
# original SeqXGPT codebase.
# ---------------------------------------------------------------------------
def load_extractor(llm_name, device, cache_dir):
    import torch
    from omini_text.detectors.seqxgpt_detector import FeatureExtractor
    extractor = FeatureExtractor(
        model_names=[llm_name],
        devices=[device],
        cache_dir=cache_dir,
    )
    return extractor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True,
                    choices=["gpt2-xl", "gpt-neo-2.7b", "gpt-j-6b", "llama-7b"])
    ap.add_argument("--gpu", required=True,
                    help="GPU id (e.g. '0'); also sets CUDA_VISIBLE_DEVICES for safety")
    ap.add_argument("--splits", nargs="+", default=["test", "dev", "train"],
                    help="Run small splits first so we fail fast if anything's wrong")
    ap.add_argument("--domains", nargs="+",
                    default=["essay", "abstract", "news", "report"])
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--csv-dir", default=str(PREPARED_CSV_DIR),
                    help="Directory containing <domain>.csv prepared files")
    ap.add_argument("--cache-dir",
                    default="cache")
    ap.add_argument("--flush-every", type=int, default=50,
                    help="Flush jsonl output every N docs")
    ap.add_argument("--max-docs-per-split", type=int, default=None,
                    help="Cap for debugging")
    # For cross-GPU parallelism within one LLM / split: each helper takes a slice
    # (half, third, etc.) of the doc list. Doc ordering is deterministic so
    # disjoint [start, end) slices write to different doc_ids -> no file conflict.
    ap.add_argument("--doc-range-start", type=float, default=0.0,
                    help="Fractional start within each split (0.0 = beginning)")
    ap.add_argument("--doc-range-end", type=float, default=1.0,
                    help="Fractional end within each split (1.0 = end)")
    # Gap-filler mode: K-way shard across the full row list, so 4 workers all
    # share [0, 1.0) range but each only touches rows where i % K == shard.
    # Combined with `already`-set filtering this parallelizes recovery of
    # missing docs without conflicts.
    ap.add_argument("--shard-i", type=int, default=0)
    ap.add_argument("--shard-of", type=int, default=1)
    args = ap.parse_args()
    assert 0 <= args.shard_i < args.shard_of, "shard-i must be in [0, shard-of)"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # After setting CUDA_VISIBLE_DEVICES=<gpu>, the only visible GPU is cuda:0
    device = "cuda:0"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Materialize all rows + write meta once per split
    split_rows = {}
    for split in args.splits:
        rows = load_rows(args.domains, split, csv_dir=args.csv_dir)
        if args.max_docs_per_split:
            rows = rows[:args.max_docs_per_split]
        # Write meta with FULL rows so the output stays consistent across slices
        write_meta_once(split, rows, out_dir)
        # Then apply the range slice just for this worker's work
        s = int(len(rows) * args.doc_range_start)
        e = int(len(rows) * args.doc_range_end)
        split_rows[split] = rows[s:e]
        print(f"[meta] {split}: {len(rows)} docs total, this worker handles [{s}:{e}) = {e-s} docs",
              flush=True)

    # 2. Load the one LLM and iterate
    print(f"[extract] Loading {args.llm} on {device} (visible: {args.gpu})...",
          flush=True)
    extractor = load_extractor(args.llm, device=device, cache_dir=args.cache_dir)

    total_start = time.time()
    for split in args.splits:
        rows = split_rows[split]
        out_path = out_dir / split / f"{args.llm}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        already = load_existing_ids(out_path)
        progress_path = out_dir / split / "progress.json"

        print(f"[extract] {split}: {len(rows)} docs, {len(already)} already done",
              flush=True)
        n_done = 0
        n_err = 0
        t0 = time.time()
        with out_path.open("a") as fout:
            for i, row in enumerate(rows):
                if args.shard_of > 1 and (i % args.shard_of) != args.shard_i:
                    continue
                if row["doc_id"] in already:
                    continue
                try:
                    ll_tokens_list, begin_idx_list = extractor.extract(row["text_clean"])
                    rec = {
                        "doc_id": row["doc_id"],
                        "ll_tokens": ll_tokens_list[0],
                        "begin_idx": int(begin_idx_list[0]),
                    }
                    fout.write(json.dumps(rec) + "\n")
                except Exception as e:
                    n_err += 1
                    print(f"  [err] doc {row['doc_id']} split={split}: "
                          f"{type(e).__name__}: {e}", flush=True)
                    continue
                n_done += 1
                if n_done % args.flush_every == 0:
                    fout.flush()
                    elapsed = time.time() - t0
                    rate = n_done / max(1e-6, elapsed)
                    eta = (len(rows) - len(already) - n_done) / max(1e-6, rate)
                    msg = (f"  [progress] {split}/{args.llm} {n_done + len(already)}/"
                           f"{len(rows)}  rate={rate:.2f} doc/s  eta={eta/60:.1f}min  "
                           f"errors={n_err}")
                    print(msg, flush=True)
                    progress_path.write_text(json.dumps({
                        "llm": args.llm, "split": split,
                        "done": n_done + len(already), "total": len(rows),
                        "rate_doc_per_s": rate, "eta_min": eta / 60,
                        "errors": n_err,
                        "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }, indent=2))

        print(f"[extract] {split}/{args.llm}: {n_done} new docs in "
              f"{(time.time()-t0)/60:.1f}min (errors={n_err})", flush=True)

    print(f"[done] {args.llm} all splits finished in "
          f"{(time.time()-total_start)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
