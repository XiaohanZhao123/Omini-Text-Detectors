#!/usr/bin/env python3
"""Aggregate Task-1 Qwen transfer-eval results into a single summary table.

Reads summary.json from each per-detector-per-domain run under results/aes_doc_eval_qwen/
and prints a 6x4 matrix of (accuracy, AUROC, f1_ai) per detector/domain.

Handles two output conventions:
  * eval_doc_level.py   -> {method}_qwen_{field}_{model_short}_{ts}/summary.json
  * eval_finetuned_detectors.py -> {method}_{ts}/{domain}/summary.json

Usage:
    uv run python evaluate/aes/_aggregate_qwen.py \
        --results-dir results/aes_doc_eval_qwen
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict


def _extract_doclevel_run(d: Path):
    """Parse `{method}_qwen_{field}_{model}_{ts}/` into (method, field).

    Field is the 3rd token from the right after the timestamp split. E.g.:
        adaloc_qwen_abstracts_Qwen3-8B_2026-04-23_05-40-47
        -> method=adaloc, field=abstracts
    """
    parts = d.name.split("_")
    if len(parts) < 5:
        return None
    # Find the `qwen` dataset-name marker (parts[1])
    if parts[1] != "qwen":
        return None
    method = parts[0]
    field = parts[2]
    return (method, field)


def _extract_ftuned_run(d: Path):
    """Parse `{method}_{ts}/{domain}/` — method is everything before the timestamp."""
    parts = d.name.split("_")
    if len(parts) < 2:
        return None
    # Last 2 parts are date_time, rest is method
    if "-" not in parts[-1] or "-" not in parts[-2]:
        return None
    method = "_".join(parts[:-2])
    return method


def collect(results_dir: Path):
    """Return {(method, field) -> metrics dict}."""
    collected = {}

    # 1. eval_doc_level convention
    for d in sorted(results_dir.glob("*_qwen_*_*")):
        if not d.is_dir():
            continue
        summary = d / "summary.json"
        if not summary.exists():
            continue
        parsed = _extract_doclevel_run(d)
        if not parsed:
            continue
        method, field = parsed
        with open(summary) as f:
            data = json.load(f)
        collected[(method, field)] = {
            "overall": data.get("metrics_overall", {}),
            "by_version": data.get("metrics_by_version", {}),
            "n": data.get("dataset", {}).get("n_samples", 0),
            "runtime_s": data.get("runtime", {}).get("score_seconds"),
            "run_dir": str(d),
        }

    # 2. eval_finetuned_detectors convention
    for top in sorted(results_dir.iterdir()):
        if not top.is_dir() or "_qwen_" in top.name:
            continue
        if top.name.startswith("_"):
            continue
        method = _extract_ftuned_run(top)
        if not method:
            continue
        for sub in sorted(top.iterdir()):
            if not sub.is_dir():
                continue
            summary = sub / "summary.json"
            if not summary.exists():
                continue
            # Field mapping: abstract -> abstracts, essay -> essays, etc.
            field_map = {"abstract": "abstracts", "essay": "essays",
                         "news": "news", "report": "reports"}
            field = field_map.get(sub.name, sub.name)
            with open(summary) as f:
                data = json.load(f)
            # eval_finetuned_detectors.py schema: {document: {...}, token: {...}, ...}
            doc_block = data.get("document", {})
            if doc_block:
                overall = {
                    "accuracy": doc_block.get("accuracy"),
                    "auroc":    doc_block.get("auroc"),
                    "f1_ai":    doc_block.get("ai_f1"),
                    "f1_human": doc_block.get("human_f1"),
                    "precision_ai": doc_block.get("ai_precision"),
                    "recall_ai":    doc_block.get("ai_recall"),
                    "n":            doc_block.get("n"),
                }
            else:
                overall = (data.get("metrics_overall") or
                           data.get("doc_level", {}).get("overall") or
                           data.get("overall", {}))
            by_version = (data.get("by_version") or
                          data.get("metrics_by_version") or
                          data.get("doc_level", {}).get("by_version") or {})
            n = doc_block.get("n", 0) or \
                data.get("dataset", {}).get("n_samples", 0) or \
                data.get("n_samples", 0)
            collected[(method, field)] = {
                "overall": overall,
                "by_version": by_version,
                "n": n,
                "runtime_s": data.get("runtime", {}).get("score_seconds")
                             if isinstance(data.get("runtime"), dict) else None,
                "run_dir": str(sub),
            }

    return collected


def fmt(x, width=6, digits=3):
    if x is None:
        return "    - "
    if isinstance(x, (int, float)):
        return f"{x:.{digits}f}".rjust(width)
    return str(x).rjust(width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/aes_doc_eval_qwen")
    ap.add_argument("--metric", default="all",
                    choices=["all", "accuracy", "auroc", "f1_ai", "f1_human"])
    args = ap.parse_args()

    rd = Path(args.results_dir)
    collected = collect(rd)
    if not collected:
        print(f"No runs found under {rd}")
        return

    # Pivot: rows=method, cols=field
    methods = sorted({k[0] for k in collected})
    fields = ["abstracts", "essays", "news", "reports"]

    print("\n" + "=" * 96)
    print(f"  Task 1 Qwen transfer eval — {rd}")
    print("=" * 96)

    print(f"\n{'detector':<22}", end="")
    for f in fields:
        print(f"  {f:^22}", end="")
    print()
    print(f"{'':22}", end="")
    for f in fields:
        print(f"  {'acc / AUROC / F1':^22}", end="")
    print()
    print("-" * 110)

    for m in methods:
        print(f"{m:<22}", end="")
        for f in fields:
            entry = collected.get((m, f))
            if entry is None:
                print(f"  {'(missing)':^22}", end="")
                continue
            o = entry["overall"]
            acc = o.get("accuracy")
            auroc = o.get("auroc")
            f1 = o.get("f1_ai")
            cell = f"{fmt(acc)}/{fmt(auroc)}/{fmt(f1)}"
            print(f"  {cell:^22}", end="")
        print()

    print("\n" + "-" * 110)
    print(f"{'detector':<22}  {'docs_total':>10}  {'score_seconds_total':>20}")
    for m in methods:
        n_total = sum((e["n"] or 0) for k, e in collected.items() if k[0] == m)
        t_total = sum((e["runtime_s"] or 0) for k, e in collected.items() if k[0] == m)
        print(f"{m:<22}  {n_total:>10d}  {t_total:>19.1f}s")
    print()


if __name__ == "__main__":
    main()
