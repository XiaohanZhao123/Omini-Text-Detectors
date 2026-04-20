"""Derive per-sentence AI/human metrics for DAMASHA from saved predictions.

DAMASHA paper defines ONLY token-level + span-boundary metrics; it does not
prescribe a sentence-level aggregation. For HAT-Bench sentence evaluation we
adopt a `>50% majority` rule (wrapper convention, not paper-prescribed):

    a sentence is predicted AI iff > 50% of its words are labelled AI;
    otherwise it is predicted human.

Per-sentence continuous score is the AI-word ratio, so a downstream harness
can re-threshold without re-running the model.

Inputs (read per cell):
    results/new_data_eval/doc_level/default_setting/damasha_new4d_*/predictions.jsonl

For each row we need these fields (present in our saved DAMASHA output):
    detector.metadata.word_labels   : list["ai"|"human"]
    detector.metadata.word_positions: list[[char_start, char_end]]
    sentences                       : list[str]  (HAT-Bench GT sentences)
    gt_sent_labels                  : list[int]  (HAT-Bench GT per-sentence)
    text_clean                      : full text
    version, operation, model_used  : slice keys

Output (written per cell):
    summary_sentence.json  — canonical shape, sibling of summary.json
      {
        "aggregation_rule": ">50% majority of word_labels per sentence",
        "note": "Paper defines no sentence-level rule; this is a wrapper convention.",
        "metrics_overall"      : {...},
        "metrics_by_version"   : {...},
        "metrics_by_operation" : {...},
        "metrics_by_generator" : {...},
      }

Usage:
    uv run python evaluate/aes/scripts/derive_damasha_sentence.py \
        --root results/new_data_eval/doc_level/default_setting
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.reproductions.run_on_new_data import compute_metrics, slice_metrics  # noqa: E402

SENTINEL_RULE = ">50% majority of word_labels per sentence"
SENTINEL_NOTE = (
    "Paper defines no sentence-level rule; this is a wrapper convention. "
    "Per-sentence score is the AI-word ratio, so the threshold can be "
    "re-derived downstream without re-running the model."
)


def _parse_list(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            return None
    return None


def sentence_char_ranges(sentences: list[str], text_clean: str) -> list[tuple[int, int]]:
    """Walk text_clean left-to-right; for each sentence, find its span.

    Uses `text_clean.find(sentence, cursor)` — robust to leading whitespace
    between sentences because HAT-Bench `sentences` are already trimmed
    tag-boundary segments that appear verbatim in `text_clean`.
    """
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for s in sentences:
        if not s:
            ranges.append((cursor, cursor))
            continue
        idx = text_clean.find(s, cursor)
        if idx < 0:
            # Fallback: stretch forward by sentence length.
            idx = cursor
        ranges.append((idx, idx + len(s)))
        cursor = idx + len(s)
    return ranges


def per_sentence_ai_ratio(
    word_labels: list[str],
    word_positions: list[list[int]],
    sent_ranges: list[tuple[int, int]],
) -> tuple[list[float], list[int]]:
    """For each sentence return (ai_ratio, n_words_in_sentence).

    A word is assigned to the sentence whose char range contains the word's
    start offset. Words outside all sentence ranges (truncation tail) are
    dropped from the ratio denominator.
    """
    n = len(sent_ranges)
    ai_counts = [0] * n
    tot_counts = [0] * n
    for lbl, pos in zip(word_labels, word_positions):
        if not pos or len(pos) < 1:
            continue
        start = pos[0]
        # Find owning sentence (linear walk; sentences are already in order).
        for i, (a, b) in enumerate(sent_ranges):
            if a <= start < b:
                tot_counts[i] += 1
                if lbl == "ai":
                    ai_counts[i] += 1
                break
    ratios = [
        (ai_counts[i] / tot_counts[i]) if tot_counts[i] > 0 else 0.0
        for i in range(n)
    ]
    return ratios, tot_counts


def rows_for_cell(cell: Path) -> list[dict]:
    """Read predictions.jsonl, return one record per (row, sentence)."""
    pj = cell / "predictions.jsonl"
    if not pj.exists():
        return []

    out: list[dict] = []
    for line in pj.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        det = r.get("detector") or {}
        md = det.get("metadata") or {}

        word_labels = md.get("word_labels")
        word_positions = md.get("word_positions")
        sentences = _parse_list(r.get("sentences"))
        gt_sent = _parse_list(r.get("gt_sent_labels"))
        text_clean = r.get("text_clean", "") or ""

        if not (word_labels and word_positions and sentences and gt_sent):
            continue

        sent_ranges = sentence_char_ranges(sentences, text_clean)
        ratios, tots = per_sentence_ai_ratio(word_labels, word_positions, sent_ranges)

        version = str(r.get("version", ""))
        operation = str(r.get("operation", ""))
        model_used = str(r.get("model_used", ""))

        for i, gt in enumerate(gt_sent):
            if i >= len(ratios):
                break
            ai_ratio = float(ratios[i])
            pred_label = 1 if ai_ratio > 0.5 else 0
            out.append({
                "gt_label": int(gt),
                "pred_label": pred_label,
                "pred_score": ai_ratio,
                "version": version,
                "operation": operation,
                "model_used": model_used,
            })
    return out


def summarize_cell(cell: Path) -> dict | None:
    rows = rows_for_cell(cell)
    if not rows:
        return None
    yt = [r["gt_label"] for r in rows]
    yp = [r["pred_label"] for r in rows]
    ys = [r["pred_score"] for r in rows]
    return {
        "aggregation_rule": SENTINEL_RULE,
        "note": SENTINEL_NOTE,
        "metrics_overall":      compute_metrics(yt, yp, ys),
        "metrics_by_version":   slice_metrics(rows, "version"),
        "metrics_by_operation": slice_metrics(rows, "operation"),
        "metrics_by_generator": slice_metrics(rows, "model_used"),
        "n_sentences_scored":   len(rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                    help="A default_setting/ directory containing damasha cells.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cells = sorted(
        d for d in args.root.iterdir()
        if d.is_dir() and d.name.startswith("damasha_")
    )
    if not cells:
        print(f"[warn] no damasha_* cells under {args.root}", file=sys.stderr)
        return 1

    print(f"[scan] {len(cells)} damasha cells under {args.root}")
    for cell in cells:
        summary = summarize_cell(cell)
        if summary is None:
            print(f"[skip] {cell.name} — no usable rows")
            continue
        ov = summary["metrics_overall"]
        print(
            f"[ok]   {cell.name}  "
            f"n_sent={summary['n_sentences_scored']:>6}  "
            f"acc={ov.get('accuracy'):.4f}  "
            f"f1_ai={ov.get('f1_ai'):.4f}  "
            f"f1_h={ov.get('f1_human'):.4f}  "
            f"auroc={ov.get('auroc')}"
        )
        if args.dry_run:
            continue
        out_path = cell / "summary_sentence.json"
        out_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
