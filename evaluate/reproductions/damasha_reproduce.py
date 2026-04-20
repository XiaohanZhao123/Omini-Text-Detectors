"""Reproduce DAMASHA (saiteja33/DAMASHA-RMC) on its own DAMASHA-MAS benchmark.

Paper: "DAMASHA: Detecting AI in Mixed Adversarial Texts via Segmentation
with Human-interpretable Attribution" (Lekkala Sai Teja et al., AAAI 2026 sub).
GitHub: https://github.com/saitejalekkala33/DAMASHA
HF model card: https://huggingface.co/saiteja33/DAMASHA-RMC
HF dataset:    https://huggingface.co/datasets/saiteja33/DAMASHA

Reported in README & model card (paper's "RMC*" best-config):
    Token-level: Accuracy / Precision / Recall / F1 ≈ 0.98
    Span-level (strict):  SBDA ≈ 0.45, SegPre ≈ 0.41
    Span-level (relaxed IoU ≥ 0.5): ≈ 0.82
    The card explicitly warns: "exact numbers for THIS specific checkpoint
    may differ depending on training run and configuration."

Setup choices (no test split documented anywhere, see notes below):
- Dataset file: `DAMASHA_Final_No_ADV.csv` (162 MB, 96,692 rows). This is the
  smallest published DAMASHA-MAS file and matches the non-adversarial setting
  the headline ~0.98 corresponds to. The 6 attacked CSVs are 325–921 MB each.
- Schema: `hybrid_text` is a string with `<AI_Start>...</AI_End>` markers
  delimiting AI-generated spans. `has_pair` is constantly 1.
- We compute per-word ground-truth labels by aligning the markers to the
  whitespace-tokenized cleaned text (matches the wrapper's tokenization).
- We subsample 1000 rows with seed=42 (no defined test split exists).
- Metric: micro-averaged token-level F1 across all (sample, word) pairs,
  positive class = AI. This matches `boundary_metrics.token_level_metrics`
  but micro-pooled to mirror the paper's single 0.98 number rather than a
  mean-of-per-doc number.

Wrapper: `omini_text.pipeline("ai-text-detection", model="damasha")`. The
wrapper enforces min_words=30 (paper's training assumption).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evaluate.boundary_metrics import token_level_metrics, span_iou  # noqa: E402
from omini_text import pipeline  # noqa: E402


AI_TAG_RE = re.compile(r"</?AI_Start>|</?AI_End>")


def parse_word_labels(hybrid_text: str) -> tuple[list[str], list[int], list[list[int]]]:
    """Convert a tagged text into (words, word_labels, ai_char_intervals).

    Procedure:
      1. Walk through hybrid_text, toggling ai_flag on `<AI_Start>` and off on
         `</AI_End>`. Build the cleaned text incrementally and remember the
         AI-flag at each character position.
      2. Whitespace-tokenise the cleaned text — same scheme as the wrapper
         (`text.split()`) so labels align 1-to-1 with predictions.
      3. A word is labelled AI if any of its characters were inside an AI span.
      4. Char intervals are derived from the cleaned-text positions.
    """
    cleaned: list[str] = []
    ai_flags: list[bool] = []  # per cleaned-char
    intervals_clean: list[list[int]] = []
    cur_start: int | None = None
    ai_active = False
    i = 0
    n = len(hybrid_text)
    while i < n:
        if hybrid_text.startswith("<AI_Start>", i):
            if not ai_active:
                cur_start = len(cleaned)
            ai_active = True
            i += len("<AI_Start>")
            continue
        if hybrid_text.startswith("</AI_End>", i):
            if ai_active and cur_start is not None:
                intervals_clean.append([cur_start, len(cleaned)])
                cur_start = None
            ai_active = False
            i += len("</AI_End>")
            continue
        # also tolerate edge variants seen rarely
        if hybrid_text.startswith("<AI_End>", i):
            if ai_active and cur_start is not None:
                intervals_clean.append([cur_start, len(cleaned)])
                cur_start = None
            ai_active = False
            i += len("<AI_End>")
            continue
        if hybrid_text.startswith("</AI_Start>", i):
            i += len("</AI_Start>")
            continue
        cleaned.append(hybrid_text[i])
        ai_flags.append(ai_active)
        i += 1
    if ai_active and cur_start is not None:
        intervals_clean.append([cur_start, len(cleaned)])

    cleaned_str = "".join(cleaned)
    words = cleaned_str.split()

    # Word-level labels using the same str.split scheme the wrapper uses
    word_labels: list[int] = []
    pos = 0
    for w in words:
        # find this word starting at pos in cleaned_str
        start = cleaned_str.find(w, pos)
        if start == -1:
            start = pos
        end = start + len(w)
        # AI if ANY char in [start, end) is AI-flagged
        word_is_ai = any(ai_flags[start:end])
        word_labels.append(1 if word_is_ai else 0)
        pos = end
    return words, word_labels, intervals_clean


def make_clean_text(hybrid_text: str) -> str:
    return AI_TAG_RE.sub("", hybrid_text)


def main(args: argparse.Namespace) -> None:
    print("[damasha] downloading DAMASHA_Final_No_ADV.csv (162 MB)")
    csv_path = hf_hub_download(
        repo_id="saiteja33/DAMASHA",
        filename="DAMASHA_Final_No_ADV.csv",
        repo_type="dataset",
    )
    print(f"  cached at: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    print(f"  loaded {len(df)} rows, columns={list(df.columns)}")

    # Paper-faithful filter: only the wrapper's documented min_words=30 guard.
    # Previously we also imposed max_words<=350 to dodge RoBERTa's 512-subtoken
    # truncation tail-padding artefact; that was an unpublished cherry-pick
    # that inflated token F1 by ~3.8 pt. We now keep all docs >=30 words and
    # apply a *window-aware mask* below at scoring time (see lines comparing
    # `wl_int` length vs GT) — words beyond the model window are excluded
    # from BOTH pred and GT arrays, not silently labelled human=0.
    df["__clean_words"] = df["hybrid_text"].apply(
        lambda t: len(make_clean_text(t).split())
    )
    n_total = len(df)
    df = df[df["__clean_words"] >= 30].reset_index(drop=True)
    print(
        f"  filtered to {len(df)} rows with n_words >= 30 "
        f"(dropped {n_total - len(df)})"
    )

    rng = np.random.default_rng(args.seed)
    if args.n_samples < len(df):
        idx = rng.choice(len(df), size=args.n_samples, replace=False)
        idx.sort()
        df = df.iloc[idx].reset_index(drop=True)
    print(f"  subsampled {len(df)} rows with seed={args.seed}")

    # Build ground truth + cleaned texts
    gt_word_labels: list[list[int]] = []
    gt_intervals: list[list[list[int]]] = []
    cleaned_texts: list[str] = []
    skipped = 0
    for _, row in df.iterrows():
        words, labels, intervals = parse_word_labels(row["hybrid_text"])
        if not words or len(words) < 30:
            skipped += 1
            continue
        gt_word_labels.append(labels)
        gt_intervals.append(intervals)
        cleaned_texts.append(" ".join(words))  # canonical form, matches str.split()
    print(f"  built GT for {len(cleaned_texts)} rows (skipped {skipped})")

    # Class balance
    total_words = sum(len(l) for l in gt_word_labels)
    total_ai = sum(sum(l) for l in gt_word_labels)
    print(f"  GT class balance: {total_ai}/{total_words} = {total_ai/total_words:.3f} AI words")

    # Load wrapper pipeline
    print(f"[damasha] loading wrapper (device=cuda:{args.gpu})")
    pipe = pipeline(
        "ai-text-detection",
        model="damasha",
        device=f"cuda:{args.gpu}",
    )

    # Inference (one-by-one — wrapper has no batched path; ~1-2s each)
    print(f"[damasha] running inference on {len(cleaned_texts)} samples")
    t0 = time.time()
    pred_word_labels: list[list[int]] = []
    pred_intervals: list[list[list[int]]] = []
    pred_doc_labels: list[int] = []
    pred_scores: list[float] = []
    err = 0
    for i, text in enumerate(cleaned_texts):
        try:
            out = pipe(text)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}] inference error: {e}")
            err += 1
            pred_word_labels.append([0] * len(gt_word_labels[i]))
            pred_intervals.append([])
            pred_doc_labels.append(0)
            pred_scores.append(0.0)
            continue
        wl = out["metadata"].get("word_labels", [])
        # Map "ai"/"human" strings to int.
        wl_int = [1 if w == "ai" else 0 for w in wl]
        # Window-aware mask: the wrapper exposes `n_scored_words` = the number
        # of input words that actually got at least one subtoken inside the
        # 512-subtoken model window. Words beyond that index are silently
        # padded as `human=0` inside `_map_to_words` and would otherwise create
        # false negatives whenever AI content lives in the truncated tail.
        # We clip BOTH wrapper output AND GT to that window.
        n_scored = int(out["metadata"].get("n_scored_words", len(wl_int)))
        win = min(n_scored, len(wl_int), len(gt_word_labels[i]))
        wl_int = wl_int[:win]
        gt_word_labels[i] = gt_word_labels[i][:win]
        pred_word_labels.append(wl_int)
        pred_intervals.append(out["metadata"].get("ai_intervals", []))
        pred_doc_labels.append(int(out["label"]))
        pred_scores.append(float(out["score"]))
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"  [{i + 1}/{len(cleaned_texts)}] "
                f"elapsed={elapsed:.1f}s rate={(i + 1) / elapsed:.2f}/s"
            )
    elapsed = time.time() - t0
    print(f"[damasha] done in {elapsed:.1f}s ({err} errors)")

    pipe.cleanup()

    # ----- Metrics -----
    # Micro-pooled token-level metrics (concatenate all word labels across docs)
    flat_pred = np.concatenate([np.array(p) for p in pred_word_labels])
    flat_true = np.concatenate([np.array(t) for t in gt_word_labels])
    micro = token_level_metrics(flat_pred.tolist(), flat_true.tolist())

    # Macro per-doc token F1 (average of per-doc F1)
    per_doc = [token_level_metrics(p, t) for p, t in zip(pred_word_labels, gt_word_labels)]
    macro_f1 = float(np.mean([m["f1"] for m in per_doc]))
    macro_acc = float(np.mean([m["accuracy"] for m in per_doc]))
    macro_prec = float(np.mean([m["precision"] for m in per_doc]))
    macro_rec = float(np.mean([m["recall"] for m in per_doc]))

    # Span IoU at character level — wrapper returns char intervals in the
    # cleaned text, GT intervals_clean are also cleaned-text indices
    iou_list = [span_iou(p, t) for p, t in zip(pred_intervals, gt_intervals)]
    mean_iou = float(np.mean(iou_list))
    relaxed_iou_05 = float(np.mean([1.0 if x >= 0.5 else 0.0 for x in iou_list]))

    print("\n===== Results =====")
    print(f"Samples scored: {len(cleaned_texts)}  (errors: {err})")
    print(f"Runtime: {elapsed:.1f}s ({len(cleaned_texts)/elapsed:.2f} samples/s)")
    print(f"\nToken-level (MICRO-pooled across all words, AI=positive):")
    print(f"  Accuracy : {micro['accuracy']:.4f}")
    print(f"  Precision: {micro['precision']:.4f}")
    print(f"  Recall   : {micro['recall']:.4f}")
    print(f"  F1       : {micro['f1']:.4f}")
    print(f"\nToken-level (MACRO mean of per-doc, AI=positive):")
    print(f"  Accuracy : {macro_acc:.4f}")
    print(f"  Precision: {macro_prec:.4f}")
    print(f"  Recall   : {macro_rec:.4f}")
    print(f"  F1       : {macro_f1:.4f}")
    print(f"\nSpan-level (char IoU on cleaned text):")
    print(f"  Mean IoU         : {mean_iou:.4f}")
    print(f"  IoU >= 0.5 (relaxed): {relaxed_iou_05:.4f}")
    print("\nPaper (RMC*, model card, no-ADV):")
    print("  Token-level F1 ≈ 0.98 ; Span IoU>=0.5 ≈ 0.82")

    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = out_dir / f"damasha_no_adv_{stamp}"
    run_dir.mkdir(exist_ok=True)

    # Per-sample JSONL
    with open(run_dir / "predictions.jsonl", "w") as f:
        for i, (pred, true, ints_p, ints_t, lab, sc) in enumerate(
            zip(
                pred_word_labels,
                gt_word_labels,
                pred_intervals,
                gt_intervals,
                pred_doc_labels,
                pred_scores,
            )
        ):
            f.write(
                json.dumps(
                    {
                        "idx": int(i),
                        "n_words": len(true),
                        "gt_ai_word_count": int(sum(true)),
                        "pred_ai_word_count": int(sum(pred)),
                        "pred_doc_label": int(lab),
                        "pred_score": float(sc),
                        "iou": float(span_iou(ints_p, ints_t)),
                        "token_f1": float(token_level_metrics(pred, true)["f1"]),
                    }
                )
                + "\n"
            )

    summary = {
        "detector": "damasha",
        "wrapper": 'omini_text.pipeline("ai-text-detection", model="damasha")',
        "checkpoint": "saiteja33/DAMASHA-RMC :: RoBERTa_ModernBERT_CRF.pth",
        "dataset": "saiteja33/DAMASHA :: DAMASHA_Final_No_ADV.csv",
        "n_samples_total_in_csv": int(n_total),
        "min_words_filter": 30,
        "max_words_filter": None,  # paper has no length cap; window-aware scoring used instead
        "scoring_protocol": "window-aware: token-F1 computed on first N words where N = min(wrapper_output_len, gt_len)",
        "n_samples_after_word_count_filter": int(len(df)),
        "n_samples_evaluated": int(len(cleaned_texts)),
        "n_inference_errors": int(err),
        "seed": args.seed,
        "runtime_s": float(elapsed),
        "samples_per_s": float(len(cleaned_texts) / elapsed),
        "gt_class_balance_ai_word_frac": float(total_ai / total_words),
        "token_level_micro": {k: float(v) for k, v in micro.items()},
        "token_level_macro_per_doc": {
            "accuracy": macro_acc,
            "precision": macro_prec,
            "recall": macro_rec,
            "f1": macro_f1,
        },
        "span_iou_mean": mean_iou,
        "span_iou_ge_0p5": relaxed_iou_05,
        "paper_targets": {
            "token_f1_approx": 0.98,
            "span_iou_relaxed_approx": 0.82,
            "span_strict_sbda_approx": 0.45,
            "span_strict_segpre_approx": 0.41,
        },
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="number of random rows to evaluate (no test split is published)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0, help="cuda index inside process")
    p.add_argument(
        "--out_dir",
        default=str(Path(__file__).parent / "results"),
    )
    main(p.parse_args())
