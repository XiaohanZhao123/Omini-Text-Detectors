#!/usr/bin/env python3
"""Gemma 4 E4B-it sentence-level AI-detection judge.

Standalone runner for the local Gemma judge. Lives outside the main `omini_text`
pipeline because Gemma 4 needs `transformers >= 5.x`, which is incompatible with
the pinned `transformers == 4.50.1` for the rest of the project.

Run from the `gemma_judge` conda env:
    conda run -n gemma_judge python evaluate/aes/gemma_judge_runner.py \\
        --fields essays abstracts news reports \\
        --models gpt-5.4 gpt-5.4-nano gemini-2.5-flash qwen3-8b \\
        --split test

Output schema matches existing llm_judge/ cells exactly so the aggregator
picks them up unchanged. Per-cell layout written under
results/new_data_eval/sentence/per_row/llm_judge/<cell>/.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Settings — match the existing API judges' protocol exactly
# ---------------------------------------------------------------------------

MODEL_NAME = "google/gemma-4-E4B-it"
METHOD = "gemma-4-E4B-it"
GREEDY = True              # do_sample=False — fully deterministic
MAX_NEW_TOKENS = 2048      # matches gpt54 judge's max_completion_tokens
ENABLE_THINKING = False    # disable Gemma 4's <|think|> channel — matches gpt54 "reasoning=none" + gemini "thinking=minimal"
# NOTE: We deliberately do NOT report a calibrated `at_conf>=X` block here.
# The legacy 0.15 threshold used by other judges was calibrated for one specific
# model (gpt54) and silently transferred to others — that's a cross-model
# calibration leak. We report only the LLM's raw hard label.

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _safe_relpath(p) -> str:
    """str(p) relative to PROJECT_ROOT if possible, else absolute."""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)

DATASET_MANIFEST = {
    ("essays", "gpt-5.4"):          "Essays/essays_v0_v8_spans_gpt-5.4-2026-03-05_clean.csv",
    ("essays", "gpt-5.4-nano"):     "Essays/essays_v0_v8_spans_gpt-5.4-nano-2026-03-17_clean.csv",
    ("essays", "gemini-2.5-flash"): "Essays/essays_v0_v8_spans_gemini-2.5-flash_clean.csv",
    ("essays", "qwen3-8b"):         "Qwen3-8B/essays_v0_v8_spans_qwen3-8b.csv",
    ("abstracts", "gpt-5.4"):          "Abstract/abstracts_v0_v8_spans_gpt-5.4-2026-03-05_clean.csv",
    ("abstracts", "gpt-5.4-nano"):     "Abstract/abstracts_v0_v8_spans_gpt-5.4-nano_clean.csv",
    ("abstracts", "gemini-2.5-flash"): "Abstract/abstracts_v0_v8_spans_gemini-2.5-flash_clean.csv",
    ("abstracts", "qwen3-8b"):         "Qwen3-8B/abstracts_v0_v8_spans_qwen3-8b.csv",
    ("news", "gpt-5.4"):          "News/news_v0_v8_spans_gpt-5.4-2026-03-05_clean.csv",
    ("news", "gpt-5.4-nano"):     "News/news_v0_v8_spans_gpt-5.4-nano-2026-03-17_clean.csv",
    ("news", "gemini-2.5-flash"): "News/news_v0_v8_spans_gemini-2.5-flash_clean.csv",
    ("news", "qwen3-8b"):         "Qwen3-8B/news_v0_v8_spans_qwen3-8b.csv",
    ("reports", "gemini-2.5-flash"): "Reports/gov_reports_v0_v8_spans_gemini-2.5-flash_clean.csv",
    ("reports", "gpt-5.4-nano"):     "Reports/gove_report_v0_v8_spans_gpt-5.4-nano_clean.csv",
    ("reports", "qwen3-8b"):         "Qwen3-8B/reports_v0_v8_spans_qwen3-8b.csv",
}

DATA_DIR = PROJECT_ROOT / "draft" / "data_25_04_15"

SENTENCE_CONF_PROMPT = """You are an expert linguist and writing analyst specializing in distinguishing human-written text from AI-generated text.

The following text has been split into numbered sentences. For EACH sentence, classify it as human-written (0) or AI-generated (1), and estimate the probability that it is AI-generated.

Text:
\"\"\"
{numbered_sentences}
\"\"\"

Respond in JSON format:
{{"labels": [0, 1, ...], "confidences": [0.1, 0.9, ...]}}
- labels: array of integers (0 = human-written, 1 = AI-generated), one per sentence.
- confidences: array of floats (0.0 to 1.0), one per sentence. Each is your estimated probability that the sentence is AI-generated (0.0 = certainly human, 1.0 = certainly AI).
- Both arrays must contain exactly {num_sentences} elements, in sentence order.
- Do not include any other keys or text outside the JSON object."""


# ---------------------------------------------------------------------------
# Lenient JSON parser — Gemma without constrained decoding may emit prose
# around the JSON; extract the first {...} block and parse it. One retry on
# failure (length mismatch).
# ---------------------------------------------------------------------------

JSON_BLOCK_RE = re.compile(r"\{[^{}]*\"labels\"[^{}]*\"confidences\"[^{}]*\}", re.DOTALL)


def extract_json(text: str) -> dict | None:
    m = JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # Greedy fallback: find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


def parse_response(text: str, num_sentences: int) -> tuple[list[int], list[float], bool]:
    """Return (labels, confidences, length_mismatch)."""
    parsed = extract_json(text)
    if not parsed or "labels" not in parsed or "confidences" not in parsed:
        return ([0] * num_sentences, [0.0] * num_sentences, True)
    try:
        labels = [max(0, min(1, int(x))) for x in parsed["labels"]]
        confs = [max(0.0, min(1.0, float(x))) for x in parsed["confidences"]]
    except (TypeError, ValueError):
        return ([0] * num_sentences, [0.0] * num_sentences, True)
    length_mismatch = (len(labels) != num_sentences) or (len(confs) != num_sentences)
    # Pad / truncate to expected length
    if len(labels) < num_sentences:
        labels.extend([0] * (num_sentences - len(labels)))
    elif len(labels) > num_sentences:
        labels = labels[:num_sentences]
    if len(confs) < num_sentences:
        confs.extend([0.0] * (num_sentences - len(confs)))
    elif len(confs) > num_sentences:
        confs = confs[:num_sentences]
    return (labels, [round(c, 4) for c in confs], length_mismatch)


# ---------------------------------------------------------------------------
# Model loading + inference
# ---------------------------------------------------------------------------

def load_model(device: str = "cuda:0"):
    print(f"[load] tokenizer + Gemma 4 E4B-it on {device}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Gemma 4 is multimodal — try image-text-to-text class first, fall back to causal LM.
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map=device,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map=device,
        )
    model.eval()
    return tok, model


def _empty_result(err: str | None = None) -> dict:
    return {
        "sentence_labels": [], "sentence_confidences": [],
        "label": 0, "score": 0.0, "model": MODEL_NAME, "variant": METHOD,
        "length_mismatch": False, "error": err,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def _build_prompt(sentences: list[str]) -> tuple[str, int]:
    n = len(sentences)
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    return SENTENCE_CONF_PROMPT.format(numbered_sentences=numbered, num_sentences=n), n


def infer_batch(tok, model, sentences_list: list[list[str]], device: str) -> list[dict]:
    """Batched greedy generation. Variable-length output handled by per-row trim."""
    valid = [(i, s) for i, s in enumerate(sentences_list) if len(s) > 0]
    if not valid:
        return [_empty_result("no sentences") for _ in sentences_list]

    prompts, ns = [], []
    for _, s in valid:
        p, n = _build_prompt(s)
        prompts.append(p); ns.append(n)
    msgs = [[{"role": "user", "content": [{"type": "text", "text": p}]}] for p in prompts]
    # Left-pad so all prompts end at the same column → generated tokens align after `n_in_max`.
    if tok.padding_side != "left":
        tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # enable_thinking=False matches the existing API judges' "minimal/none reasoning"
    # protocol — Gemma 4 has a <|think|> chain-of-thought channel; we explicitly disable.
    enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt",
        return_dict=True, padding=True, enable_thinking=False,
    ).to(device)
    n_in = enc["input_ids"].shape[1]

    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, temperature=None, top_p=None,
            pad_token_id=tok.pad_token_id,
        )
    new_tokens = out[:, n_in:]
    texts = tok.batch_decode(new_tokens, skip_special_tokens=True)

    # Per-essay input-token counts (unpadded) for usage accounting
    per_in = enc["attention_mask"].sum(dim=1).tolist()
    # Per-essay output-token counts (count non-pad in new_tokens)
    pad_id = tok.pad_token_id
    per_out = (new_tokens != pad_id).sum(dim=1).tolist()

    results = [None] * len(sentences_list)
    for slot_idx, (orig_idx, _) in enumerate(valid):
        n_sent = ns[slot_idx]
        text = texts[slot_idx]
        labels, confs, mismatch = parse_response(text, n_sent)
        doc_label = 1 if any(l == 1 for l in labels) else 0
        doc_score = sum(confs) / max(1, len(confs))
        results[orig_idx] = {
            "sentence_labels": labels,
            "sentence_confidences": confs,
            "label": doc_label,
            "score": float(doc_score),
            "model": MODEL_NAME,
            "variant": METHOD,
            "length_mismatch": mismatch,
            "error": None,
            "usage": {"input_tokens": int(per_in[slot_idx]), "output_tokens": int(per_out[slot_idx])},
        }
    for i, r in enumerate(results):
        if r is None:
            results[i] = _empty_result("no sentences")
    return results


# ---------------------------------------------------------------------------
# Per-cell driver
# ---------------------------------------------------------------------------

def process_cell(field: str, model_short: str, csv_path: Path, split: str,
                 tok, model, device: str, limit: int | None,
                 batch_size: int = 1,
                 out_root_override: Path | None = None,
                 cell_name_override: str | None = None,
                 dataset_name_override: str | None = None) -> Path:
    df = pd.read_csv(csv_path)
    if "id" in df.columns and "essay_id" not in df.columns:
        df = df.rename(columns={"id": "essay_id"})
    df = df[df["split"] == split]
    if limit:
        df = df.head(limit)

    out_root = out_root_override or (PROJECT_ROOT / "results/new_data_eval/sentence/per_row/llm_judge")
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cell_prefix = cell_name_override or f"{METHOD}_new4d_{field}_{model_short}"
    out_dir = out_root / f"{cell_prefix}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pj = out_dir / "predictions.jsonl"
    print(f"[run]  {field}/{model_short}: {len(df)} rows, batch_size={batch_size}  →  {out_dir.name}")
    n_api_errors = 0
    n_length_mismatch = 0
    total_in = total_out = 0
    t0 = time.time()

    rows = list(df.iterrows())
    with pj.open("w") as f:
        for i in tqdm(range(0, len(rows), batch_size), desc=f"{field}/{model_short}"):
            batch = rows[i : i + batch_size]
            sent_lists = []
            gt_lists = []
            for _, row in batch:
                try:
                    sentences = json.loads(row["sentences"]) if isinstance(row["sentences"], str) else row["sentences"]
                    gt_labels = json.loads(row["sent_labels"]) if isinstance(row["sent_labels"], str) else row["sent_labels"]
                except (TypeError, ValueError, KeyError):
                    sentences, gt_labels = [], []
                sent_lists.append(list(sentences) if sentences else [])
                gt_lists.append(list(gt_labels) if hasattr(gt_labels, "__iter__") else [])

            try:
                results = infer_batch(tok, model, sent_lists, device)
            except Exception as e:
                results = [_empty_result(str(e)) for _ in batch]

            for (_, row), gt_labels, result in zip(batch, gt_lists, results):
                if result.get("error"):
                    n_api_errors += 1
                if result.get("length_mismatch"):
                    n_length_mismatch += 1
                total_in += result["usage"]["input_tokens"]
                total_out += result["usage"]["output_tokens"]
                payload = dict(row.to_dict())
                payload["gt_sent_labels"] = gt_labels
                payload["gpt"] = result
                f.write(json.dumps(payload, default=str) + "\n")

    runtime = time.time() - t0

    # --- summary.json (matches API-judge schema) ---
    rows = [json.loads(l) for l in pj.open()]
    n_docs = len(rows)
    n_pos = sum(1 for r in rows if any(r.get("gt_sent_labels", [])))
    n_human = n_docs - n_pos
    flat_yt, flat_lab, flat_cnf = [], [], []
    by_ver = {}
    for r in rows:
        gt = r.get("gt_sent_labels", []) or []
        gpt = r.get("gpt", {}) or {}
        labels = gpt.get("sentence_labels", [])
        confs = gpt.get("sentence_confidences", [])
        L = min(len(gt), len(labels), len(confs))
        if L == 0: continue
        flat_yt.extend(gt[:L]); flat_lab.extend(labels[:L]); flat_cnf.extend(confs[:L])
        by_ver.setdefault(r.get("version"), {"yt": [], "lab": [], "cnf": []})
        by_ver[r["version"]]["yt"].extend(gt[:L])
        by_ver[r["version"]]["lab"].extend(labels[:L])
        by_ver[r["version"]]["cnf"].extend(confs[:L])

    def metrics(yt, yp, ys=None):
        n = len(yt)
        if n == 0: return {"n": 0}
        tp = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(yt, yp) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(yt, yp) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 0)
        pos, neg = tp + fn, tn + fp
        pa = tp / (tp + fp) if (tp + fp) else 0.0
        ra = tp / pos if pos else 0.0
        f1a = 2 * pa * ra / (pa + ra) if (pa + ra) else 0.0
        ph = tn / (tn + fn) if (tn + fn) else 0.0
        rh = tn / neg if neg else 0.0
        f1h = 2 * ph * rh / (ph + rh) if (ph + rh) else 0.0
        out = {"n": n, "n_pos_ai": pos, "n_neg_human": neg,
               "accuracy": (tp + tn)/n,
               "precision_ai": pa, "recall_ai": ra, "f1_ai": f1a,
               "precision_human": ph, "recall_human": rh, "f1_human": f1h,
               "fpr": fp/neg if neg else 0.0, "fnr": fn/pos if pos else 0.0,
               "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}
        return out

    summary = {
        "detector": METHOD,
        "dataset": {
            "name": dataset_name_override or "hat_bench_new4d_2026_04_15",
            "field": field,
            "model_short": model_short,
            "csv_path": _safe_relpath(csv_path),
            "split": split,
            "n_docs": n_docs, "n_docs_ai_positive_any": n_pos, "n_docs_human_only": n_human,
            "n_sentences_total": len(flat_yt),
        },
        "protocol": {
            "pipeline_call": f"GemmaSentenceJudge('{METHOD}')",
            "extra_kwargs": None,
            "yaml_config_snapshot": {
                "model": MODEL_NAME, "method": METHOD,
                "do_sample": False, "max_new_tokens": MAX_NEW_TOKENS,
                "enable_thinking": ENABLE_THINKING,
                "prompt_template": "SENTENCE_CONF_PROMPT",
                # Intentionally NO conf_threshold — see top-of-file note about
                # cross-model calibration leak. Per-sentence confidences are still
                # in predictions.jsonl for downstream analysis.
            },
            "gt_label_rule": "per-sentence 0/1 from sent_labels (native HAT-Bench)",
            "input_field": "sentences",
            "thresholds_reported": {"label_threshold": 0.5},
        },
        "domain_of_validity": {
            "language": "English", "granularity": "sentence-level binary",
            "supported": True,
            "caveats": [
                "Greedy decoding (do_sample=False); deterministic.",
                "Reported metrics use the LLM's RAW hard label only. Per-sentence "
                "confidences are saved in predictions.jsonl but no ad-hoc threshold "
                "is applied (avoids cross-model calibration leak).",
            ],
        },
        "metrics_overall": {
            "at_label_threshold_0.5": metrics(flat_yt, flat_lab),
        },
        "metrics_by_version": {
            v: {
                "n_docs": sum(1 for r in rows if r.get("version") == v),
                "at_label_threshold": metrics(d["yt"], d["lab"]),
            }
            for v, d in sorted(by_ver.items())
        },
        "runtime": {
            "total_seconds": round(runtime, 2),
            "n_api_calls": n_docs,
            "calls_per_second": round(n_docs / runtime, 3) if runtime > 0 else None,
            "total_input_tokens": int(total_in),
            "total_output_tokens": int(total_out),
        },
        "errors": {"n_api_errors": n_api_errors, "n_length_mismatch": n_length_mismatch},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (out_dir / "provenance.json").write_text(json.dumps({
        "category": "llm_judge",
        "detector": METHOD,
        "training_free": True,
        "training_data_for_this_eval": None,
        "calibration_on_new_data": None,
        "config": (
            f"Local Gemma 4 E4B-it (8B total / 4B active MoE) loaded from {MODEL_NAME} "
            f"with bf16 on {device}, greedy decoding (do_sample=False), max_new_tokens={MAX_NEW_TOKENS}; "
            f"prompt=SENTENCE_CONF_PROMPT (per-sentence labels + confidences)."
        ),
    }, indent=2) + "\n")
    (out_dir / "run_config.json").write_text(json.dumps({
        "detector": METHOD,
        "field": field, "model_short": model_short, "split": split,
        "device": device,
        "csv_path": _safe_relpath(csv_path),
        "timestamp": ts,
        "yaml_config": {
            "method": METHOD, "model": MODEL_NAME,
            "do_sample": False, "max_new_tokens": MAX_NEW_TOKENS,
            "enable_thinking": ENABLE_THINKING,
            "prompt": "SENTENCE_CONF_PROMPT",
            # No conf_threshold — see runner header note (cross-model calibration leak).
        },
    }, indent=2) + "\n")
    print(f"  ✓ {field}/{model_short}: {n_docs} rows in {runtime:.1f}s, "
          f"errors={n_api_errors}, length_mismatch={n_length_mismatch}")
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", nargs="+", default=["essays", "abstracts", "news", "reports"])
    ap.add_argument("--models", nargs="+", default=["gpt-5.4", "gpt-5.4-nano", "gemini-2.5-flash", "qwen3-8b"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None, help="cap rows per cell (smoke)")
    ap.add_argument("--batch-size", type=int, default=8, help="essays per generate() call")
    ap.add_argument("--csvs", nargs="+", default=None,
                    help="Bypass DATASET_MANIFEST and run on these exact CSV paths.")
    ap.add_argument("--cell-name-template", default=None,
                    help="With --csvs: template using {stem}, {field}, {model_short}. "
                         "Default: '{stem}'.")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="Override output root (default: results/new_data_eval/sentence/per_row/llm_judge).")
    ap.add_argument("--dataset-name", default=None,
                    help="Override summary.json `dataset.name`. Default: 'hat_bench_new4d_2026_04_15'.")
    args = ap.parse_args()

    cells: list[tuple[str, str, Path, str | None]] = []  # (field, model_short, path, cell_name)
    if args.csvs:
        tmpl = args.cell_name_template or "{stem}"
        for cs in args.csvs:
            p = Path(cs)
            if not p.exists():
                print(f"[skip] --csvs path missing: {p}")
                continue
            stem = p.stem
            parts = stem.split("_")
            field = parts[0] if parts else "unknown"
            model_short = parts[-1] if parts else "unknown"
            cell_name = tmpl.format(stem=stem, field=field, model_short=model_short)
            cells.append((field, model_short, p, cell_name))
    else:
        for (f, m), rel in DATASET_MANIFEST.items():
            if f not in args.fields or m not in args.models:
                continue
            path = DATA_DIR / rel
            if not path.exists():
                print(f"[skip] missing CSV: {path}")
                continue
            cells.append((f, m, path, None))
    if not cells:
        print("no cells to run")
        return 1
    print(f"[plan] {len(cells)} cells, device={args.device}, limit={args.limit}")

    tok, model = load_model(args.device)
    for f, m, p, cell_name in cells:
        process_cell(
            f, m, p, args.split, tok, model, args.device, args.limit, args.batch_size,
            out_root_override=args.out_root,
            cell_name_override=cell_name,
            dataset_name_override=args.dataset_name,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
