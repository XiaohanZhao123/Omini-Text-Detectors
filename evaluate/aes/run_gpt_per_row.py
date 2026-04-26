"""Run an OpenAI sentence-level detector (e.g. gpt54-sent-conf-none) on
HAT-Bench CSVs with PER-ROW output, organised in HAT-Baselines layout:

  <out_root>/<method>_new4d_<field>_<model_short>_<timestamp>/
      predictions.jsonl    # one line per (essay_id, version) input row
      provenance.json      # detector class info
      run_config.json      # dataset path, timestamp, yaml snapshot
      summary.json         # sentence-level metrics computed at end

One (field, model_short) CSV -> one folder. Resume-aware: if a folder for the
same (method, field, model_short) already exists, append to its
predictions.jsonl instead of creating a new timestamp.

Usage:
    conda run -n omni-text python evaluate/aes/run_gpt_per_row.py \
        --method gpt54-sent-conf-none \
        --data-dir draft/data_25_04_15 \
        --fields essays abstracts news \
        --models gpt-5.4 \
        --split test \
        --out-root results/new_data_eval/sentence/per_row/default_setting \
        [--limit 5]            # smoke test: stop after N rows per folder
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from evaluate.aes.data_loader_unified import get_dataset_files  # noqa: E402
from evaluate.aes.sentence_level_v0v8 import (  # noqa: E402
    GEMINI_SENT_MAP,
    OPENAI_SENT_MAP,
    CLAUDE_SENT_MAP,
    GeminiSentenceDetector,
    OpenAISentenceDetector,
    ClaudeSentenceDetector,
)


def _all_methods() -> list[str]:
    return sorted(list(OPENAI_SENT_MAP.keys())
                  + list(GEMINI_SENT_MAP.keys())
                  + list(CLAUDE_SENT_MAP.keys()))


def _is_gemini_method(method: str) -> bool:
    return method in GEMINI_SENT_MAP


def _is_claude_method(method: str) -> bool:
    return method in CLAUDE_SENT_MAP


def parse_list_str(val: Any) -> list:
    if isinstance(val, list):
        return val
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    s = str(val).strip()
    if not s:
        return []
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return ast.literal_eval(s)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def find_or_create_folder(out_root: Path, method: str, field: str,
                         model_short: str,
                         cell_name: str | None = None) -> tuple[Path, bool]:
    """Return (folder, is_new). Default name = `<method>_new4d_<field>_<model_short>_<ts>`.
    If `cell_name` is given (used by --csvs / ablations) the folder is named
    `<cell_name>_<ts>` directly under `out_root`. In both cases an existing
    matching folder is reused so resume works."""
    prefix = f"{cell_name}_" if cell_name else f"{method}_new4d_{field}_{model_short}_"
    existing = sorted(
        [p for p in out_root.glob(f"{prefix}*") if p.is_dir()],
        key=lambda p: p.name,
    )
    if existing:
        return existing[-1], False
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = out_root / f"{prefix}{ts}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder, True


def load_existing_keys(predictions_path: Path) -> set[tuple]:
    keys: set[tuple] = set()
    if not predictions_path.exists():
        return keys
    with predictions_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add((r.get("essay_id"), r.get("version")))
    return keys


def compute_summary(
    predictions_path: Path,
    detector_method: str,
    model_short: str,
    field: str,
    csv_path: Path,
    split: str,
    runtime_seconds: float,
    label_threshold: float = 0.5,
    conf_threshold: float | None = None,  # None → don't emit calibrated block (cross-model leak fix)
    dataset_name: str = "hat_bench_new4d_2026_04_15",
) -> dict:
    """Compute sentence-level metrics from the predictions.jsonl file."""
    try:
        from sklearn.metrics import roc_auc_score
        has_sklearn = True
    except ImportError:
        has_sklearn = False

    rows = []
    n_api_errors = 0
    n_length_mismatch = 0
    total_input_tokens = 0
    total_output_tokens = 0

    with predictions_path.open() as f:
        for line in f:
            rows.append(json.loads(line))

    def flat_metrics(y_true: list[int], y_pred: list[int],
                     y_score: list[float] | None) -> dict:
        n = len(y_true)
        if n == 0:
            return {"n": 0}
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        pos = tp + fn
        neg = tn + fp
        acc = (tp + tn) / n
        prec_ai = tp / (tp + fp) if (tp + fp) else 0.0
        rec_ai = tp / pos if pos else 0.0
        f1_ai = 2 * prec_ai * rec_ai / (prec_ai + rec_ai) if (prec_ai + rec_ai) else 0.0
        prec_h = tn / (tn + fn) if (tn + fn) else 0.0
        rec_h = tn / neg if neg else 0.0
        f1_h = 2 * prec_h * rec_h / (prec_h + rec_h) if (prec_h + rec_h) else 0.0
        out = {
            "n": n,
            "n_pos_ai": pos,
            "n_neg_human": neg,
            "accuracy": acc,
            "precision_ai": prec_ai,
            "recall_ai": rec_ai,
            "f1_ai": f1_ai,
            "precision_human": prec_h,
            "recall_human": rec_h,
            "f1_human": f1_h,
            "fpr": fp / neg if neg else 0.0,
            "fnr": fn / pos if pos else 0.0,
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        }
        if has_sklearn and y_score is not None and pos > 0 and neg > 0:
            try:
                out["auroc"] = float(roc_auc_score(y_true, y_score))
            except ValueError:
                out["auroc"] = None
        return out

    # Flatten to sentence level
    y_true_all: list[int] = []
    y_pred_label_all: list[int] = []  # at threshold 0.5 on label binary
    y_score_conf_all: list[float] = []
    per_version: dict[str, dict] = {}
    n_docs = len(rows)
    n_ai_pos_doc = 0
    n_human_neg_doc = 0

    for r in rows:
        gpt = r.get("gpt") or {}
        if gpt.get("error"):
            n_api_errors += 1
        if gpt.get("length_mismatch"):
            n_length_mismatch += 1
        usage = gpt.get("usage") or {}
        total_input_tokens += int(usage.get("input_tokens", 0) or 0)
        total_output_tokens += int(usage.get("output_tokens", 0) or 0)

        gt = r.get("gt_sent_labels") or []
        labels = gpt.get("sentence_labels") or []
        confs = gpt.get("sentence_confidences") or [0.5] * len(gt)
        # align lengths defensively
        L = min(len(gt), len(labels), len(confs))
        gt, labels, confs = gt[:L], labels[:L], confs[:L]

        y_true_all.extend(gt)
        y_pred_label_all.extend(labels)
        y_score_conf_all.extend(confs)

        ver = r.get("version") or "?"
        if ver not in per_version:
            per_version[ver] = {"y_true": [], "y_pred": [], "y_score": []}
        per_version[ver]["y_true"].extend(gt)
        per_version[ver]["y_pred"].extend(labels)
        per_version[ver]["y_score"].extend(confs)

        ai_ratio = r.get("AI_sent_ratio")
        if isinstance(ai_ratio, (int, float)):
            if ai_ratio > 0:
                n_ai_pos_doc += 1
            else:
                n_human_neg_doc += 1

    metrics_overall_label = flat_metrics(
        y_true_all, y_pred_label_all, y_score_conf_all
    )
    # Conditional: emit the calibrated block only if conf_threshold was set explicitly.
    metrics_overall_conf = None
    if conf_threshold is not None:
        y_pred_conf_all = [1 if c >= conf_threshold else 0 for c in y_score_conf_all]
        metrics_overall_conf = flat_metrics(
            y_true_all, y_pred_conf_all, y_score_conf_all
        )

    metrics_by_version = {}
    for ver, d in sorted(per_version.items()):
        entry = {
            "n_docs": sum(1 for r in rows if r.get("version") == ver),
            "at_label_threshold": flat_metrics(d["y_true"], d["y_pred"], d["y_score"]),
        }
        if conf_threshold is not None:
            y_pred_conf = [1 if c >= conf_threshold else 0 for c in d["y_score"]]
            entry[f"at_conf>=_{conf_threshold}"] = flat_metrics(d["y_true"], y_pred_conf, d["y_score"])
        metrics_by_version[ver] = entry

    summary = {
        "detector": detector_method,
        "dataset": {
            "name": dataset_name,
            "field": field,
            "model_short": model_short,
            "csv_path": str(csv_path),
            "split": split,
            "n_docs": n_docs,
            "n_docs_ai_positive_any": n_ai_pos_doc,
            "n_docs_human_only": n_human_neg_doc,
            "n_sentences_total": len(y_true_all),
            "n_sentences_ai": sum(y_true_all),
            "n_sentences_human": len(y_true_all) - sum(y_true_all),
        },
        "protocol": {
            "pipeline_call": (
                f"ClaudeSentenceDetector('{detector_method}')"
                if detector_method in CLAUDE_SENT_MAP else
                f"GeminiSentenceDetector('{detector_method}')"
                if detector_method in GEMINI_SENT_MAP else
                f"OpenAISentenceDetector('{detector_method}')"
            ),
            "extra_kwargs": None,
            "yaml_config_snapshot": (
                {
                    "model": CLAUDE_SENT_MAP[detector_method][0],
                    "thinking_enabled": CLAUDE_SENT_MAP[detector_method][1],
                    "temperature": 0.0,
                    "max_tokens": 2048,
                    "prompt_template": "SENTENCE_CONF_PROMPT",
                    "method": detector_method,
                }
                if detector_method in CLAUDE_SENT_MAP else
                {
                    "model": GEMINI_SENT_MAP[detector_method][0],
                    "thinking_level": GEMINI_SENT_MAP[detector_method][1],
                    "prompt_template": "SENTENCE_CONF_PROMPT",
                    "method": detector_method,
                }
                if detector_method in GEMINI_SENT_MAP else
                {
                    "model": OPENAI_SENT_MAP[detector_method][0],
                    "reasoning_effort": OPENAI_SENT_MAP[detector_method][1],
                    "prompt_template": "SENTENCE_CONF_PROMPT",
                    "max_completion_tokens": 2048,
                    "method": detector_method,
                }
            ),
            "gt_label_rule": "per-sentence 0/1 from sent_labels column (native HAT-Bench annotation)",
            "input_field": "sentences",
            "thresholds_reported": {
                "label_threshold": label_threshold,
                "conf_threshold": conf_threshold,
            },
        },
        "domain_of_validity": {
            "language": "English",
            "granularity": "sentence-level binary",
            "supported": True,
            "caveats": (
                [f"Additional calibrated readout at conf >= {conf_threshold} (judge-specific; tuned on held-out slice)."]
                if conf_threshold is not None else
                ["Hard-label only: we report the LLM's raw sentence_labels; "
                 "per-sentence confidences are in predictions.jsonl but no "
                 "cross-model conf_threshold is applied (avoids calibration leak)."]
            ),
        },
        "metrics_overall": (
            {
                "at_label_threshold_0.5": metrics_overall_label,
                f"at_conf>=_{conf_threshold}": metrics_overall_conf,
            }
            if conf_threshold is not None else
            {"at_label_threshold_0.5": metrics_overall_label}
        ),
        "metrics_by_version": metrics_by_version,
        "runtime": {
            "total_seconds": round(runtime_seconds, 2),
            "n_api_calls": n_docs,
            "calls_per_second": round(n_docs / runtime_seconds, 3) if runtime_seconds > 0 else None,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
        },
        "errors": {
            "n_api_errors": n_api_errors,
            "n_length_mismatch": n_length_mismatch,
        },
        "git_commit": git_commit(),
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    help=f"One of: {_all_methods()}")
    ap.add_argument("--data-dir", default="draft/data_25_04_15")
    ap.add_argument("--fields", nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--split", default="test",
                    choices=["train", "dev", "test"])
    ap.add_argument("--out-root", type=Path,
                    default=Path("results/new_data_eval/sentence/per_row/default_setting"))
    ap.add_argument("--limit", type=int, default=None,
                    help="Smoke test: stop after N new rows per folder.")
    ap.add_argument("--conf-threshold", type=float, default=None,
                    help=("Optional confidence threshold to ALSO report. DEFAULT IS None "
                          "— i.e. NO calibrated block in summary.json. Only set this if "
                          "you have calibrated the threshold for THIS judge on a held-out "
                          "slice; never reuse another judge's value (cross-model leak)."))
    ap.add_argument("--workers", type=int, default=1,
                    help="Concurrent API workers per cell (default 1 = sequential).")
    ap.add_argument("--csvs", nargs="+", default=None,
                    help="Bypass --fields/--models manifest matcher and run on these "
                         "exact CSV paths. Used by ablation runs.")
    ap.add_argument("--cell-name-template", default=None,
                    help="When --csvs is used, name each cell folder with this "
                         "template applied to the CSV stem. {stem} is replaced. "
                         "Default = '{stem}'.")
    ap.add_argument("--dataset-name", default=None,
                    help="Override summary.json `dataset.name`. Default = "
                         "'hat_bench_new4d_2026_04_15'. Use a unique name for "
                         "ablation runs (e.g. 'ablation1_covctrl').")
    args = ap.parse_args()

    if (args.method not in OPENAI_SENT_MAP
        and args.method not in GEMINI_SENT_MAP
        and args.method not in CLAUDE_SENT_MAP):
        print(f"ERROR: unknown method {args.method!r}. "
              f"Available: {_all_methods()}")
        return 2

    if args.csvs:
        # Ablation mode: each CSV is one cell. Use the CSV stem to derive a
        # cell name; field/model_short are best-effort metadata only.
        files = []
        for cs in args.csvs:
            p = Path(cs)
            if not p.exists():
                print(f"ERROR: --csvs path does not exist: {p}")
                return 2
            stem = p.stem
            # Heuristic field/model from stem (e.g. abstracts_covctrl_compress_gemini-2.5-flash):
            parts = stem.split("_")
            field = parts[0] if parts else "unknown"
            model_short = parts[-1] if parts else "unknown"
            files.append((field, model_short, str(p)))
    else:
        files = get_dataset_files(
            data_dir=args.data_dir,
            fields=args.fields,
            models=args.models,
        )
        if not files:
            print(f"ERROR: no CSV files matched fields={args.fields} "
                  f"models={args.models} under {args.data_dir}")
            return 2

    is_gemini = _is_gemini_method(args.method)
    is_claude = _is_claude_method(args.method)
    if is_gemini:
        model_name_api, reasoning = GEMINI_SENT_MAP[args.method]
        api_provider = "gemini"
    elif is_claude:
        model_name_api, reasoning = CLAUDE_SENT_MAP[args.method]
        api_provider = "anthropic"
    else:
        model_name_api, reasoning = OPENAI_SENT_MAP[args.method]
        api_provider = "openai"
    print(f"[config] method={args.method} (model={model_name_api}, reasoning={reasoning}, provider={api_provider})")
    print(f"[config] data_dir={args.data_dir} split={args.split} "
          f"fields={args.fields} models={args.models}")
    print(f"[config] out_root={args.out_root} limit={args.limit}")
    print(f"[files] matched {len(files)}:")
    for field, model_short, csv_path in files:
        print(f"  - {field} / {model_short} -> {csv_path}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    if is_gemini:
        detector = GeminiSentenceDetector(args.method)
    elif is_claude:
        detector = ClaudeSentenceDetector(args.method)
    else:
        detector = OpenAISentenceDetector(args.method)

    for field, model_short, csv_path in files:
        cell_name = None
        if args.csvs:
            stem = Path(csv_path).stem
            tmpl = args.cell_name_template or "{stem}"
            cell_name = tmpl.format(stem=stem, method=args.method,
                                    field=field, model_short=model_short)
        folder, is_new = find_or_create_folder(
            args.out_root, args.method, field, model_short, cell_name=cell_name,
        )
        predictions_path = folder / "predictions.jsonl"
        print(f"\n[folder] {folder}  (new={is_new})")

        # provenance.json (write/overwrite)
        # category = the parent directory name of out_root (e.g. "llm_judge",
        # "default_setting", "tuned_on_new_data") so the metadata matches the
        # location the user uploads to.
        provenance = {
            "category": args.out_root.name or "default_setting",
            "detector": args.method,
            "training_free": True,
            "training_data_for_this_eval": None,
            "calibration_on_new_data": None,
            "config": (
                (
                    f"Anthropic Claude {model_name_api} with extended thinking DISABLED "
                    f"(thinking_enabled={reasoning}), temperature=0, max_tokens=2048; "
                    f"prompt=SENTENCE_CONF_PROMPT (per-sentence labels + confidences)."
                )
                if is_claude else
                (
                    f"Gemini {model_name_api} with thinking_level={reasoning!r}; "
                    f"prompt=SENTENCE_CONF_PROMPT (per-sentence labels + confidences); "
                    f"structured JSON output; sequential sync calls."
                )
                if is_gemini else
                (
                    f"OpenAI {model_name_api} with reasoning_effort={reasoning!r}; "
                    f"prompt=SENTENCE_CONF_PROMPT (per-sentence labels + confidences); "
                    f"max_completion_tokens=2048; sequential sync calls."
                )
            ),
        }
        (folder / "provenance.json").write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False) + "\n"
        )

        # run_config.json (write/overwrite with current timestamp tag)
        run_config = {
            "detector": args.method,
            "field": field,
            "model_short": model_short,
            "split": args.split,
            "device": ("anthropic-api" if is_claude else
                       "gemini-api" if is_gemini else "openai-api"),
            "max_samples": args.limit,
            "csv_path": str(csv_path),
            "timestamp": folder.name.rsplit("_", 2)[-2] + "_" + folder.name.rsplit("_", 2)[-1],
            "git_commit": git_commit(),
            "yaml_config": (
                {
                    "method": args.method, "model": model_name_api,
                    "thinking_enabled": reasoning,
                    "temperature": 0.0, "max_tokens": 2048,
                    "prompt": "SENTENCE_CONF_PROMPT",
                }
                if is_claude else
                {
                    "method": args.method, "model": model_name_api,
                    "thinking_level": reasoning,
                    "prompt": "SENTENCE_CONF_PROMPT",
                }
                if is_gemini else
                {
                    "method": args.method, "model": model_name_api,
                    "reasoning_effort": reasoning,
                    "prompt": "SENTENCE_CONF_PROMPT",
                    "max_completion_tokens": 2048,
                }
            ),
        }
        (folder / "run_config.json").write_text(
            json.dumps(run_config, indent=2, ensure_ascii=False) + "\n"
        )

        # Load CSV
        df = pd.read_csv(csv_path)
        if "essay_id" not in df.columns and "id" in df.columns:
            df = df.rename(columns={"id": "essay_id"})
        df = df[df["split"] == args.split].reset_index(drop=True)
        print(f"[{field}/{model_short}] {len(df)} rows in split={args.split}")

        existing = load_existing_keys(predictions_path)
        if existing:
            print(f"  [resume] {len(existing)} rows already scored; skipping those")

        new_rows = 0
        t_folder = time.time()

        # Build the to-do list of rows (resume-aware + --limit aware).
        todo = []
        for _, row in df.iterrows():
            key = (row["essay_id"], row["version"])
            if key in existing:
                continue
            if args.limit is not None and len(todo) >= args.limit:
                break
            todo.append(row)

        write_lock = threading.Lock()
        df_columns = list(df.columns)

        def _process(row):
            sentences_list = parse_list_str(row["sentences"])
            sentences_dicts = [{"text": s} for s in sentences_list]
            gt_sent_labels = parse_list_str(row["sent_labels"])
            try:
                pred = detector.predict(sentences_dicts)
                pred["error"] = None
            except Exception as e:
                pred = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "sentence_labels": [0] * len(sentences_list),
                    "model": detector.model_name,
                    "reasoning_effort": detector.reasoning_effort,
                    "variant": args.method,
                }
            out_row: dict = {}
            for col in df_columns:
                v = row[col]
                if not isinstance(v, (list, dict)) and pd.isna(v):
                    out_row[col] = None
                elif hasattr(v, "item"):
                    out_row[col] = v.item()
                else:
                    out_row[col] = v
            out_row["_source"] = {
                "csv_path": str(csv_path),
                "field": field,
                "model_short": model_short,
                "split": args.split,
            }
            out_row["gt_sent_labels"] = gt_sent_labels
            out_row["num_sentences_parsed"] = len(sentences_list)
            out_row["gpt"] = pred
            return out_row

        n_workers = max(1, int(args.workers))
        with predictions_path.open("a") as f_out:
            if n_workers == 1:
                for row in tqdm(todo, total=len(todo),
                                desc=f"{field}/{model_short}"):
                    out_row = _process(row)
                    f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    f_out.flush()
                    new_rows += 1
            else:
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    futures = [ex.submit(_process, row) for row in todo]
                    for fut in tqdm(as_completed(futures), total=len(futures),
                                    desc=f"{field}/{model_short}"):
                        out_row = fut.result()
                        with write_lock:
                            f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                            f_out.flush()
                            new_rows += 1

        folder_elapsed = time.time() - t_folder
        print(f"  [{field}/{model_short}] wrote {new_rows} new rows in "
              f"{folder_elapsed:.1f}s -> {predictions_path}")

        # summary.json — always recompute from the full predictions.jsonl
        try:
            summary = compute_summary(
                predictions_path=predictions_path,
                detector_method=args.method,
                model_short=model_short,
                field=field,
                csv_path=csv_path,
                split=args.split,
                runtime_seconds=folder_elapsed,
                conf_threshold=args.conf_threshold,
                dataset_name=args.dataset_name or "hat_bench_new4d_2026_04_15",
            )
            (folder / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
            )
            print(f"  [summary] wrote {folder / 'summary.json'}")
        except Exception as e:
            print(f"  [summary] WARNING: failed to compute summary: "
                  f"{type(e).__name__}: {e}")

    print("\n[done] all folders processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
