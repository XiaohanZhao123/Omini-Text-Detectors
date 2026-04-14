"""Reproduce Binoculars detector (ICML 2024) via the omini-text wrapper.

Paper: "Spotting LLMs with Binoculars" (Hans et al., ICML 2024, arxiv 2401.12070)
Upstream: https://github.com/ahans30/Binoculars
Upstream eval script: baseline/binoculars/experiments/run.py

PAPER / UPSTREAM PROTOCOL (extracted from run.py + detector.py):
  - Model pair: tiiuae/falcon-7b (observer) + tiiuae/falcon-7b-instruct (performer)
  - Precision: bfloat16
  - mode="accuracy"  (upstream run.py line 17 — paper Section 4)
  - max_token_observed=512 (tokens_seen default arg)
  - batch_size=32
  - Score formula: perplexity(performer) / cross_entropy(observer, performer)
  - AI = positive class (class=1)
  - Label rule: pred = 1 if score < THRESHOLD else 0 (threshold=0.9015 for accuracy)
  - Metrics: F1 (at thr), AUROC on -score (so positive = high -score), TPR@FPR=0.01%
    AUROC computed on `-score` (so class 1 = positive) via sklearn.metrics.roc_curve
    TPR@FPR=0.01% interpolated with np.interp(0.01/100, fpr, tpr)

DATASETS (baseline/binoculars/datasets/core/*_llama2_13.jsonl):
  - cc_news: 4713 pairs; human_key="text", machine_key="meta-llama-Llama-2-13b-hf_generated_text_wo_prompt"
  - cnn    : 2207 pairs; human_key="article", machine_key="meta-llama-Llama-2-13b-hf_generated_text_wo_prompt"
  - pubmed : 2197 pairs; human_key="article", machine_key="meta-llama-Llama-2-13b-hf_generated_text_wo_prompt"

WRAPPER vs UPSTREAM DEFAULTS:
  omini_text/configs/binoculars.yaml ships with mode="low-fpr" and max_token_observed=2048.
  Upstream's run.py uses mode="accuracy" and tokens_seen=512.
  => to match the paper's protocol exactly we override via kwargs (mode, max_token_observed).
  => FLAG to maintainers: default config diverges from paper protocol; noted in report, not changed here.

Label convention (Omini-Text): 0 = human, 1 = AI.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn import metrics as skmetrics

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from omini_text import pipeline  # noqa: E402


DATA_ROOT = REPO_ROOT / "baseline" / "binoculars" / "datasets" / "core"

DATASETS = [
    {
        "name": "cc_news",
        "pretty": "CC-News",
        "path": DATA_ROOT / "cc_news" / "cc_news-llama2_13.jsonl",
        "human_key": "text",
        "machine_key": "meta-llama-Llama-2-13b-hf_generated_text_wo_prompt",
    },
    {
        "name": "cnn",
        "pretty": "CNN",
        "path": DATA_ROOT / "cnn" / "cnn-llama2_13.jsonl",
        "human_key": "article",
        "machine_key": "meta-llama-Llama-2-13b-hf_generated_text_wo_prompt",
    },
    {
        "name": "pubmed",
        "pretty": "PubMed",
        "path": DATA_ROOT / "pubmed" / "pubmed-llama2_13.jsonl",
        "human_key": "article",
        "machine_key": "meta-llama-Llama-2-13b-hf_generated_text_wo_prompt",
    },
]


def load_dataset(cfg: dict) -> tuple[list[str], list[str]]:
    """Return (human_texts, machine_texts) from a jsonl file."""
    human_texts: list[str] = []
    machine_texts: list[str] = []
    with open(cfg["path"]) as f:
        for line in f:
            obj = json.loads(line)
            h = (obj.get(cfg["human_key"]) or "").strip()
            m = (obj.get(cfg["machine_key"]) or "").strip()
            if h and m:
                human_texts.append(h)
                machine_texts.append(m)
    return human_texts, machine_texts


def score_texts(pipe, texts: list[str], batch_size: int, tag: str) -> list[float]:
    """Score texts through the wrapper. Returns the raw Binoculars score (ppl/x-ppl).

    Note: the wrapper returns `score = -binoculars_score` (polarity flipped so higher=AI).
    Upstream's formula is on the raw binoculars score; we recover it from metadata.
    """
    n = len(texts)
    raw_scores: list[float] = []
    t0 = time.time()
    for i in range(0, n, batch_size):
        batch = texts[i : i + batch_size]
        outs = pipe(batch)
        for r in outs:
            raw_scores.append(float(r["metadata"]["binoculars_score"]))
        if (i // batch_size) % 5 == 0:
            elapsed = time.time() - t0
            done = i + len(batch)
            rate = done / max(elapsed, 1e-6)
            eta = (n - done) / max(rate, 1e-6)
            print(
                f"  [{tag}] {done}/{n} elapsed={elapsed:.1f}s rate={rate:.2f}/s eta={eta:.1f}s"
            )
    return raw_scores


def evaluate(human_scores: list[float], machine_scores: list[float], threshold: float) -> dict:
    """Reproduce upstream's run.py metric computation."""
    y = np.array([0] * len(human_scores) + [1] * len(machine_scores), dtype=np.int64)
    s = np.array(human_scores + machine_scores, dtype=np.float64)
    # Predictions: score < threshold => class 1 (machine)
    preds = (s < threshold).astype(np.int64)
    f1 = skmetrics.f1_score(y, preds)
    # AUROC on negated score (so class 1 = positive class)
    neg_s = -s
    fpr, tpr, _ = skmetrics.roc_curve(y_true=y, y_score=neg_s, pos_label=1)
    roc_auc = skmetrics.auc(fpr, tpr)
    tpr_at_fpr_0_01 = float(np.interp(0.01 / 100, fpr, tpr))
    acc = float((preds == y).mean())
    # per-class accuracy
    acc_h = float((preds[y == 0] == 0).mean())
    acc_m = float((preds[y == 1] == 1).mean())
    return {
        "n_human": int((y == 0).sum()),
        "n_machine": int((y == 1).sum()),
        "threshold": float(threshold),
        "f1": float(f1) * 100,
        "auroc": float(roc_auc) * 100,
        "tpr_at_fpr_0.01%": float(tpr_at_fpr_0_01) * 100,
        "accuracy": acc * 100,
        "accuracy_human": acc_h * 100,
        "accuracy_machine": acc_m * 100,
        "human_score_mean": float(np.mean(human_scores)),
        "machine_score_mean": float(np.mean(machine_scores)),
    }


def main(args: argparse.Namespace) -> None:
    # Paper target numbers for CC-News + CNN + PubMed vs LLaMA-2-13B, mode=accuracy.
    # The paper does NOT publish one consolidated table of F1/AUROC/TPR@0.01%FPR for
    # these three datasets. Targets are reconstructed from:
    #   - Figure 12 (Appendix A.3): AUC per dataset (values read approximately)
    # F1 and TPR@0.01%FPR targets are not listed in the paper per dataset; leave as
    # NaN to avoid fake-precision comparisons. Verdict uses AUROC as the primary.
    PAPER_TARGETS = {
        "cc_news": {"f1": float("nan"), "auroc": 0.99, "tpr@0.01fpr": float("nan")},
        "cnn":     {"f1": float("nan"), "auroc": 0.98, "tpr@0.01fpr": float("nan")},
        "pubmed":  {"f1": float("nan"), "auroc": 0.99, "tpr@0.01fpr": float("nan")},
    }

    # Build pipeline with paper-exact protocol.
    # NOTE: wrapper yaml default is mode=low-fpr / max_token_observed=2048,
    #       upstream run.py uses mode=accuracy / tokens_seen=512. We override.
    print("[binoculars] loading wrapper (mode=accuracy, max_token_observed=512)")
    print(f"             device override = {args.device!r}  (None=wrapper default multi-GPU logic)")
    t_load = time.time()
    kwargs = dict(mode="accuracy", max_token_observed=512)
    if args.device:
        kwargs["device"] = args.device
    pipe = pipeline("ai-text-detection", model="binoculars", **kwargs)
    print(f"  load time: {time.time() - t_load:.1f}s")
    threshold = pipe.detector.detector.threshold
    print(f"  active threshold = {threshold:.6f}  (upstream BINOCULARS_ACCURACY_THRESHOLD)")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = out_root / f"binoculars_falcon7_llama2_13_{stamp}"
    run_dir.mkdir(exist_ok=True)

    all_summary: dict[str, dict] = {}
    for cfg in DATASETS:
        if args.only and cfg["name"] not in args.only:
            continue
        print("\n" + "=" * 72)
        print(f"[binoculars] dataset: {cfg['pretty']}  ({cfg['path'].name})")
        print("=" * 72)

        t_ds = time.time()
        human, machine = load_dataset(cfg)
        # Optional subsample (for debugging only)
        if args.max_samples:
            human = human[: args.max_samples]
            machine = machine[: args.max_samples]
        print(f"  n_human={len(human)}  n_machine={len(machine)}")

        print(f"\n  scoring human text …")
        h_scores = score_texts(pipe, human, args.batch_size, tag=f"{cfg['name']}/human")
        print(f"  scoring machine text …")
        m_scores = score_texts(pipe, machine, args.batch_size, tag=f"{cfg['name']}/machine")

        summary = evaluate(h_scores, m_scores, threshold=threshold)
        summary["dataset"] = cfg["pretty"]
        summary["dataset_path"] = str(cfg["path"])
        summary["runtime_s"] = time.time() - t_ds

        tgt = PAPER_TARGETS.get(cfg["name"], {})
        print(
            "\n  ----- metrics -----\n"
            f"  F1                  = {summary['f1']:6.2f}   (paper {100 * tgt.get('f1', float('nan')):6.2f})\n"
            f"  AUROC               = {summary['auroc']:6.2f}   (paper {100 * tgt.get('auroc', float('nan')):6.2f})\n"
            f"  TPR @ FPR=0.01%     = {summary['tpr_at_fpr_0.01%']:6.2f}   (paper {100 * tgt.get('tpr@0.01fpr', float('nan')):6.2f})\n"
            f"  accuracy @ threshold= {summary['accuracy']:6.2f} "
            f"(H={summary['accuracy_human']:.2f}, M={summary['accuracy_machine']:.2f})\n"
            f"  human score mean    = {summary['human_score_mean']:.4f}\n"
            f"  machine score mean  = {summary['machine_score_mean']:.4f}\n"
            f"  runtime             = {summary['runtime_s']:.1f}s"
        )

        # Save per-dataset predictions
        with open(run_dir / f"{cfg['name']}_scores.jsonl", "w") as f:
            for s, cls in zip(h_scores, [0] * len(h_scores)):
                f.write(json.dumps({"score": s, "class": cls}) + "\n")
            for s, cls in zip(m_scores, [1] * len(m_scores)):
                f.write(json.dumps({"score": s, "class": cls}) + "\n")

        all_summary[cfg["name"]] = summary

    pipe.cleanup()

    final = {
        "detector": "binoculars",
        "wrapper_path": "omini_text/detectors/binoculars_detector.py",
        "config_path": "omini_text/configs/binoculars.yaml",
        "protocol": {
            "source": "baseline/binoculars/experiments/run.py",
            "observer_model": "tiiuae/falcon-7b",
            "performer_model": "tiiuae/falcon-7b-instruct",
            "mode": "accuracy",
            "max_token_observed": 512,
            "batch_size": args.batch_size,
            "threshold": float(threshold),
            "precision": "bfloat16",
        },
        "wrapper_default_mismatch": {
            "yaml_default_mode": "low-fpr",
            "paper_mode": "accuracy",
            "yaml_default_max_tokens": 2048,
            "paper_max_tokens": 512,
            "note": (
                "Wrapper's YAML defaults diverge from upstream run.py. "
                "This reproduction overrides via kwargs to match paper protocol. "
                "Config changes flagged to maintainers, not applied here."
            ),
        },
        "paper_targets": PAPER_TARGETS,
        "results": all_summary,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n[binoculars] summary saved to {run_dir / 'summary.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--device",
        default=None,
        help=(
            "Device spec forwarded to wrapper. None=wrapper picks cuda:0+cuda:1. "
            "Comma-syntax places observer,performer, e.g. 'cuda:0,cuda:1'."
        ),
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_samples", type=int, default=0, help="0 = full dataset")
    p.add_argument(
        "--only", nargs="*", default=None, choices=[d["name"] for d in DATASETS],
        help="Run only a subset of datasets (debug)"
    )
    p.add_argument("--out_dir", default=str(Path(__file__).parent / "results"))
    main(p.parse_args())
