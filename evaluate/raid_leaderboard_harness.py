"""RAID leaderboard-compatible evaluation harness.

Reproduces the official RAID benchmark metric (raid-bench.xyz leaderboard):

    Accuracy @ FPR=5 %
    ------------------
    1. Run the detector on the full eval set (human + AI rows).
    2. For each domain d, find threshold t_d on HUMAN rows such that
       FPR(human_d, t_d) ≈ 5 %.
    3. Accuracy on AI rows = fraction of AI rows with score ≥ t_d (per-domain).
       All AI rows count as positives — accuracy @ FPR=5% == TPR @ FPR=5%.

This mirrors exactly the reference implementation at
https://github.com/liamdugan/raid/blob/main/raid/evaluate.py
(`find_threshold`, `compute_scores`).

Design decisions (documented so the numbers are reproducible later):
  * **Dataset**: HF `Shengkun/Raid_split` (already cached). It is a third-party
    repackage of `liamdugan/raid` that preserves the schema (id, model, attack,
    domain, decoding, repetition_penalty, generation, …) and — crucially —
    keeps labels on the test split, which the canonical `liamdugan/raid:test`
    does NOT (leaderboard is blind). Splits used:
        - `test` (112 K rows, attack="none") → "no-attack" leaderboard metric
        - `test_attack` (103 K rows) → "with-attack" leaderboard metric
  * **Calibration source**: the human rows of the evaluation split itself.
    This is what the official `raid/evaluate.py::find_threshold` does (it
    filters `df[df.model == "human"]` from the same dataframe used to compute
    accuracy). The RAID paper (§5) confirms: "We tune the threshold … to yield
    a 5 % false positive rate … on the human-written portion of the dataset."
  * **Per-domain tuning**: on by default (`per_domain_tuning=True`, the
    leaderboard default). Each of the 8 domains gets its own threshold.
  * **Positive class**: AI (leaderboard default; y_true=1 for all non-human).
  * **Score semantics**: wrapper's `score` is P(AI), so higher score ⇒ more
    AI-like. Directly compatible with the leaderboard's "score ≥ t ⇒ AI".
  * **Adversarial splits**: evaluated separately. The leaderboard publishes
    two top-line numbers per detector: "No-attack" and "With-attack". We
    replicate both. (With-attack still calibrates its threshold on its own
    human rows, but `Shengkun/Raid_split:test_attack` has no human=none rows —
    so we calibrate with-attack on `test` human rows, documented below.)
  * **Subsampling**: full test split is 112 K; scoring all of it with
    DeBERTa-v3-large (desklib) at batch=4 takes ~3 h. We default to a
    stratified subsample: keep ALL human rows (for accurate threshold calib)
    + subsample AI rows to `--n_ai` (default 10 K, stratified by domain).
    Pass `--n_ai 0` to disable subsampling and score everything.

Run:
    CUDA_VISIBLE_DEVICES=4 conda run -n omni-text python \
        evaluate/raid_leaderboard_harness.py --detector e5-small

    CUDA_VISIBLE_DEVICES=4 conda run -n omni-text python \
        evaluate/raid_leaderboard_harness.py --detector desklib --batch_size 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from omini_text import pipeline  # noqa: E402


# ---------- Official RAID threshold search (copied from raid/evaluate.py) ----


def _compute_fpr(y_scores: Sequence[float], threshold: float) -> float:
    """FPR of treating score ≥ threshold as positive on human rows."""
    y_pred = [1 if y >= threshold else 0 for y in y_scores]
    y_true = [0] * len(y_pred)
    return 1 - accuracy_score(y_true, y_pred)


def _find_threshold(
    human_scores: np.ndarray, target_fpr: float, epsilon: float
) -> tuple[float, float]:
    """Iterative bisection matching raid/evaluate.py::find_threshold."""
    y_scores = human_scores.tolist()
    if not y_scores:
        raise ValueError("no human scores available for this split")

    def sign(x: float) -> int:
        return -1 if x < 0 else 1

    threshold = float(sum(y_scores) / len(y_scores))  # init: mean human score
    step_size = 0.5
    prev_dist: float | None = None
    iteration = 1
    found_list: list[tuple[float, float]] = []

    fpr = _compute_fpr(y_scores, threshold)
    while abs(fpr - target_fpr) > epsilon:
        found_list.append((threshold, fpr))

        iteration += 1
        dist = target_fpr - fpr

        if prev_dist is not None and sign(dist) != sign(prev_dist):
            step_size *= -0.5
        elif prev_dist is not None and abs(dist) - abs(prev_dist) > 0.01:
            step_size *= -1

        threshold += step_size
        prev_dist = dist

        if iteration > 50:
            diffs = [
                (target_fpr - f, t) for t, f in found_list if f > 0.0
            ]
            pos_diffs = [(d, t) for d, t in diffs if d >= 0]
            if pos_diffs:
                threshold = min(pos_diffs)[1]
            else:
                threshold = max(diffs)[1]
            break

        fpr = _compute_fpr(y_scores, threshold)

    return float(threshold), _compute_fpr(y_scores, threshold)


# ---------- Data loading ----------------------------------------------------


def load_raid_split(split: str, attack_filter: str | None = None) -> pd.DataFrame:
    """Load `Shengkun/Raid_split:<split>` as a DataFrame with needed columns.

    `attack_filter`:
        None        — return everything (leaderboard's `all` aggregate).
        "none"      — only attack=="none" rows (leaderboard's `no_adversarial`).
        "non_none"  — only attack!="none" rows (strict adversarial-only subset).
    """
    print(f"[data] loading Shengkun/Raid_split split={split} attack_filter={attack_filter}")
    ds = load_dataset("Shengkun/Raid_split", split=split)
    cols = ["id", "model", "attack", "domain", "generation"]
    df = pd.DataFrame({c: ds[c] for c in cols})
    df["generation"] = df["generation"].fillna("").astype(str)
    df = df[df["generation"].str.strip().astype(bool)].reset_index(drop=True)
    if attack_filter == "none":
        df = df[df["attack"] == "none"].reset_index(drop=True)
    elif attack_filter == "non_none":
        df = df[df["attack"] != "none"].reset_index(drop=True)
    print(
        f"  loaded {len(df)} rows after filter. "
        f"unique domains={sorted(df['domain'].unique())}; "
        f"human rows={int((df['model'] == 'human').sum())}; "
        f"ai rows={int((df['model'] != 'human').sum())}"
    )
    return df


def subsample(
    df: pd.DataFrame, n_ai: int, seed: int
) -> pd.DataFrame:
    """Keep all humans; stratify-sample AI to n_ai (per-domain proportional)."""
    if n_ai <= 0 or n_ai >= (df["model"] != "human").sum():
        return df
    rng = np.random.default_rng(seed)
    human = df[df["model"] == "human"].copy()
    ai = df[df["model"] != "human"].copy()

    # per-domain proportional sampling
    out_parts = []
    per_domain_total = ai.groupby("domain").size()
    for dom, n_dom in per_domain_total.items():
        n_take = int(round(n_ai * n_dom / per_domain_total.sum()))
        sub = ai[ai["domain"] == dom]
        if n_take >= len(sub):
            out_parts.append(sub)
        else:
            idx = rng.choice(len(sub), size=n_take, replace=False)
            out_parts.append(sub.iloc[idx])

    ai_sub = pd.concat(out_parts, ignore_index=True)
    out = pd.concat([human, ai_sub], ignore_index=True)
    print(
        f"  subsampled: human={len(human)}, ai={len(ai_sub)} (from {len(ai)}), "
        f"total={len(out)}, seed={seed}"
    )
    return out


# ---------- Scoring --------------------------------------------------------


def score_dataframe(
    df: pd.DataFrame, detector_name: str, batch_size: int, gpu: int
) -> tuple[np.ndarray, float]:
    """Returns (scores_P_AI, runtime_seconds)."""
    print(f"[score] loading detector '{detector_name}' on cuda:{gpu}")
    pipe = pipeline(
        "ai-text-detection", model=detector_name, device=f"cuda:{gpu}"
    )
    texts = df["generation"].tolist()
    n = len(texts)
    print(f"[score] scoring {n} texts, batch={batch_size}")

    scores = np.zeros(n, dtype=np.float64)
    t0 = time.time()
    last_log = t0
    for i in range(0, n, batch_size):
        batch = texts[i : i + batch_size]
        outs = pipe(batch)
        for j, o in enumerate(outs):
            scores[i + j] = float(o["score"])
        now = time.time()
        if now - last_log > 30 or i == 0:
            elapsed = now - t0
            rate = (i + len(batch)) / max(elapsed, 1e-6)
            eta = (n - i - len(batch)) / max(rate, 1e-6)
            print(
                f"  [{i + len(batch)}/{n}] elapsed={elapsed:.0f}s "
                f"rate={rate:.1f}/s eta={eta:.0f}s"
            )
            last_log = now

    elapsed = time.time() - t0
    print(f"[score] done in {elapsed:.1f}s ({n / elapsed:.1f} samples/s)")
    pipe.cleanup()
    return scores, elapsed


# ---------- Evaluation -----------------------------------------------------


def evaluate_split(
    df: pd.DataFrame,
    scores: np.ndarray,
    target_fpr: float,
    epsilon: float,
    calibration_df: pd.DataFrame | None = None,
    calibration_scores: np.ndarray | None = None,
) -> dict:
    """Run official RAID protocol on a scored dataframe.

    Parameters
    ----------
    df, scores : eval frame + its P(AI) scores. Same length.
    calibration_df, calibration_scores : alternative human source for threshold
        search. If None, uses df's own human rows (standard leaderboard protocol).
    """
    df = df.reset_index(drop=True).copy()
    df["score"] = scores

    calib_df = calibration_df if calibration_df is not None else df
    calib_scores = (
        calibration_scores if calibration_scores is not None else scores
    )
    calib_df = calib_df.reset_index(drop=True).copy()
    calib_df["score"] = calib_scores

    domains = sorted(df["domain"].unique())
    thresholds: dict[str, float] = {}
    true_fprs: dict[str, float] = {}

    for dom in domains:
        h = calib_df[(calib_df["model"] == "human") & (calib_df["domain"] == dom)]
        if len(h) == 0:
            # fall back to global human if this domain has no calibration humans
            h = calib_df[calib_df["model"] == "human"]
            if len(h) == 0:
                raise ValueError(
                    f"no human rows in calibration set for domain={dom}"
                )
        t, tf = _find_threshold(h["score"].to_numpy(), target_fpr, epsilon)
        thresholds[dom] = t
        true_fprs[dom] = tf
        print(
            f"  domain={dom:10s} n_human={len(h):5d} "
            f"thr={t:+.4f} true_fpr={tf:.4f}"
        )

    # Per-domain accuracy on AI rows
    ai_df = df[df["model"] != "human"].copy()
    hu_df = df[df["model"] == "human"].copy()
    preds = np.zeros(len(ai_df), dtype=np.int64)
    for i, (dom, sc) in enumerate(zip(ai_df["domain"], ai_df["score"])):
        preds[i] = 1 if sc >= thresholds[dom] else 0

    ai_acc = float(preds.mean())  # TPR @ FPR=5%
    n_ai = len(ai_df)
    n_hu = len(hu_df)

    # Human FPR actually achieved over entire eval frame
    hu_pred = np.array(
        [
            1 if sc >= thresholds[dom] else 0
            for dom, sc in zip(hu_df["domain"], hu_df["score"])
        ],
        dtype=np.int64,
    )
    actual_fpr = float(hu_pred.mean()) if n_hu else float("nan")

    # AUROC / AUPR over the whole split (combines human + AI)
    y_true = np.concatenate(
        [np.ones(n_ai, dtype=np.int64), np.zeros(n_hu, dtype=np.int64)]
    )
    y_score = np.concatenate(
        [ai_df["score"].to_numpy(), hu_df["score"].to_numpy()]
    )
    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auroc = float("nan")
    try:
        aupr = float(average_precision_score(y_true, y_score))
    except ValueError:
        aupr = float("nan")

    # TPR@FPR=5% via sklearn ROC curve (cross-check, not per-domain)
    try:
        fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)
        tpr_at_fpr5 = float(np.interp(target_fpr, fpr_arr, tpr_arr))
    except ValueError:
        tpr_at_fpr5 = float("nan")

    # Confusion matrix using per-domain thresholds
    combined_pred = np.zeros(len(df), dtype=np.int64)
    for i, (dom, sc) in enumerate(zip(df["domain"], df["score"])):
        combined_pred[i] = 1 if sc >= thresholds[dom] else 0
    combined_true = (df["model"] != "human").astype(int).to_numpy()
    cm = confusion_matrix(combined_true, combined_pred, labels=[0, 1])

    # Per-domain accuracy breakdown
    per_domain = {}
    for dom in domains:
        mask = ai_df["domain"] == dom
        per_domain[dom] = {
            "n_ai": int(mask.sum()),
            "accuracy_at_fpr5": float(preds[mask.to_numpy()].mean())
            if mask.sum() > 0
            else None,
            "threshold": thresholds[dom],
            "calib_fpr": true_fprs[dom],
        }

    return {
        "target_fpr": target_fpr,
        "acc_at_fpr5": ai_acc,
        "actual_fpr": actual_fpr,
        "tpr_at_fpr5_global": tpr_at_fpr5,
        "auroc": auroc,
        "aupr": aupr,
        "n_ai": int(n_ai),
        "n_human": int(n_hu),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "per_domain": per_domain,
        "thresholds": thresholds,
    }


# ---------- Driver ---------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    # ---- Load both splits up front -----------------------------------------
    # `Shengkun/Raid_split:test` is a mixed-attack subset of the canonical
    # RAID test. Leaderboard's `no_adversarial` is strictly attack=="none"; we
    # filter to match. Leaderboard's `all` aggregates no-attack + adversarial;
    # `test_attack` split is adversarial-only (no attack=="none" rows) and we
    # use it plus the attack=="none" rows from `test` to form the `all`
    # aggregate below.
    df_noattack = load_raid_split("test", attack_filter="none")
    df_attack = load_raid_split("test_attack")  # already adversarial-only

    # Subsample each (keep all humans)
    df_noattack = subsample(df_noattack, args.n_ai, args.seed)
    df_attack = subsample(df_attack, args.n_ai, args.seed)

    # ---- Score everything in one pipeline load -----------------------------
    # Concatenate, track indices so we can split back out.
    df_noattack["_split"] = "no_attack"
    df_attack["_split"] = "with_attack"
    full_df = pd.concat([df_noattack, df_attack], ignore_index=True)
    print(f"\n[score] total rows to score: {len(full_df)}")

    scores, runtime_s = score_dataframe(
        full_df, args.detector, args.batch_size, args.gpu
    )
    full_df["score"] = scores

    noattack_df = full_df[full_df["_split"] == "no_attack"].reset_index(drop=True)
    noattack_scores = noattack_df["score"].to_numpy()
    attack_df = full_df[full_df["_split"] == "with_attack"].reset_index(drop=True)
    attack_scores = attack_df["score"].to_numpy()

    # ---- Evaluate ----------------------------------------------------------
    print("\n===== No-attack split =====")
    noattack_eval = evaluate_split(
        noattack_df,
        noattack_scores,
        target_fpr=args.target_fpr,
        epsilon=args.epsilon,
    )
    print(
        f"  Acc@FPR=5% = {100 * noattack_eval['acc_at_fpr5']:.2f} | "
        f"AUROC = {100 * noattack_eval['auroc']:.2f} | "
        f"AUPR = {100 * noattack_eval['aupr']:.2f} | "
        f"actual FPR = {100 * noattack_eval['actual_fpr']:.2f}"
    )

    print("\n===== With-attack split =====")
    # test_attack has no attack='none'+model='human' rows to calibrate on;
    # leaderboard protocol: share calibration with no-attack's human set.
    attack_eval = evaluate_split(
        attack_df,
        attack_scores,
        target_fpr=args.target_fpr,
        epsilon=args.epsilon,
        calibration_df=noattack_df,
        calibration_scores=noattack_scores,
    )
    print(
        f"  Acc@FPR=5% = {100 * attack_eval['acc_at_fpr5']:.2f} | "
        f"AUROC = {100 * attack_eval['auroc']:.2f} | "
        f"AUPR = {100 * attack_eval['aupr']:.2f} | "
        f"actual FPR = {100 * attack_eval['actual_fpr']:.2f}"
    )

    # ---- Persist -----------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = out_dir / f"raid_lb_{args.detector}_{stamp}"
    run_dir.mkdir()

    # Per-sample predictions
    with open(run_dir / "predictions.jsonl", "w") as f:
        for _, row in full_df.iterrows():
            f.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "split": row["_split"],
                        "model": row["model"],
                        "domain": row["domain"],
                        "attack": row["attack"],
                        "score_p_ai": float(row["score"]),
                    }
                )
                + "\n"
            )

    summary = {
        "detector": args.detector,
        "dataset": "Shengkun/Raid_split",
        "protocol": "raid-bench.xyz official (per-domain threshold @ FPR=5%)",
        "splits": {"no_attack": "test", "with_attack": "test_attack"},
        "positive_class": "AI (non-human model output)",
        "calibration_source": (
            "human rows of each split (no-attack uses its own; with-attack "
            "reuses no-attack humans because test_attack has no human=attack-free rows)"
        ),
        "per_domain_tuning": True,
        "target_fpr": args.target_fpr,
        "epsilon": args.epsilon,
        "seed": args.seed,
        "n_ai_subsample_per_split": args.n_ai,
        "batch_size": args.batch_size,
        "gpu": args.gpu,
        "runtime_s": runtime_s,
        "no_attack": noattack_eval,
        "with_attack": attack_eval,
    }

    # Add comparison hooks (filled by docs rather than code)
    summary["leaderboard_comparison"] = {
        "url": "https://raid-bench.xyz/leaderboard",
        "note": (
            "E5-small (MayZhou) HF-card target: 93.9 % acc@FPR=5% no-attack. "
            "Desklib v1.01 HF-card target: 'top submission'; see leaderboard."
        ),
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[done] saved to {run_dir}")
    print(f"  summary.json")
    print(f"  predictions.jsonl ({len(full_df)} rows)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--detector", required=True, help="omini-text detector name")
    p.add_argument(
        "--n_ai",
        type=int,
        default=10000,
        help="AI rows per split (stratified by domain); 0 = use all",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--gpu", type=int, default=0, help="process-local cuda index")
    p.add_argument("--target_fpr", type=float, default=0.05)
    p.add_argument("--epsilon", type=float, default=0.0005)
    p.add_argument(
        "--out_dir",
        default=str(Path(__file__).parent / "reproductions" / "results"),
    )
    main(p.parse_args())
