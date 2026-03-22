#!/usr/bin/env python3
"""Proper calibration analysis for 6 native-confidence methods on v0-v8 data.

Three-tier analysis comparing like-for-like:

Tier A — Coverage calibration (1 method: GigaCheck)
    Score = AI char coverage fraction ≈ GT AI_char_ratio.
    Reliability diagram + scatter plot + ECE, Pearson r, MAE.

Tier B — Word-level emission calibration (2 methods: DAMASHA, SeqXGPT)
    Pre-CRF softmax P(AI) per word vs GT tok_labels.
    CRF caveat: these are emission probabilities, NOT true posteriors.
    SeqXGPT: must filter out space tokens before alignment.
    Reliability diagrams + score distributions + ECE, AUC, Brier.

Tier C — Document-level sensitivity (3 classifiers: desklib, e5-small, radar)
    P(doc is AI) is a binary classification confidence, NOT an AI-fraction estimate.
    Calibrating against AI_char_ratio would be conceptually wrong.
    Instead: trajectory sensitivity, binary calibration at 2 thresholds (0.0 and 0.5),
    Spearman rank correlation, AUC.

Tier D — Cross-tier summary (bar charts + metrics JSON).

Usage:
    cd /data/spiderman/jiachengl/Omni-text
    python draft/calibration_analysis.py
"""

import ast
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")

NC_DIR = Path(__file__).resolve().parent / "results" / "native_confidence"
GT_CSV = Path(__file__).resolve().parent / "essays_v0_v8_spans_with_eval.csv"
OUT_DIR = Path(__file__).resolve().parent / "results" / "calibration"
VERSIONS = ["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"]

DISPLAY = {
    "desklib": "Desklib", "e5-small": "E5-Small", "radar": "RADAR",
    "gigacheck": "GigaCheck", "damasha": "DAMASHA", "seqxgpt": "SeqXGPT",
}


def load_gt():
    """Load ground truth, indexed by (essay_id, version)."""
    df = pd.read_csv(GT_CSV)
    gt = {}
    for _, row in df.iterrows():
        key = (str(row["essay_id"]), row["version"])
        tok_labels = None
        if pd.notna(row.get("tok_labels")):
            try:
                tok_labels = ast.literal_eval(row["tok_labels"])
            except (ValueError, SyntaxError):
                pass
        tokens = None
        if pd.notna(row.get("tokens")):
            try:
                tokens = ast.literal_eval(row["tokens"])
            except (ValueError, SyntaxError):
                pass
        gt[key] = {
            "ai_char_ratio": float(row["AI_char_ratio"]),
            "ai_token_ratio": float(row["AI_token_ratio"]),
            "tok_labels": tok_labels,
            "tokens": tokens,
        }
    return gt


def load_native_confidence(method):
    path = NC_DIR / f"{method}.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


# ─── Calibration helpers ───

def compute_ece(predicted, actual, n_bins=10):
    """Expected Calibration Error with bin details."""
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(predicted)
    if total == 0:
        return 0.0, [], [], []
    bin_counts, bin_pred_means, bin_actual_means = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (predicted >= lo) & (predicted <= hi)
        else:
            mask = (predicted >= lo) & (predicted < hi)
        count = mask.sum()
        if count > 0:
            pm = predicted[mask].mean()
            am = actual[mask].mean()
            ece += count * abs(pm - am)
            bin_counts.append(int(count))
            bin_pred_means.append(float(pm))
            bin_actual_means.append(float(am))
        else:
            bin_counts.append(0)
            bin_pred_means.append(float((lo + hi) / 2))
            bin_actual_means.append(0.0)
    return float(ece / total), bin_pred_means, bin_actual_means, bin_counts


def plot_reliability(ax, bin_preds, bin_actuals, bin_counts, ece, title,
                     xlabel="Predicted", ylabel_right="Actual fraction",
                     n_samples=None, extra_text=None):
    """Plot a single reliability diagram on the given axis."""
    bar_width = 0.08
    ax.bar(bin_preds, bin_counts, width=bar_width, alpha=0.2, color="steelblue")
    ax.set_ylabel("Count", color="steelblue", fontsize=9)
    ax.tick_params(axis="y", labelcolor="steelblue", labelsize=8)

    ax2 = ax.twinx()
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    valid = [i for i in range(len(bin_counts)) if bin_counts[i] > 0]
    if valid:
        vp = [bin_preds[i] for i in valid]
        va = [bin_actuals[i] for i in valid]
        ax2.plot(vp, va, "o-", color="crimson", markersize=5, linewidth=1.5,
                 label=f"ECE = {ece:.3f}")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel(ylabel_right, fontsize=9)
    ax2.tick_params(labelsize=8)
    n_str = f"  (n={n_samples:,})" if n_samples else ""
    ax.set_title(f"{title}{n_str}", fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    if extra_text:
        ax2.text(0.98, 0.12, extra_text, transform=ax2.transAxes, fontsize=7,
                 ha="right", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    ax2.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.2)


# ═══════════════════════════════════════════════════════════════════════════════
# TIER A: Coverage calibration — GigaCheck
# ═══════════════════════════════════════════════════════════════════════════════

def tier_a_gigacheck(gt, all_metrics):
    """GigaCheck coverage ≈ AI char fraction → calibrate against AI_char_ratio."""
    records = load_native_confidence("gigacheck")
    predicted, actual = [], []
    for rec in records:
        for ver in VERSIONS:
            vdata = rec["versions"].get(ver, {})
            if "error" in vdata:
                continue
            cov = vdata.get("confidence")
            if cov is None:
                continue
            gt_key = (rec["q_id"], ver)
            if gt_key in gt:
                predicted.append(float(cov))
                actual.append(gt[gt_key]["ai_char_ratio"])

    predicted = np.array(predicted)
    actual = np.array(actual)

    ece, bp, ba, bc = compute_ece(predicted, actual)
    pearson_r = float(np.corrcoef(predicted, actual)[0, 1])
    mae = float(np.mean(np.abs(predicted - actual)))

    # Plot: reliability diagram + scatter
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    plot_reliability(ax1, bp, ba, bc, ece,
                     "GigaCheck: coverage vs GT AI char ratio",
                     xlabel="Predicted AI coverage",
                     ylabel_right="Actual AI char ratio",
                     n_samples=len(predicted))

    # Scatter plot
    ax2.scatter(predicted, actual, alpha=0.3, s=20, color="steelblue", edgecolors="none")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax2.set_xlabel("Predicted AI coverage", fontsize=10)
    ax2.set_ylabel("Actual AI char ratio (GT)", fontsize=10)
    ax2.set_title(f"GigaCheck scatter  (r={pearson_r:.3f}, MAE={mae:.3f})",
                  fontsize=10, fontweight="bold")
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Tier A: Coverage Calibration — GigaCheck",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "tier_A_gigacheck_coverage.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Binary AUC (any AI → 1) for summary chart
    binary_gt = (actual > 0).astype(int)
    auc = roc_auc_score(binary_gt, predicted) if len(np.unique(binary_gt)) > 1 else 0.0

    all_metrics["gigacheck"] = {
        "tier": "A", "analysis": "coverage_calibration",
        "score_meaning": "AI char coverage fraction (P(ai)+P(mixed) from 3-class softmax, or DETR coverage)",
        "gt_compared_to": "AI_char_ratio",
        "ece": ece, "pearson_r": pearson_r, "mae": mae, "auc": float(auc),
        "n_samples": int(len(predicted)),
    }
    print(f"  Tier A: GigaCheck — ECE={ece:.3f}, r={pearson_r:.3f}, MAE={mae:.3f} (n={len(predicted)})")


# ═══════════════════════════════════════════════════════════════════════════════
# TIER B: Word-level emission calibration — DAMASHA, SeqXGPT
# ═══════════════════════════════════════════════════════════════════════════════

def collect_word_scores(method, gt):
    """Collect per-word P(AI) and GT labels from native_confidence JSONL.

    For SeqXGPT: filters out space tokens before alignment.
    For DAMASHA: words align directly to GT tokens (both whitespace-split).
    """
    records = load_native_confidence(method)
    predicted, actual = [], []

    for rec in records:
        q_id = rec["q_id"]
        for ver in VERSIONS:
            vdata = rec["versions"].get(ver, {})
            if "error" in vdata:
                continue
            word_conf = vdata.get("word_confidence")
            words = vdata.get("words")
            if not word_conf or not words:
                continue
            gt_key = (q_id, ver)
            if gt_key not in gt or gt[gt_key]["tok_labels"] is None:
                continue
            gt_tok_labels = gt[gt_key]["tok_labels"]

            if method == "seqxgpt":
                # Filter out space tokens
                filtered = [(w, p) for w, p in zip(words, word_conf) if w.strip()]
                if not filtered:
                    continue
                pred_words, pred_scores = zip(*filtered)
            else:
                pred_words = words
                pred_scores = word_conf

            # Align: sequential matching (both whitespace-split)
            n = min(len(pred_scores), len(gt_tok_labels))
            for i in range(n):
                predicted.append(float(pred_scores[i]))
                actual.append(float(gt_tok_labels[i]))

    return np.array(predicted), np.array(actual)


def tier_b_wordlevel(gt, all_metrics):
    """Word-level P(AI) vs GT labels for DAMASHA and SeqXGPT.

    Saves two separate plots:
      tier_B_wordlevel_calibration.png  — reliability diagrams
      tier_B_wordlevel_distributions.png — score distributions by GT label
    """
    # Collect data for both methods first (reuse across both plots)
    method_data = {}
    for method in ["damasha", "seqxgpt"]:
        pred, actual = collect_word_scores(method, gt)
        ece, bp, ba, bc = compute_ece(pred, actual)
        auc = roc_auc_score(actual, pred) if len(np.unique(actual)) > 1 else 0.0
        brier = brier_score_loss(actual, pred)
        method_data[method] = {
            "pred": pred, "actual": actual,
            "ece": ece, "bp": bp, "ba": ba, "bc": bc,
            "auc": auc, "brier": brier,
        }
        all_metrics[method] = {
            "tier": "B", "analysis": "word_level_emission_calibration",
            "score_meaning": "per-word softmax P(AI) from pre-CRF emission logits",
            "gt_compared_to": "per-word binary label from tok_labels",
            "crf_caveat": "emission probabilities before CRF; CRF transition matrix can override",
            "ece": float(ece), "auc": float(auc), "brier": float(brier),
            "n_words": int(len(pred)),
            "n_human_words": int((actual == 0).sum()),
            "n_ai_words": int((actual == 1).sum()),
        }
        print(f"  Tier B: {DISPLAY[method]:8s} — ECE={ece:.3f}, AUC={auc:.3f}, "
              f"Brier={brier:.3f} (n={len(pred):,} words)")

    # ── Plot 1: Reliability diagrams ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for col, method in enumerate(["damasha", "seqxgpt"]):
        d = method_data[method]
        crf_note = "CRF caveat: emission probs, NOT true posteriors"
        plot_reliability(axes[col], d["bp"], d["ba"], d["bc"], d["ece"],
                         f"{DISPLAY[method]}: word P(AI) vs GT label",
                         xlabel="Predicted P(AI) [pre-CRF softmax]",
                         ylabel_right="Actual P(word is AI)",
                         n_samples=len(d["pred"]),
                         extra_text=crf_note)
    plt.suptitle("Tier B: Word-Level Emission Calibration — P(AI) vs GT Word Labels\n"
                 "(pre-CRF softmax emissions, NOT true CRF posteriors)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "tier_B_wordlevel_calibration.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 2: Score distributions by GT label ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for col, method in enumerate(["damasha", "seqxgpt"]):
        d = method_data[method]
        ax = axes[col]
        human_scores = d["pred"][d["actual"] == 0]
        ai_scores = d["pred"][d["actual"] == 1]
        ax.hist(human_scores, bins=50, alpha=0.5, color="steelblue",
                label=f"GT=Human (n={len(human_scores):,})", density=True)
        ax.hist(ai_scores, bins=50, alpha=0.5, color="crimson",
                label=f"GT=AI (n={len(ai_scores):,})", density=True)
        ax.set_xlabel("Predicted P(AI)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"{DISPLAY[method]}: score distribution by GT label",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    plt.suptitle("Tier B: Word-Level Score Distributions\n"
                 "(pre-CRF softmax emissions, NOT true CRF posteriors)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "tier_B_wordlevel_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# TIER C: Document-level sensitivity — desklib, e5-small, radar
# ═══════════════════════════════════════════════════════════════════════════════

def collect_doc_scores(method, gt):
    """Collect document-level P(AI) and GT AI_char_ratio."""
    records = load_native_confidence(method)
    scores, gt_ratios, versions_list = [], [], []
    for rec in records:
        for ver in VERSIONS:
            vdata = rec["versions"].get(ver, {})
            if "error" in vdata:
                continue
            s = vdata.get("confidence")
            if s is None:
                continue
            gt_key = (rec["q_id"], ver)
            if gt_key in gt:
                scores.append(float(s))
                gt_ratios.append(gt[gt_key]["ai_char_ratio"])
                versions_list.append(ver)
    return np.array(scores), np.array(gt_ratios), versions_list


def tier_c_sensitivity(gt, all_metrics):
    """Document-level binary classifiers: trajectory + binary calibration + Spearman."""
    methods = ["desklib", "e5-small", "radar"]

    # ── Plot 1: Trajectory sensitivity ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, method in enumerate(methods):
        records = load_native_confidence(method)
        ax = axes[idx]

        # Per-version stats
        version_scores = {v: [] for v in VERSIONS}
        version_gt = {v: [] for v in VERSIONS}
        for rec in records:
            for ver in VERSIONS:
                vdata = rec["versions"].get(ver, {})
                if "error" in vdata:
                    continue
                s = vdata.get("confidence")
                if s is None:
                    continue
                gt_key = (rec["q_id"], ver)
                if gt_key in gt:
                    version_scores[ver].append(float(s))
                    version_gt[ver].append(gt[gt_key]["ai_char_ratio"])

        means = [np.mean(version_scores[v]) if version_scores[v] else np.nan for v in VERSIONS]
        stds = [np.std(version_scores[v]) if version_scores[v] else np.nan for v in VERSIONS]
        gt_means = [np.mean(version_gt[v]) if version_gt[v] else np.nan for v in VERSIONS]

        x = np.arange(len(VERSIONS))
        ax.errorbar(x, means, yerr=stds, fmt="o-", color="steelblue",
                     capsize=3, label="P(doc is AI)", markersize=5, linewidth=1.5)
        ax.plot(x, gt_means, "s--", color="crimson", label="GT AI char ratio",
                markersize=5, alpha=0.8, linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(VERSIONS, fontsize=8)
        ax.set_ylabel("Score / Ratio", fontsize=9)
        ax.set_ylim(-0.05, 1.1)
        ax.set_title(f"{DISPLAY[method]}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        # Monotonicity annotation
        valid_pairs = [(i, i+1) for i in range(len(means)-1)
                       if not np.isnan(means[i]) and not np.isnan(means[i+1])]
        if valid_pairs:
            increases = sum(1 for i, j in valid_pairs if means[j] > means[i])
            mono = increases / len(valid_pairs)
        else:
            mono = 0.0
        ax.text(0.98, 0.02, f"monotonicity={mono:.0%}", transform=ax.transAxes,
                fontsize=8, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.7))

    plt.suptitle("Tier C: Document-Level Sensitivity — P(doc is AI) across v0→v8\n"
                 "Note: these are binary classification confidences, NOT AI-fraction estimates",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "tier_C_trajectory_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Tier C: Saved trajectory sensitivity plot")

    # ── Plot 2: Binary calibration at two thresholds ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    thresholds = [("threshold=0.0 (any AI)", 0.0), ("threshold=0.5 (majority AI)", 0.5)]

    for row, (thresh_label, thresh_val) in enumerate(thresholds):
        for col, method in enumerate(methods):
            scores, gt_ratios, _ = collect_doc_scores(method, gt)
            binary_gt = (gt_ratios > thresh_val).astype(float)
            base_rate = binary_gt.mean()

            ece, bp, ba, bc = compute_ece(scores, binary_gt)
            auc = roc_auc_score(binary_gt, scores) if len(np.unique(binary_gt)) > 1 else 0.0

            ax = axes[row, col]
            base_rate_note = f"Base rate: {base_rate:.1%}"
            plot_reliability(ax, bp, ba, bc, ece,
                             f"{DISPLAY[method]} — {thresh_label}",
                             xlabel="Predicted P(doc is AI)",
                             ylabel_right="Actual P(AI > threshold)",
                             n_samples=len(scores),
                             extra_text=f"{base_rate_note}\nAUC = {auc:.3f}")

            # Store metrics for the more informative threshold (0.5)
            if thresh_val == 0.5:
                spearman_r, spearman_p = stats.spearmanr(scores, gt_ratios)
                all_metrics[method] = {
                    "tier": "C", "analysis": "document_level_sensitivity",
                    "score_meaning": "P(doc is AI) from trained binary classifier",
                    "gt_compared_to": "binary detection + AI_char_ratio (ordinal)",
                    "why_not_fraction_calibration": (
                        "P(doc=AI) is a binary classification confidence, not an AI-fraction estimate. "
                        "These classifiers were trained on fully-human vs fully-AI documents."
                    ),
                    "ece_at_threshold_0.0": float(compute_ece(scores, (gt_ratios > 0).astype(float))[0]),
                    "ece_at_threshold_0.5": float(ece),
                    "auc_at_threshold_0.0": float(
                        roc_auc_score((gt_ratios > 0).astype(float), scores)
                        if len(np.unique((gt_ratios > 0).astype(float))) > 1 else 0.0
                    ),
                    "auc_at_threshold_0.5": float(auc),
                    "base_rate_at_0.0": float((gt_ratios > 0).mean()),
                    "base_rate_at_0.5": float(base_rate),
                    "spearman_rho": float(spearman_r),
                    "spearman_p": float(spearman_p),
                    "n_samples": int(len(scores)),
                }
                print(f"  Tier C: {DISPLAY[method]:8s} — ECE@0.5={ece:.3f}, AUC@0.5={auc:.3f}, "
                      f"Spearman rho={spearman_r:.3f}")

    plt.suptitle("Tier C: Binary Calibration at Two Thresholds\n"
                 "Top: any AI content (base rate ~89%); Bottom: majority AI (base rate ~56%)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "tier_C_binary_calibration.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Tier C: Saved binary calibration plots")


# ═══════════════════════════════════════════════════════════════════════════════
# TIER D: Cross-tier summary
# ═══════════════════════════════════════════════════════════════════════════════

def tier_d_summary(all_metrics):
    """Summary bar charts: ECE (Tiers A+B), AUC (all 6), Spearman (Tier C)."""
    from matplotlib.patches import Patch

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5.5))

    # ── ECE (Tiers A + B only — like-for-like) ──
    ece_methods = []
    for m, v in all_metrics.items():
        if v["tier"] in ("A", "B") and "ece" in v:
            ece_methods.append((m, v["ece"], v["tier"]))
    ece_methods.sort(key=lambda x: x[1])

    colors_ece = {"A": "mediumseagreen", "B": "coral"}
    bars = ax1.barh(
        [DISPLAY[m] for m, _, _ in ece_methods],
        [e for _, e, _ in ece_methods],
        color=[colors_ece[t] for _, _, t in ece_methods],
        edgecolor="white"
    )
    for bar, (_, val, _) in zip(bars, ece_methods):
        ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=9)
    ax1.set_xlabel("ECE (lower = better calibrated)", fontsize=10)
    ax1.set_title("Calibration (ECE)\nTiers A + B only", fontsize=11, fontweight="bold")
    ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.3)
    ax1.legend(handles=[
        Patch(facecolor="mediumseagreen", label="A: Coverage vs AI_char_ratio"),
        Patch(facecolor="coral", label="B: Word P(AI) vs GT labels"),
    ], fontsize=7, loc="lower right")

    # ── AUC (all 6 methods) ──
    auc_data = []
    for m, v in all_metrics.items():
        if "auc" in v:
            auc_data.append((m, v["auc"], v["tier"]))
        elif "auc_at_threshold_0.5" in v:
            auc_data.append((m, v["auc_at_threshold_0.5"], v["tier"]))
    auc_data.sort(key=lambda x: x[1], reverse=True)

    colors_auc = {"A": "mediumseagreen", "B": "coral", "C": "steelblue"}
    bars = ax2.barh(
        [DISPLAY[m] for m, _, _ in auc_data],
        [a for _, a, _ in auc_data],
        color=[colors_auc[t] for _, _, t in auc_data],
        edgecolor="white"
    )
    for bar, (_, val, _) in zip(bars, auc_data):
        ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=9)
    ax2.set_xlabel("AUC (higher = better discrimination)", fontsize=10)
    ax2.set_title("Discrimination (AUC)\nAll 6 methods", fontsize=11, fontweight="bold")
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.1)
    ax2.grid(axis="x", alpha=0.3)
    ax2.legend(handles=[
        Patch(facecolor="mediumseagreen", label="A: Coverage"),
        Patch(facecolor="coral", label="B: Word-level"),
        Patch(facecolor="steelblue", label="C: Doc-level classifier"),
    ], fontsize=7, loc="lower right")

    # ── Spearman rho (Tier C only — rank correlation with AI fraction) ──
    spearman_data = []
    for m, v in all_metrics.items():
        if "spearman_rho" in v:
            spearman_data.append((m, v["spearman_rho"]))
    spearman_data.sort(key=lambda x: x[1], reverse=True)

    bars = ax3.barh(
        [DISPLAY[m] for m, _ in spearman_data],
        [s for _, s in spearman_data],
        color="steelblue", edgecolor="white"
    )
    for bar, (_, val) in zip(bars, spearman_data):
        ax3.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=9)
    ax3.set_xlabel("Spearman rho (higher = better ordinal tracking)", fontsize=10)
    ax3.set_title("Rank Correlation with AI Fraction\nTier C classifiers only", fontsize=11, fontweight="bold")
    ax3.invert_yaxis()
    ax3.set_xlim(-0.1, 1.1)
    ax3.grid(axis="x", alpha=0.3)

    plt.suptitle("Cross-Tier Summary — 6 Native Confidence Methods",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "summary_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Tier D: Saved summary comparison")


# ─── Main ───

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Clean old files
    for f in OUT_DIR.glob("*"):
        f.unlink()

    gt = load_gt()
    all_metrics = {}

    print("Running calibration analysis (6 methods, 3 tiers)...\n")

    tier_a_gigacheck(gt, all_metrics)
    tier_b_wordlevel(gt, all_metrics)
    tier_c_sensitivity(gt, all_metrics)
    tier_d_summary(all_metrics)

    # Save metrics JSON
    with open(OUT_DIR / "calibration_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("\n  Saved calibration_metrics.json")

    # Print summary table
    print(f"\n{'='*100}")
    print(f"{'Method':<12} {'Tier':<4} {'Analysis':<32} {'ECE':>6} {'AUC':>6} {'Spearman':>9} {'N':>8}")
    print(f"{'-'*100}")
    for m in sorted(all_metrics.keys()):
        met = all_metrics[m]
        ece_str = f"{met['ece']:.3f}" if "ece" in met else "  —"
        if "ece_at_threshold_0.5" in met:
            ece_str = f"{met['ece_at_threshold_0.5']:.3f}"
        auc_str = "  —"
        if "auc" in met:
            auc_str = f"{met['auc']:.3f}"
        elif "auc_at_threshold_0.5" in met:
            auc_str = f"{met['auc_at_threshold_0.5']:.3f}"
        spearman_str = "  —"
        if "spearman_rho" in met:
            spearman_str = f"{met['spearman_rho']:.3f}"
        n = met.get("n_samples", met.get("n_words", 0))
        print(f"{DISPLAY[m]:<12} {met['tier']:<4} {met['analysis'][:32]:<32} "
              f"{ece_str:>6} {auc_str:>6} {spearman_str:>9} {n:>8}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
