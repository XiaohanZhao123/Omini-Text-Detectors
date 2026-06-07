#!/usr/bin/env python3
"""Quantify human vs AI sample redundancy in the token-detector training data.

Motivating claim:
    Even though the row-count is ~50/50 human vs AI tokens, the underlying
    *content* of human tokens is highly redundant because v0 (human baseline)
    and v1..v8 (AI-edited variants) all descend from the same essay_id and
    share most of the human-written spans verbatim.

This script makes that precise by measuring:
  A. Row-level duplication:
       - # of rows per (essay_id, version)
       - token/sentence overlap between v0 rows that share the same essay_id
         (these should be identical if v0 is truly human-baseline)

  B. Content-level redundancy — the important one:
       For each essay_id, we take ALL rows (v0..v8 across AI models) and
       look at the concatenation of human-labeled tokens (label==0) and
       AI-labeled tokens (label==1).  We ask:
         - What fraction of human tokens in this essay's training contribution
           are duplicates of tokens already seen from the *same essay_id*?
         - Same for AI tokens.
       Duplication is measured at two granularities:
         - "token-span" level: a (word_lowercased, essay_id) pair that appeared
           in an earlier row of the same essay.
         - "sentence" level: a (sentence_text_normalized, label) that appeared
           in an earlier row.
       The sentence-level one is the most meaningful — it corresponds to the
       claim that "the same human sentence is fed to the trainer 9+ times".

  C. Corpus-level redundancy:
       Over the whole train split, how many unique human sentences vs AI
       sentences are there, vs how many occurrences the trainer sees?
       -> gives a multiplicative factor for "effective dataset size"
          human side vs AI side.

Outputs:
  - prints a summary to stdout
  - writes per-domain JSON reports to
    results/redundancy_analysis/<domain>.json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


PREPARED_CSV_DIR = Path(
    "data_local/external/opai_bench/v2/prepared/csv"
)
OUTPUT_DIR = Path(
    "results/redundancy_analysis"
)


_WS_RE = re.compile(r"\s+")


def norm_sentence(s: str) -> str:
    """Lowercase + collapse whitespace. Good enough for exact-dup detection."""
    return _WS_RE.sub(" ", s.strip().lower())


def parse_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)
    except Exception:
        return []


def analyze_domain(csv_path: Path, split: str = "train", max_rows: int | None = None):
    """Compute the redundancy stats for one domain CSV."""
    df = pd.read_csv(csv_path)
    if "split" in df.columns:
        df = df[df["split"].str.lower().str.strip() == split]
    if max_rows is not None:
        df = df.head(max_rows)
    df = df.reset_index(drop=True)

    report = {
        "csv_path": str(csv_path),
        "split": split,
        "n_rows": int(len(df)),
        "n_unique_essay_id": int(df["essay_id"].nunique()),
        "n_unique_essay_id_version": int(df.groupby(["essay_id", "version"]).ngroups),
        "n_rows_per_essay_id": df.groupby("essay_id").size().describe().to_dict(),
    }

    # --- Row-level: is v0 duplicated across AI models? ---
    v0 = df[df["version"] == "v0"]
    v0_per_essay = v0.groupby("essay_id")
    v0_dupe_text_frac = 0.0
    if len(v0) > 0:
        # within each essay_id, check if all v0 `text_clean` are identical
        exact_match_count = 0
        total_groups = 0
        for eid, grp in v0_per_essay:
            if len(grp) <= 1:
                continue
            total_groups += 1
            texts = grp["text_clean"].astype(str).map(norm_sentence).unique()
            if len(texts) == 1:
                exact_match_count += 1
        report["v0_rows_per_essay"] = int(v0_per_essay.size().median())
        report["v0_text_identical_across_rows_frac"] = (
            exact_match_count / total_groups if total_groups else None
        )

    # --- Content-level: token + sentence redundancy per essay_id ---
    # For each essay_id, walk through its rows in a deterministic order
    # (version asc, ai_model asc) and count, for each token / sentence,
    # whether we've seen it before in this essay_id under the same label.
    human_token_total = 0
    human_token_first_seen = 0  # unique occurrences within essay scope
    ai_token_total = 0
    ai_token_first_seen = 0

    human_sent_total = 0
    human_sent_unique_in_essay = 0
    ai_sent_total = 0
    ai_sent_unique_in_essay = 0

    # corpus-level: global uniqueness of (sentence_norm, label)
    global_human_sent_counter: Counter[str] = Counter()
    global_ai_sent_counter: Counter[str] = Counter()

    ordered = df.sort_values(["essay_id", "version", "ai_model"])
    for eid, grp in ordered.groupby("essay_id", sort=False):
        # local scope per essay
        seen_human_tokens: set[str] = set()
        seen_ai_tokens: set[str] = set()
        seen_human_sents: set[str] = set()
        seen_ai_sents: set[str] = set()
        for _, row in grp.iterrows():
            # TOKENS + per-token labels
            toks = parse_list(row.get("tokens"))
            tlabs = parse_list(row.get("tok_labels"))
            if len(toks) == len(tlabs) and len(toks) > 0:
                for t, lab in zip(toks, tlabs):
                    t_norm = t.lower()
                    if lab == 0:
                        human_token_total += 1
                        if t_norm not in seen_human_tokens:
                            human_token_first_seen += 1
                            seen_human_tokens.add(t_norm)
                    else:
                        ai_token_total += 1
                        if t_norm not in seen_ai_tokens:
                            ai_token_first_seen += 1
                            seen_ai_tokens.add(t_norm)

            # SENTENCES + per-sentence labels
            sents = parse_list(row.get("sentences"))
            slabs = parse_list(row.get("sent_labels"))
            if len(sents) == len(slabs) and len(sents) > 0:
                for s, lab in zip(sents, slabs):
                    s_norm = norm_sentence(str(s))
                    if not s_norm:
                        continue
                    if lab == 0:  # human sentence
                        human_sent_total += 1
                        global_human_sent_counter[s_norm] += 1
                        if s_norm not in seen_human_sents:
                            human_sent_unique_in_essay += 1
                            seen_human_sents.add(s_norm)
                    else:  # AI sentence
                        ai_sent_total += 1
                        global_ai_sent_counter[s_norm] += 1
                        if s_norm not in seen_ai_sents:
                            ai_sent_unique_in_essay += 1
                            seen_ai_sents.add(s_norm)

    # fill report
    report["content"] = {
        "human_token_total": human_token_total,
        "human_token_unique_per_essay": human_token_first_seen,
        "human_token_redundancy_ratio_per_essay": (
            human_token_total / max(1, human_token_first_seen)
        ),
        "ai_token_total": ai_token_total,
        "ai_token_unique_per_essay": ai_token_first_seen,
        "ai_token_redundancy_ratio_per_essay": (
            ai_token_total / max(1, ai_token_first_seen)
        ),
        "human_sent_total": human_sent_total,
        "human_sent_unique_per_essay": human_sent_unique_in_essay,
        "human_sent_redundancy_ratio_per_essay": (
            human_sent_total / max(1, human_sent_unique_in_essay)
        ),
        "ai_sent_total": ai_sent_total,
        "ai_sent_unique_per_essay": ai_sent_unique_in_essay,
        "ai_sent_redundancy_ratio_per_essay": (
            ai_sent_total / max(1, ai_sent_unique_in_essay)
        ),
    }

    # global dedup (across all essays in this split)
    def _global_stats(counter: Counter[str]):
        total = sum(counter.values())
        uniq = len(counter)
        mult = total / max(1, uniq)
        # how many unique sentences appear >1 times globally?
        dup_unique = sum(1 for v in counter.values() if v > 1)
        # mass-weighted duplication
        dup_occurrences = sum(v for v in counter.values() if v > 1)
        return {
            "total_occurrences": int(total),
            "unique": int(uniq),
            "redundancy_multiplier": mult,
            "unique_sents_seen_more_than_once": int(dup_unique),
            "occurrences_on_dup_sents": int(dup_occurrences),
            "dup_occurrence_frac": dup_occurrences / max(1, total),
        }

    report["corpus_global_sentence"] = {
        "human": _global_stats(global_human_sent_counter),
        "ai": _global_stats(global_ai_sent_counter),
    }

    # top-duplicated human sentences (useful narrative evidence)
    top_human = global_human_sent_counter.most_common(10)
    report["top_duplicated_human_sentences"] = [
        {"count": c, "sentence": s[:200]} for s, c in top_human
    ]
    top_ai = global_ai_sent_counter.most_common(10)
    report["top_duplicated_ai_sentences"] = [
        {"count": c, "sentence": s[:200]} for s, c in top_ai
    ]

    return report


def pretty_print(domain: str, r: dict):
    print(f"\n{'='*70}")
    print(f"DOMAIN: {domain}    split={r['split']}")
    print(f"{'='*70}")
    print(f"rows={r['n_rows']}, unique essay_id={r['n_unique_essay_id']}, "
          f"unique (essay_id,version)={r['n_unique_essay_id_version']}")
    if "v0_text_identical_across_rows_frac" in r:
        frac = r["v0_text_identical_across_rows_frac"]
        frac_s = f"{frac:.3f}" if frac is not None else "n/a"
        print(f"v0 rows per essay (median): {r['v0_rows_per_essay']}, "
              f"v0 text identical across rows: {frac_s}")

    c = r["content"]
    print("\n-- Per-essay content redundancy (tokens seen per essay_id, "
          "normalised to lowercase) --")
    print(f"  Human tokens: total={c['human_token_total']:>10}  "
          f"unique-per-essay={c['human_token_unique_per_essay']:>10}  "
          f"ratio={c['human_token_redundancy_ratio_per_essay']:.2f}x")
    print(f"     AI tokens: total={c['ai_token_total']:>10}  "
          f"unique-per-essay={c['ai_token_unique_per_essay']:>10}  "
          f"ratio={c['ai_token_redundancy_ratio_per_essay']:.2f}x")
    print(f"  Human sents:  total={c['human_sent_total']:>10}  "
          f"unique-per-essay={c['human_sent_unique_per_essay']:>10}  "
          f"ratio={c['human_sent_redundancy_ratio_per_essay']:.2f}x")
    print(f"     AI sents:  total={c['ai_sent_total']:>10}  "
          f"unique-per-essay={c['ai_sent_unique_per_essay']:>10}  "
          f"ratio={c['ai_sent_redundancy_ratio_per_essay']:.2f}x")

    g = r["corpus_global_sentence"]
    print("\n-- Corpus-global sentence dedup (all essays in split) --")
    print(f"  Human: total={g['human']['total_occurrences']:>10}  "
          f"unique={g['human']['unique']:>10}  "
          f"mult={g['human']['redundancy_multiplier']:.2f}x  "
          f"dup-occ-frac={g['human']['dup_occurrence_frac']:.3f}")
    print(f"  AI   : total={g['ai']['total_occurrences']:>10}  "
          f"unique={g['ai']['unique']:>10}  "
          f"mult={g['ai']['redundancy_multiplier']:.2f}x  "
          f"dup-occ-frac={g['ai']['dup_occurrence_frac']:.3f}")

    print("\n-- Top duplicated HUMAN sentences --")
    for row in r["top_duplicated_human_sentences"][:5]:
        print(f"  [{row['count']:>5}x] {row['sentence'][:120]}")
    print("-- Top duplicated AI sentences --")
    for row in r["top_duplicated_ai_sentences"][:5]:
        print(f"  [{row['count']:>5}x] {row['sentence'][:120]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--domains",
        nargs="+",
        default=["essay", "abstract", "news", "report"],
    )
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Cap rows per CSV (debugging).")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}
    for dom in args.domains:
        csv_path = PREPARED_CSV_DIR / f"{dom}.csv"
        if not csv_path.exists():
            print(f"[skip] {csv_path} missing")
            continue
        r = analyze_domain(csv_path, split=args.split, max_rows=args.max_rows)
        pretty_print(dom, r)
        summary[dom] = r
        with (OUTPUT_DIR / f"{dom}_{args.split}.json").open("w") as f:
            json.dump(r, f, indent=2, default=str)

    with (OUTPUT_DIR / f"summary_{args.split}.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote reports to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
