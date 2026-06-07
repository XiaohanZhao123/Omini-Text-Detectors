#!/usr/bin/env python3
"""Single-essay walkthrough to make the sentence-level redundancy concrete.

Picks one essay_id, shows how its 27 rows (9 versions x 3 AI models) carry
largely the SAME human sentences, only with local AI edits."""
from __future__ import annotations
import ast, json, re, sys
from collections import Counter
from pathlib import Path
import pandas as pd

PREPARED_CSV_DIR = Path(
    "data_local/external/opai_bench/v2/prepared/csv"
)

_WS = re.compile(r"\s+")
norm = lambda s: _WS.sub(" ", str(s).strip().lower())
plist = lambda x: ast.literal_eval(x) if isinstance(x, str) else (x or [])

def walkthrough(domain: str, split: str = "train", n_essays: int = 3):
    df = pd.read_csv(PREPARED_CSV_DIR / f"{domain}.csv")
    if "split" in df.columns:
        df = df[df["split"].str.lower().str.strip() == split]
    # pick the first few essay_ids deterministically
    ids = sorted(df["essay_id"].unique())[:n_essays]
    for eid in ids:
        grp = df[df["essay_id"] == eid].sort_values(["version", "ai_model"])
        print(f"\n{'='*72}\nessay_id={eid!r}  ({len(grp)} rows)")
        print(f"{'='*72}")
        all_human_sents = Counter()
        all_ai_sents = Counter()
        per_row_info = []
        for _, r in grp.iterrows():
            sents = plist(r.get("sentences"))
            labs  = plist(r.get("sent_labels"))
            if len(sents) != len(labs):
                continue
            row_human = [norm(s) for s, l in zip(sents, labs) if l == 0]
            row_ai    = [norm(s) for s, l in zip(sents, labs) if l == 1]
            for s in row_human: all_human_sents[s] += 1
            for s in row_ai:    all_ai_sents[s] += 1
            per_row_info.append({
                "version": r["version"],
                "ai_model": r["ai_model"],
                "n_sents": len(sents),
                "n_human_sents": len(row_human),
                "n_ai_sents": len(row_ai),
                "AI_sent_ratio": r.get("AI_sent_ratio"),
            })

        print(f"{'version':<6} {'ai_model':<22} {'#sent':>6} {'#hum':>5} {'#AI':>5}  "
              f"AI_sent_ratio")
        for info in per_row_info:
            print(f"{info['version']:<6} {info['ai_model']:<22} "
                  f"{info['n_sents']:>6} {info['n_human_sents']:>5} "
                  f"{info['n_ai_sents']:>5}   {info['AI_sent_ratio']}")

        total_human = sum(all_human_sents.values())
        uniq_human = len(all_human_sents)
        total_ai = sum(all_ai_sents.values())
        uniq_ai = len(all_ai_sents)
        print(f"\n -> Across this essay's {len(per_row_info)} rows:")
        print(f"      HUMAN: {total_human} sentence occurrences, "
              f"{uniq_human} unique  ({total_human/max(1,uniq_human):.1f}x)")
        print(f"         AI: {total_ai} sentence occurrences, "
              f"{uniq_ai} unique  ({total_ai/max(1,uniq_ai):.1f}x)")

        # Show how many times the TOP-3 human sentences recur
        print("\n -> Top 3 most-repeated HUMAN sentences in this essay:")
        for s, c in all_human_sents.most_common(3):
            print(f"      [{c:>3}x] {s[:140]}")
        # And AI
        print(" -> Top 3 most-repeated AI sentences in this essay:")
        for s, c in all_ai_sents.most_common(3):
            print(f"      [{c:>3}x] {s[:140]}")


if __name__ == "__main__":
    domain = sys.argv[1] if len(sys.argv) > 1 else "essay"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    walkthrough(domain, n_essays=n)
