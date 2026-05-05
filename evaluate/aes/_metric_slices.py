"""Shared row-slicing helpers for AES document-level evaluation.

Both ``evaluate/aes/eval_doc_level.py`` and
``evaluate/aes/eval_finetuned_detectors.py`` need to group prediction rows by
(version / generator / operation / version×generator / operation×generator /
AI-ratio bucket). The SLICING logic is shared; the per-group METRIC
computation differs between the two (they emit different schemas — one mirrors
teammate's calibrated-threshold format on HF, the other mirrors the raw
``eval_doc_level.py`` format). So this module exposes only the partition
primitives plus the key functions. Keep the metric functions separate in each
eval script.
"""

import math
from collections import defaultdict
from typing import Callable, Iterable, Mapping


# ---------------------------------------------------------------------------
# Canonical key functions
# ---------------------------------------------------------------------------


def op_key(row: Mapping) -> str:
    """Canonical operation key. Normalizes missing / empty / 'nan' to 'none'."""
    op = row.get("operation")
    if isinstance(op, float) and math.isnan(op):
        return "none"
    if op is None or op in ("", "none", "nan"):
        return "none"
    return str(op)


def generator_key(row: Mapping):
    """Canonical generator key. Returns None if neither field is present."""
    return row.get("ai_model") or row.get("model_used")


def version_key(row: Mapping):
    """Canonical version key. Returns None for rows missing ``version``."""
    return row.get("version")


# ---------------------------------------------------------------------------
# AI-ratio bucketing
# ---------------------------------------------------------------------------

#: Bucket labels in display order, from 0% up to 100%.
AI_RATIO_BUCKET_ORDER = [
    "0%", "(0,10]%", "(10,25]%", "(25,50]%",
    "(50,75]%", "(75,100)%", "100%",
]


def ai_ratio_bucket(row: Mapping, *, ratio_key: str = "AI_token_ratio"):
    """Bucket the per-row AI ratio into coarse bins. Returns None if missing.

    The key name varies between harnesses:
      * ``eval_doc_level.py`` rows carry the raw CSV column ``AI_token_ratio``.
      * ``eval_finetuned_detectors.py`` rows carry the renamed ``ai_ratio_gt``.

    Pass the appropriate ``ratio_key``.
    """
    r = row.get(ratio_key)
    if r is None:
        return None
    try:
        r = float(r)
    except (TypeError, ValueError):
        return None
    if r <= 0:
        return "0%"
    if r >= 1:
        return "100%"
    if r <= 0.10:
        return "(0,10]%"
    if r <= 0.25:
        return "(10,25]%"
    if r <= 0.50:
        return "(25,50]%"
    if r <= 0.75:
        return "(50,75]%"
    return "(75,100)%"


# ---------------------------------------------------------------------------
# Row partitioning primitives
# ---------------------------------------------------------------------------


def slice_rows(rows: Iterable[Mapping], key_fn: Callable[[Mapping], object],
               ) -> dict:
    """Partition ``rows`` by ``key_fn`` into ``{key: [row, ...]}``.

    Rows where ``key_fn`` returns ``None`` are skipped. Result is ordered by
    key (lexicographic for strings, natural for comparable types).
    """
    groups: dict = defaultdict(list)
    for r in rows:
        k = key_fn(r)
        if k is None:
            continue
        groups[k].append(r)
    return dict(sorted(groups.items()))


def slice_rows_nested(rows: Iterable[Mapping],
                      outer_fn: Callable[[Mapping], object],
                      inner_fn: Callable[[Mapping], object],
                      ) -> dict:
    """Two-level partition into ``{outer: {inner: [row, ...]}}``.

    Rows with ``None`` for either key are skipped. Result is ordered at both
    levels.
    """
    nested: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k1 = outer_fn(r)
        k2 = inner_fn(r)
        if k1 is None or k2 is None:
            continue
        nested[k1][k2].append(r)
    return {
        k1: dict(sorted(nested[k1].items()))
        for k1 in sorted(nested)
    }


def ai_ratio_bucket_rows(rows: Iterable[Mapping],
                         *, ratio_key: str = "AI_token_ratio",
                         ) -> dict:
    """Convenience wrapper around :func:`slice_rows` for AI-ratio buckets.

    Returns ``{bucket_label: [rows]}`` in canonical :data:`AI_RATIO_BUCKET_ORDER`,
    omitting empty buckets.
    """
    raw = slice_rows(rows, lambda r: ai_ratio_bucket(r, ratio_key=ratio_key))
    return {b: raw[b] for b in AI_RATIO_BUCKET_ORDER if b in raw}
