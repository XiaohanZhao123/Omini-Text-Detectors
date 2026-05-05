"""Data loader for evaluation datasets.

Loads datasets into EvalRecord format for detector evaluation.

Primary datasets (new baseline):
    - aes_chains:           Versioned essay editing chains (v0=human, v1-v3=AI-edited)
    - aes_chains_sentences: Sentence-level labels from the above
    - opai_bench_essays:        External essays (~4,200, v0-v8, GPT-generated)
    - opai_bench_abstracts:     External abstracts (AI-generated)

Legacy datasets (kept for backward compatibility):
    - education, enron, privacy, detectrl, m4, raid, hc3, turingbench
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd


@dataclass
class EvalRecord:
    """A single text sample with ground truth label and metadata."""

    text: str
    ground_truth_label: int  # 0=human, 1=AI
    source_file: str
    line_index: int
    text_field: str
    domain: str
    task: str
    ai_model: str | None


# All registered dataset names
DATASETS = [
    # -- Primary (new baseline) --
    "aes_chains",
    "aes_chains_sentences",
    "opai_bench_essays",
    "opai_bench_abstracts",
    # -- Legacy --
    "education",
    "enron",
    "privacy",
    "detectrl",
    "m4",
    "raid",
    "raid_train",
    "hc3",
    "turingbench",
]


def load_dataset(name: str, data_dir: str = "data/", **kwargs) -> Iterator[EvalRecord]:
    """Load a dataset by name. Yields EvalRecord for each text sample."""
    loaders = {
        # Primary
        "aes_chains": lambda: _load_aes_chains(data_dir, **kwargs),
        "aes_chains_sentences": lambda: _load_aes_chains_sentences(data_dir, **kwargs),
        "opai_bench_essays": lambda: _load_opai_bench(data_dir, "essays", **kwargs),
        "opai_bench_abstracts": lambda: _load_opai_bench(data_dir, "abstracts", **kwargs),
        # Legacy
        "education": lambda: _load_education(data_dir),
        "enron": lambda: _load_enron(data_dir),
        "privacy": lambda: _load_privacy(data_dir),
        "detectrl": lambda: _load_detectrl(data_dir),
        "m4": lambda: _load_m4(data_dir),
        "raid": lambda: _load_raid(data_dir, split="extra", **kwargs),
        "raid_train": lambda: _load_raid(data_dir, split="train", **kwargs),
        "hc3": lambda: _load_hc3(data_dir, **kwargs),
        "turingbench": lambda: _load_turingbench(data_dir, **kwargs),
    }

    if name in loaders:
        yield from loaders[name]()
    elif name.startswith("custom:"):
        yield from _load_opai_bench(data_dir, data_path=name[len("custom:"):], **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}. Options: {DATASETS}")


# ============================================================================
# PRIMARY DATASETS
# ============================================================================

def _load_aes_chains(
    data_dir: str,
    version: str = "all",
    data_path: str = None,
) -> Iterator[EvalRecord]:
    """AES Chains: versioned human-AI editing chains.

    v0 = human draft. v1/v2/v3 = progressively more AI editing.
    For binary eval: v0 is human (label=0), vN>0 is AI (label=1).
    """
    if data_path is None:
        data_path = str(Path(data_dir) / "aes_chains_pilot_aligned.jsonl")
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"AES chains data not found at {path}")

    versions_to_load = ["v1", "v2", "v3"] if version == "all" else [version]

    with open(path) as f:
        for idx, line in enumerate(f):
            doc = json.loads(line)
            version_map = {v["version_id"]: v for v in doc["history"]}
            v0 = version_map.get("v0")
            if v0 is None:
                continue  # skip docs without a human baseline

            # Each (v0, vN) pair is an independent binary classification sample.
            # v0 is intentionally emitted once per AI version it's compared against.
            for ver in versions_to_load:
                if ver not in version_map:
                    continue
                vn = version_map[ver]

                yield EvalRecord(
                    text=v0["text"], ground_truth_label=0,
                    source_file=str(path), line_index=idx,
                    text_field=f"v0_vs_{ver}", domain=doc["domain"],
                    task=f"aes_{ver}", ai_model=None,
                )
                yield EvalRecord(
                    text=vn["text"], ground_truth_label=1,
                    source_file=str(path), line_index=idx,
                    text_field=ver, domain=doc["domain"],
                    task=f"aes_{ver}", ai_model=vn.get("operation"),
                )


def _load_aes_chains_sentences(
    data_dir: str,
    data_path: str = None,
    ai_threshold: float = 0.5,
) -> Iterator[EvalRecord]:
    """AES Chains at sentence level.

    Splits each version into sentences, labels by AI token ratio:
    0% AI → human, >threshold AI → AI, otherwise dropped.
    """
    path = Path(data_path) if data_path else Path(data_dir) / "aes_chains_pilot.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"AES chains data not found at {path}")

    with open(path) as f:
        for doc_idx, line in enumerate(f):
            doc = json.loads(line)
            version_map = {v["version_id"]: v for v in doc["history"]}

            for vid in ["v0", "v1", "v2", "v3"]:
                if vid not in version_map:
                    continue
                v = version_map[vid]
                words = v["text"].split()
                labels = v.get("token_labels", [])
                if not labels or len(words) != len(labels):
                    continue

                for sent_idx, (sw, sl) in enumerate(
                    split_sentences_with_labels(words, labels)
                ):
                    ai_count = sl.count("ai")
                    ai_ratio = ai_count / len(sl)

                    if ai_ratio == 0:
                        gt = 0
                    elif ai_ratio > ai_threshold:
                        gt = 1
                    else:
                        continue

                    yield EvalRecord(
                        text=" ".join(sw), ground_truth_label=gt,
                        source_file=str(path), line_index=doc_idx,
                        text_field=f"{vid}_s{sent_idx}", domain=doc["domain"],
                        task=f"aes_sent_{vid}",
                        ai_model=v.get("operation") if gt == 1 else None,
                    )


def _load_opai_bench(
    data_dir: str,
    dataset_type: str = "essays",
    data_path: str = None,
    max_samples: int = None,
) -> Iterator[EvalRecord]:
    """Load prepared OpAI-Bench data (essays or abstracts).

    Expects JSONL from prepare_data.py with {human_text, ai_text, domain, ...}.
    """
    if data_path:
        path = Path(data_path)
    else:
        path = (
            Path(data_dir) / "external" / "opai_bench" / "prepared"
            / f"{dataset_type}_prepared.jsonl"
        )

    if not path.exists():
        raise FileNotFoundError(
            f"Data not found at {path}. "
            "Run: python evaluate/prepare_data.py data/external/opai_bench"
        )

    human_count = ai_count = 0
    with open(path) as f:
        for idx, line in enumerate(f):
            rec = json.loads(line)
            domain = rec.get("domain", "unknown")
            task = rec.get("task", "unknown")
            ai_model = rec.get("ai_model")
            source = rec.get("source_file", str(path))

            h_text = rec.get("human_text", "")
            a_text = rec.get("ai_text", "")

            if h_text.strip() and (max_samples is None or human_count < max_samples):
                yield EvalRecord(
                    text=h_text, ground_truth_label=0,
                    source_file=source, line_index=idx, text_field="human_text",
                    domain=domain, task=task, ai_model=None,
                )
                human_count += 1

            if a_text.strip() and (max_samples is None or ai_count < max_samples):
                yield EvalRecord(
                    text=a_text, ground_truth_label=1,
                    source_file=source, line_index=idx, text_field="ai_text",
                    domain=domain, task=task, ai_model=ai_model,
                )
                ai_count += 1

            if max_samples and human_count >= max_samples and ai_count >= max_samples:
                break


# ============================================================================
# SHARED UTILITIES
# ============================================================================

# Canonical implementation lives in sentence_eval_utils.py.
# Re-exported here for backward compatibility.
from evaluate.sentence_eval_utils import split_words_into_sentences as split_sentences_with_labels  # noqa: E402


# ============================================================================
# LEGACY DATASETS
# (Kept for backward compatibility. These use the old data directory layout.)
# ============================================================================

def _load_education(data_dir: str) -> Iterator[EvalRecord]:
    """Education Q&A: paired Human/Ai fields."""
    path = Path(data_dir) / "combined_human_ai_dataset.jsonl"
    with open(path) as f:
        for idx, line in enumerate(f):
            rec = json.loads(line)
            yield EvalRecord(
                text=rec["Human"], ground_truth_label=0,
                source_file=str(path), line_index=idx, text_field="Human",
                domain="education", task="qa", ai_model=None,
            )
            yield EvalRecord(
                text=rec["Ai"], ground_truth_label=1,
                source_file=str(path), line_index=idx, text_field="Ai",
                domain="education", task="qa", ai_model=None,
            )


def _load_enron(data_dir: str) -> Iterator[EvalRecord]:
    """Enron email: original_body/generated pairs."""
    enron_dir = Path(data_dir) / "Business_Marketing" / "Enron_Email"
    for jsonl_path in sorted(enron_dir.glob("enron_*.jsonl")):
        parts = jsonl_path.stem.split("_")
        ai_model = "_".join(parts[-2:])
        with open(jsonl_path) as f:
            for idx, line in enumerate(f):
                rec = json.loads(line)
                for text, label, field in [
                    (rec["original_body"], 0, "original_body"),
                    (rec["generated"], 1, "generated"),
                ]:
                    yield EvalRecord(
                        text=text, ground_truth_label=label,
                        source_file=str(jsonl_path), line_index=idx,
                        text_field=field, domain="business",
                        task=rec["task"], ai_model=ai_model if label else None,
                    )


def _load_privacy(data_dir: str) -> Iterator[EvalRecord]:
    """Privacy policy: original_text/generated pairs."""
    priv_dir = Path(data_dir) / "Law_Policy" / "Private_Policies"
    for jsonl_path in sorted(priv_dir.glob("privacy_*.jsonl")):
        parts = jsonl_path.stem.split("_")
        ai_model = "_".join(parts[-2:])
        with open(jsonl_path) as f:
            for idx, line in enumerate(f):
                rec = json.loads(line)
                for text, label, field in [
                    (rec["original_text"], 0, "original_text"),
                    (rec["generated"], 1, "generated"),
                ]:
                    yield EvalRecord(
                        text=text, ground_truth_label=label,
                        source_file=str(jsonl_path), line_index=idx,
                        text_field=field, domain="legal",
                        task=rec["task"], ai_model=ai_model if label else None,
                    )


def _load_detectrl(data_dir: str) -> Iterator[EvalRecord]:
    """DetectRL multidomain: separate human/machine JSON files."""
    d = Path(data_dir) / "DetectRL"
    for path, label, field in [
        (d / "DetectRL_multidomain_human_test.json", 0, "human_text"),
        (d / "DetectRL_multidomain_machine_test.json", 1, "machine_text"),
    ]:
        with open(path) as f:
            texts = json.load(f)[field]
        for idx, text in enumerate(texts):
            yield EvalRecord(
                text=text, ground_truth_label=label,
                source_file=str(path), line_index=idx, text_field=field,
                domain="detectrl", task="multidomain", ai_model=None,
            )


def _load_m4(data_dir: str) -> Iterator[EvalRecord]:
    """M4 detection benchmark: separate human/machine JSON files."""
    d = Path(data_dir) / "M4"
    for path, label, field in [
        (d / "M4_human_test.json", 0, "human_text"),
        (d / "M4_machine_test.json", 1, "machine_text"),
    ]:
        with open(path) as f:
            texts = json.load(f)[field]
        for idx, text in enumerate(texts):
            yield EvalRecord(
                text=text, ground_truth_label=label,
                source_file=str(path), line_index=idx, text_field=field,
                domain="m4", task="detection", ai_model=None,
            )


def _load_raid(data_dir, split="extra", max_samples=None, include_adversarial=False):
    """RAID benchmark. Requires: pip install raid-bench"""
    try:
        from raid.utils import load_data
    except ImportError:
        raise ImportError("Install raid-bench: pip install raid-bench")

    df = load_data(split=split, include_adversarial=include_adversarial)
    human_df = df[df["model"] == "human"]
    ai_df = df[df["model"] != "human"]

    if max_samples:
        n = max_samples // 2
        human_df = human_df.sample(n=min(n, len(human_df)), random_state=42)
        ai_df = ai_df.sample(n=min(n, len(ai_df)), random_state=42)

    for idx, row in pd.concat([human_df, ai_df]).iterrows():
        is_human = row["model"] == "human"
        yield EvalRecord(
            text=row["generation"], ground_truth_label=0 if is_human else 1,
            source_file=f"raid:{split}", line_index=idx, text_field="generation",
            domain=row["domain"], task=row.get("attack", "none") or "none",
            ai_model=None if is_human else row["model"],
        )


def _load_hc3(data_dir, subset="all", max_samples=None, source="huggingface"):
    """HC3: Human ChatGPT Comparison Corpus (EMNLP 2023)."""
    if source == "huggingface":
        from datasets import load_dataset as hf_load
        ds = hf_load("Hello-SimpleAI/HC3", subset, split="train")
    else:
        ds = pd.read_parquet(
            Path(data_dir) / "HC3" / f"{subset}.parquet"
        ).to_dict("records")

    h_count = a_count = 0
    for idx, item in enumerate(ds):
        domain = item.get("source", subset)
        for ans in item.get("human_answers", []):
            if max_samples and h_count >= max_samples:
                break
            if ans and ans.strip():
                yield EvalRecord(
                    text=ans, ground_truth_label=0,
                    source_file=f"hc3:{subset}", line_index=idx,
                    text_field="human_answers", domain=domain,
                    task="qa", ai_model=None,
                )
                h_count += 1
        for ans in item.get("chatgpt_answers", []):
            if max_samples and a_count >= max_samples:
                break
            if ans and ans.strip():
                yield EvalRecord(
                    text=ans, ground_truth_label=1,
                    source_file=f"hc3:{subset}", line_index=idx,
                    text_field="chatgpt_answers", domain=domain,
                    task="qa", ai_model="chatgpt",
                )
                a_count += 1
        if max_samples and h_count >= max_samples and a_count >= max_samples:
            break


def _load_turingbench(data_dir, task="TT", generator="gpt3", split="test", max_samples=None):
    """TuringBench (EMNLP 2021): human vs 19 generators."""
    import csv
    csv_path = Path(data_dir) / "TuringBench" / f"{task}_{generator}" / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"TuringBench not found at {csv_path}")

    h_count = a_count = 0
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for idx, row in enumerate(reader):
            if len(row) < 2 or not row[0].strip():
                continue
            is_human = row[1].lower() == "human"
            if max_samples:
                if is_human and h_count >= max_samples:
                    continue
                if not is_human and a_count >= max_samples:
                    continue
            yield EvalRecord(
                text=row[0], ground_truth_label=0 if is_human else 1,
                source_file=str(csv_path), line_index=idx, text_field="text",
                domain="news", task=f"{task}_{generator}",
                ai_model=None if is_human else row[1],
            )
            if is_human:
                h_count += 1
            else:
                a_count += 1
            if max_samples and h_count >= max_samples and a_count >= max_samples:
                break


# ============================================================================

if __name__ == "__main__":
    for name in DATASETS:
        try:
            records = list(load_dataset(name))
            human = sum(1 for r in records if r.ground_truth_label == 0)
            print(f"{name}: {len(records)} records ({human} human, {len(records)-human} AI)")
        except (FileNotFoundError, ImportError) as e:
            print(f"{name}: SKIPPED - {e}")
