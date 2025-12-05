"""Data loader for evaluation datasets.

Loads and flattens datasets into EvalRecord format for detector evaluation.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

DATASETS = ["education", "enron", "privacy", "detectrl", "m4", "raid", "raid_train"]


@dataclass
class EvalRecord:
    """Single evaluation record with text and metadata."""

    text: str
    ground_truth_label: int  # 0=human, 1=AI
    source_file: str
    line_index: int
    text_field: str
    domain: str
    task: str
    ai_model: str | None


def load_dataset(name: str, data_dir: str = "data/", **kwargs) -> Iterator[EvalRecord]:
    """Load and flatten a single dataset.

    Args:
        name: Dataset name - one of DATASETS
        data_dir: Base data directory path
        **kwargs: Dataset-specific options (e.g., max_samples for RAID)

    Yields:
        EvalRecord for each text sample (both human and AI)
    """
    if name == "education":
        yield from _load_education(data_dir)
    elif name == "enron":
        yield from _load_enron(data_dir)
    elif name == "privacy":
        yield from _load_privacy(data_dir)
    elif name == "detectrl":
        yield from _load_detectrl(data_dir)
    elif name == "m4":
        yield from _load_m4(data_dir)
    elif name == "raid":
        yield from _load_raid(data_dir, split="extra", **kwargs)
    elif name == "raid_train":
        yield from _load_raid(data_dir, split="train", **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}. Must be one of {DATASETS}")


def _load_education(data_dir: str) -> Iterator[EvalRecord]:
    """Load education Q&A dataset - paired Human/Ai fields."""
    path = Path(data_dir) / "combined_human_ai_dataset.jsonl"
    rel_path = str(path)

    with open(path) as f:
        for idx, line in enumerate(f):
            record = json.loads(line)

            # Human record
            yield EvalRecord(
                text=record["Human"],
                ground_truth_label=0,
                source_file=rel_path,
                line_index=idx,
                text_field="Human",
                domain="education",
                task="qa",
                ai_model=None,
            )

            # AI record
            yield EvalRecord(
                text=record["Ai"],
                ground_truth_label=1,
                source_file=rel_path,
                line_index=idx,
                text_field="Ai",
                domain="education",
                task="qa",
                ai_model=None,
            )


def _load_enron(data_dir: str) -> Iterator[EvalRecord]:
    """Load Enron email dataset - original_body/generated fields."""
    enron_dir = Path(data_dir) / "Business_Marketing" / "Enron_Email"

    for jsonl_path in sorted(enron_dir.glob("enron_*.jsonl")):
        # Extract model from filename: enron_<task>_<model>.jsonl
        # e.g., enron_title_to_body_qwen3_8b.jsonl
        filename = jsonl_path.stem
        parts = filename.split("_")
        ai_model = "_".join(parts[-2:])  # last two parts: qwen3_8b or gpt_oss_20b

        rel_path = str(jsonl_path)

        with open(jsonl_path) as f:
            for idx, line in enumerate(f):
                record = json.loads(line)
                task = record["task"]

                # Human record
                yield EvalRecord(
                    text=record["original_body"],
                    ground_truth_label=0,
                    source_file=rel_path,
                    line_index=idx,
                    text_field="original_body",
                    domain="business",
                    task=task,
                    ai_model=ai_model,
                )

                # AI record
                yield EvalRecord(
                    text=record["generated"],
                    ground_truth_label=1,
                    source_file=rel_path,
                    line_index=idx,
                    text_field="generated",
                    domain="business",
                    task=task,
                    ai_model=ai_model,
                )


def _load_privacy(data_dir: str) -> Iterator[EvalRecord]:
    """Load privacy policy dataset - original_text/generated fields."""
    privacy_dir = Path(data_dir) / "Law_Policy" / "Private_Policies"

    for jsonl_path in sorted(privacy_dir.glob("privacy_*.jsonl")):
        # Extract model from filename: privacy_<task>_<model>.jsonl
        filename = jsonl_path.stem
        parts = filename.split("_")
        ai_model = "_".join(parts[-2:])  # qwen3_8b

        rel_path = str(jsonl_path)

        with open(jsonl_path) as f:
            for idx, line in enumerate(f):
                record = json.loads(line)
                task = record["task"]

                # Human record
                yield EvalRecord(
                    text=record["original_text"],
                    ground_truth_label=0,
                    source_file=rel_path,
                    line_index=idx,
                    text_field="original_text",
                    domain="legal",
                    task=task,
                    ai_model=ai_model,
                )

                # AI record
                yield EvalRecord(
                    text=record["generated"],
                    ground_truth_label=1,
                    source_file=rel_path,
                    line_index=idx,
                    text_field="generated",
                    domain="legal",
                    task=task,
                    ai_model=ai_model,
                )


def _load_detectrl(data_dir: str) -> Iterator[EvalRecord]:
    """Load DetectRL multidomain dataset - separate human/machine JSON files."""
    detectrl_dir = Path(data_dir) / "DetectRL"
    human_path = detectrl_dir / "DetectRL_multidomain_human_test.json"
    machine_path = detectrl_dir / "DetectRL_multidomain_machine_test.json"

    # Load human texts
    with open(human_path) as f:
        human_data = json.load(f)
    human_texts = human_data["human_text"]

    for idx, text in enumerate(human_texts):
        yield EvalRecord(
            text=text,
            ground_truth_label=0,
            source_file=str(human_path),
            line_index=idx,
            text_field="human_text",
            domain="detectrl",
            task="multidomain",
            ai_model=None,
        )

    # Load machine texts
    with open(machine_path) as f:
        machine_data = json.load(f)
    machine_texts = machine_data["machine_text"]

    for idx, text in enumerate(machine_texts):
        yield EvalRecord(
            text=text,
            ground_truth_label=1,
            source_file=str(machine_path),
            line_index=idx,
            text_field="machine_text",
            domain="detectrl",
            task="multidomain",
            ai_model=None,
        )


def _load_m4(data_dir: str) -> Iterator[EvalRecord]:
    """Load M4 dataset - separate human/machine JSON files."""
    m4_dir = Path(data_dir) / "M4"
    human_path = m4_dir / "M4_human_test.json"
    machine_path = m4_dir / "M4_machine_test.json"

    # Load human texts
    with open(human_path) as f:
        human_data = json.load(f)
    human_texts = human_data["human_text"]

    for idx, text in enumerate(human_texts):
        yield EvalRecord(
            text=text,
            ground_truth_label=0,
            source_file=str(human_path),
            line_index=idx,
            text_field="human_text",
            domain="m4",
            task="detection",
            ai_model=None,
        )

    # Load machine texts
    with open(machine_path) as f:
        machine_data = json.load(f)
    machine_texts = machine_data["machine_text"]

    for idx, text in enumerate(machine_texts):
        yield EvalRecord(
            text=text,
            ground_truth_label=1,
            source_file=str(machine_path),
            line_index=idx,
            text_field="machine_text",
            domain="m4",
            task="detection",
            ai_model=None,
        )


def _load_raid(
    data_dir: str,
    split: str = "extra",
    max_samples: int = 2000,
    include_adversarial: bool = False,
) -> Iterator[EvalRecord]:
    """Load RAID benchmark dataset.

    Args:
        data_dir: Base data directory (unused, RAID uses its own cache)
        split: RAID split to use - "extra" (OOD) or "train" (in-distribution)
        max_samples: Max samples to load, stratified 50/50 human/AI (default: 2000)
        include_adversarial: Whether to include adversarial attacks (default: False)

    Yields:
        EvalRecord for each text sample
    """
    from raid.utils import load_data

    df = load_data(split=split, include_adversarial=include_adversarial)

    # Separate human and AI records
    human_df = df[df["model"] == "human"]
    ai_df = df[df["model"] != "human"]

    # Stratified sampling: 50% human, 50% AI
    samples_per_class = max_samples // 2

    if len(human_df) > samples_per_class:
        human_df = human_df.sample(n=samples_per_class, random_state=42)
    if len(ai_df) > samples_per_class:
        ai_df = ai_df.sample(n=samples_per_class, random_state=42)

    # Combine and iterate
    sampled_df = pd.concat([human_df, ai_df], ignore_index=True)

    for idx, row in sampled_df.iterrows():
        is_human = row["model"] == "human"
        yield EvalRecord(
            text=row["generation"],
            ground_truth_label=0 if is_human else 1,
            source_file=f"raid:{split}",
            line_index=idx,
            text_field="generation",
            domain=row["domain"],
            task=row.get("attack", "none") or "none",
            ai_model=None if is_human else row["model"],
        )


if __name__ == "__main__":
    # Quick test
    for dataset in DATASETS:
        try:
            records = list(load_dataset(dataset))
            human = sum(1 for r in records if r.ground_truth_label == 0)
            ai = sum(1 for r in records if r.ground_truth_label == 1)
            print(f"{dataset}: {len(records)} records ({human} human, {ai} AI)")

            if records:
                r = records[0]
                print(
                    f"  Sample: {r.text[:60]}... | label={r.ground_truth_label} | task={r.task}"
                )
            print()
        except FileNotFoundError as e:
            print(f"{dataset}: MISSING - {e}")
            print()
