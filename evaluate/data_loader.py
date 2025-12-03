"""Data loader for evaluation datasets.

Loads and flattens datasets into EvalRecord format for detector evaluation.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

DATASETS = ["education", "enron", "privacy"]


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


def load_dataset(name: str, data_dir: str = "data/") -> Iterator[EvalRecord]:
    """Load and flatten a single dataset.

    Args:
        name: Dataset name - one of "education", "enron", "privacy"
        data_dir: Base data directory path

    Yields:
        EvalRecord for each text sample (both human and AI)
    """
    if name == "education":
        yield from _load_education(data_dir)
    elif name == "enron":
        yield from _load_enron(data_dir)
    elif name == "privacy":
        yield from _load_privacy(data_dir)
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
