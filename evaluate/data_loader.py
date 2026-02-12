"""Data loader for evaluation datasets.

Loads and flattens datasets into EvalRecord format for detector evaluation.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

DATASETS = [
    "education",
    "enron",
    "privacy",
    "detectrl",
    "m4",
    "raid",
    "raid_train",
    # ICML/ACL-standard benchmarks
    "hc3",           # HC3: Human ChatGPT Comparison Corpus (EMNLP 2023)
    "turingbench",   # TuringBench (EMNLP 2021)
    # Multi-round AI-human editing chains
    "aes_chains",    # AES Chains: progressive AI editing (v0=human, v1-v3=AI-edited)
    "aes_chains_sentences",  # AES Chains sentence-level: per-sentence classification
]


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
    elif name == "hc3":
        yield from _load_hc3(data_dir, **kwargs)
    elif name == "turingbench":
        yield from _load_turingbench(data_dir, **kwargs)
    elif name == "aes_chains":
        yield from _load_aes_chains(**kwargs)
    elif name == "aes_chains_sentences":
        yield from _load_aes_chains_sentences(data_dir, **kwargs)
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
    max_samples: int = None,
    include_adversarial: bool = False,
) -> Iterator[EvalRecord]:
    """Load RAID benchmark dataset.

    Args:
        data_dir: Base data directory (unused, RAID uses its own cache)
        split: RAID split to use - "extra" (OOD), "test", or "train" (in-distribution)
        max_samples: Max samples to load (None = full dataset, or int for stratified 50/50)
        include_adversarial: Whether to include adversarial attacks (default: False)

    Yields:
        EvalRecord for each text sample

    Requires: pip install raid-bench
    """
    try:
        from raid.utils import load_data
    except ImportError:
        raise ImportError(
            "RAID benchmark requires raid-bench package. Install with: pip install raid-bench"
        )

    df = load_data(split=split, include_adversarial=include_adversarial)

    # Separate human and AI records
    human_df = df[df["model"] == "human"]
    ai_df = df[df["model"] != "human"]

    # Stratified sampling if max_samples is specified
    if max_samples is not None:
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


def _load_hc3(
    data_dir: str,
    subset: str = "all",
    max_samples: int = None,
    source: str = "huggingface",
) -> Iterator[EvalRecord]:
    """Load HC3 (Human ChatGPT Comparison Corpus) benchmark.

    Paper: "How Close is ChatGPT to Human Experts? Comparison Corpus,
           Evaluation, and Detection" (EMNLP 2023 Findings)
    URL: https://arxiv.org/abs/2301.07597
    HuggingFace: https://huggingface.co/datasets/Hello-SimpleAI/HC3

    The dataset contains question-answer pairs with both human and ChatGPT
    responses across domains: finance, medicine, open_qa, reddit_eli5, wiki_csai.

    Args:
        data_dir: Base data directory (unused when source="huggingface")
        subset: HC3 subset - "all", "finance", "medicine", "open_qa",
                "reddit_eli5", or "wiki_csai" (default: "all")
        max_samples: Maximum samples per class (human/AI) (default: None = all)
        source: "huggingface" to load from HF, "local" for local files

    Yields:
        EvalRecord for each text sample (human answers and ChatGPT answers)
    """
    if source == "huggingface":
        try:
            from datasets import load_dataset

            ds = load_dataset("Hello-SimpleAI/HC3", subset, split="train")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load HC3 from HuggingFace: {e}. "
                "Install with: pip install datasets"
            )
    else:
        # Local loading from Parquet files
        hc3_dir = Path(data_dir) / "HC3"
        parquet_path = hc3_dir / f"{subset}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"HC3 data not found at {parquet_path}. "
                "Download from https://huggingface.co/datasets/Hello-SimpleAI/HC3"
            )
        ds = pd.read_parquet(parquet_path).to_dict("records")

    human_count = 0
    ai_count = 0

    for idx, item in enumerate(ds):
        question = item.get("question", "")
        human_answers = item.get("human_answers", [])
        chatgpt_answers = item.get("chatgpt_answers", [])
        domain = item.get("source", subset)

        # Yield human answers
        for ans_idx, answer in enumerate(human_answers):
            if max_samples and human_count >= max_samples:
                break
            if not answer or not answer.strip():
                continue

            yield EvalRecord(
                text=answer,
                ground_truth_label=0,
                source_file=f"hc3:{subset}",
                line_index=idx,
                text_field=f"human_answers[{ans_idx}]",
                domain=domain,
                task="qa",
                ai_model=None,
            )
            human_count += 1

        # Yield ChatGPT answers
        for ans_idx, answer in enumerate(chatgpt_answers):
            if max_samples and ai_count >= max_samples:
                break
            if not answer or not answer.strip():
                continue

            yield EvalRecord(
                text=answer,
                ground_truth_label=1,
                source_file=f"hc3:{subset}",
                line_index=idx,
                text_field=f"chatgpt_answers[{ans_idx}]",
                domain=domain,
                task="qa",
                ai_model="chatgpt",
            )
            ai_count += 1

        # Check if we've reached max for both classes
        if max_samples and human_count >= max_samples and ai_count >= max_samples:
            break


def _load_turingbench(
    data_dir: str,
    task: str = "TT",
    generator: str = "gpt3",
    split: str = "test",
    max_samples: int = None,
) -> Iterator[EvalRecord]:
    """Load TuringBench benchmark dataset.

    Paper: "TuringBench: A Benchmark Environment for Turing Test in the
           Age of Neural Text Generation" (EMNLP 2021 Findings)
    URL: https://arxiv.org/abs/2109.13296
    HuggingFace: https://huggingface.co/datasets/turingbench/TuringBench

    TuringBench contains ~200K articles with human and machine-generated text
    from 19 different language models for Turing Test evaluation.

    Dataset structure: TuringBench/<task>_<generator>/<split>.csv
    CSV format: text,label (where label is "human" or model name)

    Available tasks:
    - TT: Turing Test (binary human vs machine)
    - AA: Authorship Attribution (multi-class)

    Available generators for TT task:
    - gpt1, gpt2_small, gpt2_medium, gpt2_large, gpt2_xl, gpt2_pytorch
    - gpt3, grover_base, grover_large, grover_mega
    - ctrl, xlm, xlnet_base, xlnet_large, fair_wmt19, fair_wmt20
    - transformer_xl, pplm_distil, pplm_gpt2

    Args:
        data_dir: Base data directory containing TuringBench folder
        task: Task type - "TT" for Turing Test (default: "TT")
        generator: Generator model for TT task (default: "gpt3")
        split: Data split - "train", "valid", or "test" (default: "test")
        max_samples: Maximum samples per class (default: None = all)

    Yields:
        EvalRecord for each text sample
    """
    import csv

    # Construct path: TuringBench/TT_gpt3/test.csv
    tb_dir = Path(data_dir) / "TuringBench" / f"{task}_{generator}"
    csv_path = tb_dir / f"{split}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"TuringBench data not found at {csv_path}. "
            "Download TuringBench.zip from "
            "https://huggingface.co/datasets/turingbench/TuringBench/tree/main "
            "and extract to data/TuringBench/"
        )

    human_count = 0
    ai_count = 0

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header row

        for idx, row in enumerate(reader):
            if len(row) < 2:
                continue

            text = row[0]
            label = row[1]

            if not text or not text.strip():
                continue

            is_human = label.lower() == "human"

            # Check max samples
            if max_samples:
                if is_human and human_count >= max_samples:
                    continue
                if not is_human and ai_count >= max_samples:
                    continue

            yield EvalRecord(
                text=text,
                ground_truth_label=0 if is_human else 1,
                source_file=str(csv_path),
                line_index=idx,
                text_field="text",
                domain="news",  # TuringBench primarily uses news articles
                task=f"{task}_{generator}",
                ai_model=None if is_human else label,
            )

            if is_human:
                human_count += 1
            else:
                ai_count += 1

            # Check if we've reached max for both classes
            if max_samples and human_count >= max_samples and ai_count >= max_samples:
                break


def _load_aes_chains(
    version: str = "all",
    data_path: str = "/data/spiderman/jiachengl/detect/aes_chains_pilot_aligned.jsonl",
) -> Iterator[EvalRecord]:
    """Load AES Chains multi-round AI-human editing dataset.

    Each document has 4 versions:
    - v0: Original human draft (ai_ratio=0.0)
    - v1: First AI edit (mean ai_ratio~0.34)
    - v2: Second AI edit (mean ai_ratio~0.56)
    - v3: Third AI edit (mean ai_ratio~0.65)

    For binary classification:
    - v0 always yields a "human" record (label=0)
    - vN (N>0) yields an "AI-assisted" record (label=1)

    Args:
        version: Which AI version to compare against v0.
                 "v1", "v2", "v3", or "all" (yields all versions).
                 Default: "all"
        data_path: Path to the aligned JSONL file.

    Yields:
        EvalRecord for each text sample
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"AES chains data not found at {data_path}")

    versions_to_load = ["v1", "v2", "v3"] if version == "all" else [version]

    with open(path) as f:
        for idx, line in enumerate(f):
            doc = json.loads(line)
            q_id = doc["q_id"]
            domain = doc["domain"]

            # Build version lookup
            version_map = {v["version_id"]: v for v in doc["history"]}

            # Always yield v0 (human) for each AI version we compare against
            v0 = version_map["v0"]
            for ver in versions_to_load:
                if ver not in version_map:
                    continue

                vn = version_map[ver]

                # Human record (v0)
                yield EvalRecord(
                    text=v0["text"],
                    ground_truth_label=0,
                    source_file=str(path),
                    line_index=idx,
                    text_field=f"v0_vs_{ver}",
                    domain=domain,
                    task=f"aes_{ver}",
                    ai_model=None,
                )

                # AI-edited record (vN)
                yield EvalRecord(
                    text=vn["text"],
                    ground_truth_label=1,
                    source_file=str(path),
                    line_index=idx,
                    text_field=ver,
                    domain=domain,
                    task=f"aes_{ver}",
                    ai_model=vn["operation"],
                )


def _split_sentences_with_labels(words, labels):
    """Split words into sentences while tracking per-word labels.

    Uses punctuation (.?!) followed by an uppercase word as sentence boundary.

    Returns:
        List of (sentence_words, sentence_labels) tuples.
    """
    sentences = []
    sw, sl = [], []
    for i, (word, label) in enumerate(zip(words, labels)):
        sw.append(word)
        sl.append(label)
        if re.search(r'[.?!]["\')\]]?$', word):
            is_end = i == len(words) - 1
            if not is_end:
                is_end = bool(re.match(r'^["\']?[A-Z]', words[i + 1]))
            if is_end:
                sentences.append((sw, sl))
                sw, sl = [], []
    if sw:
        sentences.append((sw, sl))
    return sentences


def _load_aes_chains_sentences(
    data_dir: str,
    data_path: str = None,
    ai_threshold: float = 0.5,
) -> Iterator[EvalRecord]:
    """Load AES Chains as sentence-level classification records.

    Splits each document version into sentences, labels each sentence based
    on the proportion of AI tokens it contains.

    Labeling rules:
    - 0% AI tokens -> human (label=0)
    - >ai_threshold AI tokens -> AI (label=1)
    - Otherwise -> dropped (ambiguous)

    Args:
        data_dir: Base data directory
        data_path: Override path to the JSONL file. If None, uses
            data_dir/aes_chains_pilot.jsonl
        ai_threshold: Fraction of AI tokens above which a sentence is
            labeled as AI. Default 0.5.

    Yields:
        EvalRecord for each usable sentence.
    """
    if data_path is None:
        path = Path(data_dir) / "aes_chains_pilot.jsonl"
    else:
        path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"AES chains data not found at {path}")

    with open(path) as f:
        for doc_idx, line in enumerate(f):
            doc = json.loads(line)
            q_id = doc["q_id"]
            domain = doc["domain"]
            version_map = {v["version_id"]: v for v in doc["history"]}

            for vid in ["v0", "v1", "v2", "v3"]:
                if vid not in version_map:
                    continue
                v = version_map[vid]
                words = v["text"].split()
                labels = v["token_labels"]
                if len(words) != len(labels):
                    continue

                sentences = _split_sentences_with_labels(words, labels)

                for sent_idx, (sent_words, sent_labels) in enumerate(sentences):
                    ai_count = sent_labels.count("ai")
                    total = len(sent_labels)
                    ai_ratio = ai_count / total

                    # Only keep pure human or clearly AI sentences
                    if ai_ratio == 0:
                        ground_truth = 0
                    elif ai_ratio > ai_threshold:
                        ground_truth = 1
                    else:
                        continue  # drop ambiguous

                    yield EvalRecord(
                        text=" ".join(sent_words),
                        ground_truth_label=ground_truth,
                        source_file=str(path),
                        line_index=doc_idx,
                        text_field=f"{vid}_s{sent_idx}",
                        domain=domain,
                        task=f"aes_sent_{vid}",
                        ai_model=v["operation"] if ground_truth == 1 else None,
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
