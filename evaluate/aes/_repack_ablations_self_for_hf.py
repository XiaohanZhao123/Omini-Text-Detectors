#!/usr/bin/env python3
"""Repack the 6 self-train variants' ablation eval results into HF layout.

Source: results/aes_doc_eval_ablations_self/
  - infer_damasha outputs:    <variant>/<slice>/<domain>/{predictions.jsonl,summary.json,run_config.json}
  - infer_gigacheck outputs:  <variant>/<slice>/<domain>/...   (similar)
  - eval_finetuned outputs:   gl-clic-simplified-qwen/<slice>/gl-clic_<ts>/<csv-stem>/...
  - eval_doc_level outputs:   seqxgpt-sondos_<slice>_<field>_gemini-2.5-flash_<ts>/...

Output: results/aes_doc_eval_ablations_self_hf_staging/
  - <namespace>/<gran>/<detector>/<cell>/{predictions.jsonl,summary.json,run_config.json}

Detector → (namespace, granularity, hf_folder_name) mapping. All 6 are
self-trained variants → tuned_on_new_data/. Folder naming reuses existing
HF folders where ckpt matches; creates new folders for novel ckpts.

  damasha-lora-v2          → tuned_on_new_data/token/damasha/         (existing v2-LoRA folder)
  damasha-lora-qwen        → tuned_on_new_data/token/damasha-lora-qwen/  (new)
  gigacheck-lora-v2        → tuned_on_new_data/doc/gigacheck/         (existing v2-LoRA folder)
  gigacheck-lora-qwen      → tuned_on_new_data/doc/gigacheck-lora-qwen/  (new)
  gl-clic-simplified-qwen  → tuned_on_new_data/sentence/gl-clic-simplified-qwen/  (new)
  seqxgpt-sondos           → tuned_on_new_data/sentence/seqxgpt-sondos/  (new)

Cell naming: tuned_on_new_data convention is singular domain.
Pattern: `<domain>_<slice>_gemini-2.5-flash`.
"""
import argparse
import json
import re
import shutil
from pathlib import Path


# (namespace, granularity, hf_folder_name) per local detector dir
DETECTOR_PLACEMENT = {
    "damasha-lora-v2":         ("tuned_on_new_data", "token",    "damasha"),
    "damasha-lora-qwen":       ("tuned_on_new_data", "token",    "damasha-lora-qwen"),
    "gigacheck-lora-v2":       ("tuned_on_new_data", "doc",      "gigacheck"),
    "gigacheck-lora-qwen":     ("tuned_on_new_data", "doc",      "gigacheck-lora-qwen"),
    "gl-clic-simplified-qwen": ("tuned_on_new_data", "sentence", "gl-clic-simplified-qwen"),
    "seqxgpt-sondos":          ("tuned_on_new_data", "sentence", "seqxgpt-sondos"),
}

DETECTOR_TRAINING_SOURCE = {
    "damasha-lora-v2": (
        "HuggingFace base saiteja33/DAMASHA-RMC + LoRA r=8 fine-tuned on "
        "Sondos v2 (gemini + gpt-5.4 + gpt-5.4-nano), 5 epochs"
    ),
    "damasha-lora-qwen": (
        "HuggingFace base saiteja33/DAMASHA-RMC + LoRA r=8 fine-tuned on "
        "Sondos v2 Qwen3-8B subset (task 2 continual fine-tune), 3 epochs"
    ),
    "gigacheck-lora-v2": (
        "HuggingFace base iitolstykh/GigaCheck-Detector-Multi (Mistral-7B + DETR) + "
        "LoRA r=8 fine-tuned on Sondos v2 (gemini + gpt-5.4 + gpt-5.4-nano), 5 epochs"
    ),
    "gigacheck-lora-qwen": (
        "HuggingFace base iitolstykh/GigaCheck-Detector-Multi + LoRA r=8 fine-tuned on "
        "Sondos v2 Qwen3-8B subset (task 2 continual fine-tune), 3 epochs"
    ),
    "gl-clic-simplified-qwen": (
        "DeBERTa-v3-base + LoRA + GRU trained from scratch on Sondos v2 Qwen3-8B subset "
        "(simplified arch, our trainer), 3 epochs"
    ),
    "seqxgpt-sondos": (
        "ModelWiseTransformerClassifier (CNN+Transformer+CRF) trained from scratch on "
        "Sondos v2 4-LLM log-prob features (binary AI/human BMES, 8 classes), our trainer"
    ),
}

ABLATION_DESCRIPTIONS = {
    "ablation1_paraphrase": (
        "Ablation 1 (covctrl): operation FIXED at paraphrase, AI coverage VARIED across "
        "{0%, 25%, 50%, 75%, 100%} — version column values cov00..cov100"
    ),
    "ablation1_compress": (
        "Ablation 1 (covctrl): operation FIXED at compress, AI coverage VARIED — cov00..cov100"
    ),
    "ablation1_expand": (
        "Ablation 1 (covctrl): operation FIXED at expand, AI coverage VARIED — cov00..cov100"
    ),
    "ablation2": (
        "Ablation 2 (opctrl): AI coverage levels FIXED, operation VARIED — versions are "
        "{base, paraphrase_25/50/75, compress_25/50/75, expand_25/50/75}"
    ),
    "ablation3": (
        "Ablation 3 (non-cumulative): each version v0..v8 generated INDEPENDENTLY from v0 "
        "with the same target coverage and operation as the cumulative trajectory"
    ),
}

FIELD_TO_SINGULAR = {
    "abstracts": "abstract", "abstract": "abstract",
    "essays": "essay", "essay": "essay",
    "newss": "news", "news": "news",
    "reports": "report", "report": "report",
}

GENERATOR_TAG = "gemini-2.5-flash"
ALLOWED_SLICES = {"ablation1_paraphrase", "ablation1_compress", "ablation1_expand",
                  "ablation2", "ablation3"}


# Pattern for eval_doc_level outputs (seqxgpt-sondos):
#   seqxgpt-sondos_<slice>_<field>_gemini-2.5-flash_<ts>/
EVAL_DOC_RE = re.compile(
    r"^(?P<det>seqxgpt-sondos)_"
    r"(?P<slice>ablation\d+(?:_\w+)?)_"
    r"(?P<field>abstracts|essays|newss|news|reports)_"
    rf"{re.escape(GENERATOR_TAG)}_"
    r"(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$"
)


def _load_json(p: Path):
    with open(p) as f:
        return json.load(f)


def _dump_json(obj, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _inject_ablation_metadata(run_config, detector, slice_name, domain, source_path):
    rc = dict(run_config)
    rc["transfer_setup"] = {
        "experiment": "ablation_study",
        "ablation_slice": slice_name,
        "ablation_description": ABLATION_DESCRIPTIONS.get(slice_name, "(unknown)"),
        "eval_target_generator": "gemini-2.5-flash",
        "eval_target_data": f"Sondos v2 ablations — {slice_name}",
        "domain": domain,
        "training_source": DETECTOR_TRAINING_SOURCE.get(
            detector, "(unknown — add to DETECTOR_TRAINING_SOURCE)"
        ),
        "transfer_note": (
            "Eval on a controlled-design ablation slice. "
            "This checkpoint is one of our self-trained or fine-tuned variants."
        ),
        "source_local_run_dir": str(source_path),
    }
    return rc


# ---------------------------------------------------------------------------
# Source collectors
# ---------------------------------------------------------------------------

def collect_infer_outputs(results_dir: Path, detector: str):
    """Collect infer_damasha / infer_gigacheck outputs.

    Layout: <detector>/<slice>/<csv-stem>/{predictions, summary, run_config}
    where <csv-stem> is e.g. `abstracts_covctrl_paraphrase_gemini-2.5-flash`.
    """
    base = results_dir / detector
    if not base.exists():
        return
    for slice_dir in sorted(base.iterdir()):
        if not slice_dir.is_dir():
            continue
        slice_name = slice_dir.name
        if slice_name not in ALLOWED_SLICES:
            continue
        for sub in sorted(slice_dir.iterdir()):
            if not sub.is_dir():
                continue
            if not (sub / "summary.json").exists() or not (sub / "predictions.jsonl").exists():
                continue
            field = None
            for f in ("abstracts", "essays", "news", "reports"):
                if sub.name.startswith(f):
                    field = f
                    break
            if field is None:
                continue
            yield (detector, slice_name, FIELD_TO_SINGULAR[field], sub)


def collect_glclic_qwen(results_dir: Path):
    """gl-clic-simplified-qwen layout (eval_finetuned_detectors):
        gl-clic-simplified-qwen/<slice>/gl-clic_<ts>/<csv-stem>/...
    """
    base = results_dir / "gl-clic-simplified-qwen"
    if not base.exists():
        return
    for slice_dir in sorted(base.iterdir()):
        if not slice_dir.is_dir():
            continue
        slice_name = slice_dir.name
        if slice_name not in ALLOWED_SLICES:
            continue
        for ts_dir in sorted(slice_dir.iterdir()):
            if not ts_dir.is_dir():
                continue
            for sub in sorted(ts_dir.iterdir()):
                if not sub.is_dir():
                    continue
                if not (sub / "summary.json").exists() or not (sub / "predictions.jsonl").exists():
                    continue
                field = None
                for f in ("abstracts", "essays", "news", "reports"):
                    if sub.name.startswith(f):
                        field = f
                        break
                if field is None:
                    continue
                yield ("gl-clic-simplified-qwen", slice_name,
                       FIELD_TO_SINGULAR[field], sub)


def collect_seqxgpt_sondos(results_dir: Path):
    """seqxgpt-sondos layout (eval_doc_level): flat <det>_<slice>_<field>_<gen>_<ts>/."""
    for d in sorted(results_dir.glob("seqxgpt-sondos_*_gemini-2.5-flash_*")):
        if not d.is_dir():
            continue
        m = EVAL_DOC_RE.match(d.name)
        if not m:
            continue
        if m.group("slice") not in ALLOWED_SLICES:
            continue
        if not (d / "summary.json").exists() or not (d / "predictions.jsonl").exists():
            continue
        yield (m.group("det"), m.group("slice"),
               FIELD_TO_SINGULAR[m.group("field")], d)


def repack(results_dir: Path, staging_dir: Path, dry_run: bool = False):
    runs = []
    runs += list(collect_infer_outputs(results_dir, "damasha-lora-v2"))
    runs += list(collect_infer_outputs(results_dir, "damasha-lora-qwen"))
    runs += list(collect_infer_outputs(results_dir, "gigacheck-lora-v2"))
    runs += list(collect_infer_outputs(results_dir, "gigacheck-lora-qwen"))
    runs += list(collect_glclic_qwen(results_dir))
    runs += list(collect_seqxgpt_sondos(results_dir))
    runs.sort()
    print(f"Found {len(runs)} self-train ablation source runs")

    if staging_dir.exists() and not dry_run:
        print(f"[warn] wiping existing staging dir {staging_dir}")
        shutil.rmtree(staging_dir)

    seen, dups, unknown = set(), [], []
    by_namespace = {"tuned_on_new_data": 0}

    for detector, slice_name, domain, src in runs:
        placement = DETECTOR_PLACEMENT.get(detector)
        if placement is None:
            unknown.append((detector, slice_name, domain, src))
            continue
        ns, gran, hf_folder = placement
        cell = f"{domain}_{slice_name}_{GENERATOR_TAG}"

        dst = staging_dir / ns / gran / hf_folder / cell
        key = (ns, gran, hf_folder, cell)
        if key in seen:
            dups.append((detector, slice_name, domain, src))
            continue
        seen.add(key)

        if dry_run:
            print(f"  DRY: {detector}/{slice_name}/{domain}  ->  {dst.relative_to(staging_dir)}")
            continue

        dst.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / "predictions.jsonl", dst / "predictions.jsonl")
        shutil.copyfile(src / "summary.json",      dst / "summary.json")
        # eval_doc_level writes run_config.json; infer_damasha / infer_gigacheck
        # don't. Synthesize a minimal one when missing so every cell has 3 files.
        rc_path = src / "run_config.json"
        if rc_path.exists():
            rc = _load_json(rc_path)
        else:
            rc = {
                "detector": detector,
                "field": domain,
                "model_short": "gemini-2.5-flash",
                "split": "test",
                "device": "cuda:0",
                "max_samples": None,
                "csv_path": None,  # multi-CSV inference combines 4 domain files
                "timestamp": None,
                "git_commit": None,
                "yaml_config": {
                    "model": detector,
                    # the actual ckpt path is encoded in the wrapper script,
                    # included in transfer_setup.training_source below
                },
            }
        rc = _inject_ablation_metadata(rc, detector, slice_name, domain, src)
        _dump_json(rc, dst / "run_config.json")
        by_namespace[ns] += 1
        print(f"  packed: {dst.relative_to(staging_dir)}")

    print(f"\nPacked {len(seen)} cells:")
    for ns, n in by_namespace.items():
        print(f"  {ns}: {n}")
    if dups:
        print(f"[warn] {len(dups)} duplicates:")
        for d in dups:
            print(f"   {d}")
    if unknown:
        print(f"[warn] {len(unknown)} unknown detector mapping:")
        for d in unknown:
            print(f"   {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/aes_doc_eval_ablations_self")
    ap.add_argument("--staging-dir", default="results/aes_doc_eval_ablations_self_hf_staging")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    repack(Path(args.results_dir), Path(args.staging_dir), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
