#!/usr/bin/env python3
"""Repack Qwen task-1 transfer-eval results into the HF repo's namespace layout.

Input:  results/aes_doc_eval_qwen/
          adaloc_qwen_abstracts_Qwen3-8B_<ts>/{predictions.jsonl, summary.json, run_config.json}
          damasha_qwen_abstracts_Qwen3-8B_<ts>/...
          gl-clic_<ts>/abstract/{predictions.jsonl, summary.json, run_config.json}
          ...

Output: results/aes_doc_eval_qwen_hf_staging/
          <namespace>/<granularity>/<detector>/<cell_name>/
            predictions.jsonl
            summary.json
            run_config.json

Namespace + granularity mapping follows the existing HF folders under
`<RESULTS_DATASET_REPO>/`:

  DETECTOR          NAMESPACE                  GRANULARITY  PROVENANCE
  -----------------------------------------------------------------------
  adaloc            zero_shot_methods            sentence/    paper-original ckpt
  damasha           tuned_on_new_data          token/       our v2-LoRA
  gigacheck         tuned_on_new_data          doc/         our v2-LoRA (moved from span/)
  gl-clic-simpli..  tuned_on_new_data          sentence/    our SeqXGPT-Bench train
  seqxgpt           tuned_on_new_data          sentence/    local SeqXGPT checkpoint
  sendetex          tuned_on_new_data          sentence/    our SeqXGPT-Bench train

Cell naming convention matches each namespace's existing style:
  - zero_shot_methods: <domains>_<generator>  (plural domain, `abstracts_qwen3-8b`)
  - tuned_on_new_data: <domain>_<generator> (singular domain, `abstract_qwen3-8b`)
    NOTE: tuned_on_new_data historically has cells WITHOUT generator suffix
    (in-distribution v2 eval). The `_qwen3-8b` suffix is a new convention
    introduced here to flag OOD-to-Qwen eval within that namespace.

Each run_config.json gets a `transfer_setup` block injected to clearly
document which checkpoint was used and that eval is on a held-out generator.

Usage:
    uv run python evaluate/aes/_repack_qwen_for_hf.py
        [--results-dir results/aes_doc_eval_qwen]
        [--staging-dir results/aes_doc_eval_qwen_hf_staging]
"""
import argparse
import json
import re
import shutil
from pathlib import Path


# (namespace, granularity) per detector.
#
# IMPORTANT: this mapping reflects *which ckpt was actually used at eval time*
# by the pipeline(), which is what `omini_text/configs/<detector>.yaml` points
# at. damasha + gigacheck configs point at the HuggingFace base (saiteja33
# / iitolstykh), NOT our OpAI-Bench-v2 LoRA adapters — so their task-1 OOD-on-Qwen
# result belongs in `zero_shot_methods/` (same-provenance as the existing 3
# generator cells), not `tuned_on_new_data/`.
DETECTOR_PLACEMENT = {
    "adaloc":             ("zero_shot_methods",   "sentence"),
    "damasha":            ("zero_shot_methods",   "token"),
    "gigacheck":          ("zero_shot_methods",   "span"),
    "gl-clic-simplified": ("tuned_on_new_data", "sentence"),
    "gl-clic":            ("tuned_on_new_data", "sentence"),  # alias → simplified
    "seqxgpt":            ("tuned_on_new_data", "sentence"),
    "sendetex":           ("tuned_on_new_data", "sentence"),
}


# Per-detector provenance: which checkpoint was used + what data trained it.
DETECTOR_TRAINING_SOURCE = {
    "adaloc": (
        "paper-original checkpoint (GoodNews + gpt2-xl, released 2024-03)"
    ),
    "damasha": (
        "HuggingFace base saiteja33/DAMASHA-RMC (RoBERTa + ModernBERT + CRF), "
        "as published — NOT fine-tuned on OpAI-Bench nor on Qwen"
    ),
    "gigacheck": (
        "HuggingFace base iitolstykh/GigaCheck-Detector-Multi (Mistral-7B + "
        "DETR), as published — NOT fine-tuned on OpAI-Bench nor on Qwen"
    ),
    "sendetex": (
        "trained from scratch on SeqXGPT-Bench (LLaMA-7B proxy), "
        "NOT trained on OpAI-Bench nor on Qwen"
    ),
    "seqxgpt": (
        "Local SeqXGPT checkpoint placeholder "
        "(trained on SeqXGPT-Bench), NOT trained on OpAI-Bench nor on Qwen"
    ),
    "gl-clic-simplified": (
        "DeBERTa-v3-base + LoRA + GRU trained from scratch on SeqXGPT-Bench, "
        "simplified trainer (not the full IJCNLP-AACL 2025 arch), "
        "NOT trained on OpAI-Bench nor on Qwen"
    ),
}


# Singular -> plural (for zero_shot_methods namespace style) and canonical singular.
DOMAIN_SINGULAR = {
    "abstract": "abstract", "abstracts": "abstract",
    "essay": "essay", "essays": "essay",
    "news": "news",
    "report": "report", "reports": "report",
}
DOMAIN_PLURAL = {
    "abstract": "abstracts", "abstracts": "abstracts",
    "essay": "essays", "essays": "essays",
    "news": "news",
    "report": "reports", "reports": "reports",
}

GENERATOR_TAG = "qwen3-8b"


def _cell_name(namespace: str, domain_raw: str) -> str:
    """Build the HF cell folder name per namespace style."""
    if namespace == "zero_shot_methods":
        dom = DOMAIN_PLURAL[domain_raw]
    else:
        dom = DOMAIN_SINGULAR[domain_raw]
    return f"{dom}_{GENERATOR_TAG}"


def _parse_doclevel_run_dir(d: Path):
    """Parse `{detector}_qwen_{field}_{model}_{ts}/` → (detector, domain_singular)."""
    m = re.match(r"^(?P<det>.+?)_qwen_(?P<field>[a-z]+)_Qwen3-8B_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$",
                 d.name)
    if not m:
        return None
    field = m.group("field")
    if field not in DOMAIN_SINGULAR:
        return None
    return (m.group("det"), DOMAIN_SINGULAR[field])


def _parse_ftuned_run_dir(d: Path):
    """Parse `{detector}_{ts}/{domain}/` top dir → detector."""
    m = re.match(r"^(?P<det>.+?)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", d.name)
    if not m:
        return None
    return m.group("det")


def _load_json(p: Path):
    with open(p) as f:
        return json.load(f)


def _dump_json(obj, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _inject_transfer_metadata(run_config: dict, detector: str, domain: str,
                              source_path: Path) -> dict:
    """Add transfer_setup block describing this is a Qwen held-out-generator eval."""
    run_config = dict(run_config)  # copy
    run_config["transfer_setup"] = {
        "eval_target_generator": "qwen/Qwen3-8B",
        "eval_target_data": "OpAI-Bench (Qwen3-8B generation, April 2026)",
        "domain": domain,
        "training_source": DETECTOR_TRAINING_SOURCE.get(
            detector, "(unknown — add to DETECTOR_TRAINING_SOURCE)"
        ),
        "transfer_note": (
            "Eval on unseen generator (qwen3-8b). This checkpoint has not "
            "been trained on Qwen3-8B data."
        ),
        "source_local_run_dir": str(source_path),
    }
    return run_config


def collect_doclevel_runs(results_dir: Path):
    """Yield (detector, domain_singular, src_dir) for eval_doc_level.py outputs."""
    for d in sorted(results_dir.glob("*_qwen_*_Qwen3-8B_*")):
        if not d.is_dir():
            continue
        parsed = _parse_doclevel_run_dir(d)
        if not parsed:
            continue
        detector, domain = parsed
        if (d / "summary.json").exists() and (d / "predictions.jsonl").exists():
            yield (detector, domain, d)


def collect_ftuned_runs(results_dir: Path):
    """Yield (detector, domain_singular, src_dir) for eval_finetuned_detectors.py outputs.

    We use `gl-clic-simplified` as the detector tag, rewiring the raw `gl-clic`
    detector name since the checkpoint is our simplified variant.
    """
    for top in sorted(results_dir.iterdir()):
        if not top.is_dir() or top.name.startswith("_") or "_qwen_" in top.name:
            continue
        detector = _parse_ftuned_run_dir(top)
        if not detector:
            continue
        if detector == "gl-clic":
            detector = "gl-clic-simplified"
        for sub in sorted(top.iterdir()):
            if not sub.is_dir():
                continue
            if sub.name not in DOMAIN_SINGULAR:
                continue
            domain = DOMAIN_SINGULAR[sub.name]
            if (sub / "summary.json").exists() and (sub / "predictions.jsonl").exists():
                yield (detector, domain, sub)


def repack(results_dir: Path, staging_dir: Path, dry_run: bool = False):
    runs = list(collect_doclevel_runs(results_dir)) + list(collect_ftuned_runs(results_dir))
    runs.sort()
    print(f"Found {len(runs)} complete source runs")

    if staging_dir.exists() and not dry_run:
        print(f"[warn] wiping existing staging dir {staging_dir}")
        shutil.rmtree(staging_dir)

    seen = set()
    duplicates = []
    unknown_detectors = []

    for detector, domain, src in runs:
        placement = DETECTOR_PLACEMENT.get(detector)
        if placement is None:
            unknown_detectors.append((detector, domain, src))
            continue
        namespace, gran = placement
        cell = _cell_name(namespace, domain)

        dst = staging_dir / namespace / gran / detector / cell
        key = (namespace, gran, detector, cell)
        if key in seen:
            duplicates.append((detector, domain, src))
            continue
        seen.add(key)

        if dry_run:
            print(f"  DRY: {detector}/{domain}  {src}  ->  {dst}")
            continue

        dst.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / "predictions.jsonl", dst / "predictions.jsonl")
        shutil.copyfile(src / "summary.json", dst / "summary.json")
        rc = _load_json(src / "run_config.json")
        rc = _inject_transfer_metadata(rc, detector, domain, src)
        _dump_json(rc, dst / "run_config.json")
        print(f"  packed: {dst.relative_to(staging_dir)}")

    print(f"\nPacked {len(seen)} cells")
    if duplicates:
        print(f"[warn] {len(duplicates)} duplicate source dirs (kept first):")
        for d in duplicates:
            print(f"   {d}")
    if unknown_detectors:
        print(f"[warn] {len(unknown_detectors)} detectors with no placement mapping:")
        for d in unknown_detectors:
            print(f"   {d}")

    if not dry_run:
        print("\n=== final staging tree ===")
        for p in sorted(staging_dir.rglob("*")):
            if p.is_file():
                rel = p.relative_to(staging_dir)
                sz = p.stat().st_size
                print(f"  {sz:>10d}  {rel}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/aes_doc_eval_qwen")
    ap.add_argument("--staging-dir", default="results/aes_doc_eval_qwen_hf_staging")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    repack(Path(args.results_dir), Path(args.staging_dir), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
