#!/usr/bin/env python3
"""Repack ablation eval results into the HF `<namespace>/<gran>/<detector>/<cell>` layout.

Source: results/aes_doc_eval_ablations/
  - eval_doc_level outputs:  <det>_<slice>_<field>_gemini-2.5-flash_<ts>/
                             where field ∈ {abstracts, essays, newss, news, reports}
                             (note: "newss" is a known prepare bug we normalize to "news")
  - eval_finetuned outputs:  gl-clic-simplified/<slice>/<csv-stem>/

Output: results/aes_doc_eval_ablations_hf_staging/
  - <namespace>/<gran>/<detector>/<cell>/{predictions.jsonl, summary.json, run_config.json}

Cell naming convention (matches existing OOD/Qwen pattern in HF repo):
  - zero_shot_methods: <plural domain>_<slice>_gemini-2.5-flash
      e.g.  zero_shot_methods/sentence/adaloc/abstracts_ablation1_paraphrase_gemini-2.5-flash/
  - tuned_on_new_data: <singular domain>_<slice>_gemini-2.5-flash
      e.g.  tuned_on_new_data/sentence/seqxgpt/abstract_ablation2_gemini-2.5-flash/

Per-detector (namespace, granularity) — same as Qwen OOD batch:
  adaloc      → zero_shot_methods / sentence/   (paper-original ckpt)
  damasha     → zero_shot_methods / token/      (HF base, no LoRA)
  gigacheck   → zero_shot_methods / span/       (HF base, no LoRA)
  gl-clic-simplified → tuned_on_new_data / sentence/   (our self-trained)
  seqxgpt     → tuned_on_new_data / sentence/ (local SeqXGPT checkpoint)
  sendetex    → tuned_on_new_data / sentence/ (our self-trained, partial)
"""
import argparse
import json
import re
import shutil
from pathlib import Path


# (namespace, granularity) per detector — must match the OOD repack file's mapping
DETECTOR_PLACEMENT = {
    "adaloc":             ("zero_shot_methods",   "sentence"),
    "damasha":            ("zero_shot_methods",   "token"),
    "gigacheck":          ("zero_shot_methods",   "span"),
    "gl-clic-simplified": ("tuned_on_new_data", "sentence"),
    "gl-clic":            ("tuned_on_new_data", "sentence"),  # alias → simplified
    "seqxgpt":            ("tuned_on_new_data", "sentence"),
    "sendetex":           ("tuned_on_new_data", "sentence"),
}

DETECTOR_TRAINING_SOURCE = {
    "adaloc": (
        "paper-original checkpoint (GoodNews + gpt2-xl, released 2024-03)"
    ),
    "damasha": (
        "HuggingFace base saiteja33/DAMASHA-RMC (RoBERTa + ModernBERT + CRF), "
        "as published — NOT fine-tuned on OpAI-Bench nor on these ablation generators"
    ),
    "gigacheck": (
        "HuggingFace base iitolstykh/GigaCheck-Detector-Multi (Mistral-7B + DETR), "
        "as published — NOT fine-tuned on OpAI-Bench nor on these ablation generators"
    ),
    "sendetex": (
        "trained from scratch on SeqXGPT-Bench (LLaMA-7B proxy), "
        "NOT trained on OpAI-Bench nor on these ablation generators"
    ),
    "seqxgpt": (
        "Local SeqXGPT checkpoint placeholder "
        "(trained on SeqXGPT-Bench), NOT trained on OpAI-Bench nor on these ablation generators"
    ),
    "gl-clic-simplified": (
        "DeBERTa-v3-base + LoRA + GRU trained from scratch on SeqXGPT-Bench, "
        "simplified trainer (not the full IJCNLP-AACL 2025 arch), "
        "NOT trained on OpAI-Bench nor on these ablation generators"
    ),
}

ABLATION_DESCRIPTIONS = {
    "ablation1_paraphrase": (
        "Ablation 1 (covctrl): operation FIXED at paraphrase, AI coverage VARIED across "
        "{0%, 25%, 50%, 75%, 100%} — version column values cov00..cov100"
    ),
    "ablation1_compress": (
        "Ablation 1 (covctrl): operation FIXED at compress, AI coverage VARIED across "
        "{0%, 25%, 50%, 75%, 100%} — version column values cov00..cov100"
    ),
    "ablation1_expand": (
        "Ablation 1 (covctrl): operation FIXED at expand, AI coverage VARIED across "
        "{0%, 25%, 50%, 75%, 100%} — version column values cov00..cov100"
    ),
    "ablation2": (
        "Ablation 2 (opctrl): AI coverage levels FIXED, operation VARIED — version values "
        "are {base, paraphrase_25, compress_25, expand_25, paraphrase_50, compress_50, "
        "expand_50, paraphrase_75, compress_75, expand_75}"
    ),
    "ablation3": (
        "Ablation 3 (non-cumulative): each version v0..v8 generated INDEPENDENTLY from v0 "
        "with the same target coverage and operation as the cumulative trajectory — "
        "for direct comparison against tuned_on_new_data v2 cumulative cells"
    ),
}


# Domain normalization
FIELD_TO_SINGULAR = {
    "abstracts": "abstract", "abstract": "abstract",
    "essays": "essay", "essay": "essay",
    "newss": "news", "news": "news",
    "reports": "report", "report": "report",
}
FIELD_TO_PLURAL = {
    "abstract": "abstracts", "abstracts": "abstracts",
    "essay": "essays", "essays": "essays",
    "news": "news", "newss": "news",
    "report": "reports", "reports": "reports",
}

GENERATOR_TAG = "gemini-2.5-flash"
ALLOWED_SLICES = {"ablation1_paraphrase", "ablation1_compress", "ablation1_expand",
                  "ablation2", "ablation3"}


# Pattern for eval_doc_level outputs
EVAL_DOC_RE = re.compile(
    r"^(?P<det>adaloc|damasha|gigacheck|seqxgpt|sendetex)_"
    r"(?P<slice>ablation\d+(?:_\w+)?)_"
    r"(?P<field>abstracts|essays|newss|news|reports)_"
    rf"{re.escape(GENERATOR_TAG)}_"
    r"(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$"
)


def _cell_name(namespace: str, domain_singular: str, slice_name: str) -> str:
    if namespace == "zero_shot_methods":
        dom = FIELD_TO_PLURAL[domain_singular]
    else:
        dom = domain_singular
    return f"{dom}_{slice_name}_{GENERATOR_TAG}"


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
        "ablation_description": ABLATION_DESCRIPTIONS.get(slice_name, "(unknown slice)"),
        "eval_target_generator": "gemini-2.5-flash",
        "eval_target_data": f"OpAI-Bench ablations — {slice_name}",
        "domain": domain,
        "training_source": DETECTOR_TRAINING_SOURCE.get(detector,
            "(unknown — add to DETECTOR_TRAINING_SOURCE)"),
        "transfer_note": (
            "Eval on a controlled-design ablation slice. The detector was not "
            "trained on OpAI-Bench or these ablation generators."
        ),
        "source_local_run_dir": str(source_path),
    }
    return rc


def collect_eval_doc(results_dir: Path):
    """Yield (detector, slice, domain_singular, src_dir) for eval_doc_level outputs."""
    for d in sorted(results_dir.glob("*_gemini-2.5-flash_*")):
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


def collect_glclic(results_dir: Path):
    """Yield (detector, slice, domain_singular, src_dir) for gl-clic-simplified outputs.

    Output layout from eval_finetuned_detectors.py is:
        gl-clic-simplified/<slice>/gl-clic_<ts>/<csv-stem>/{predictions, summary, run_config}
    """
    base = results_dir / "gl-clic-simplified"
    if not base.exists():
        return
    for slice_dir in sorted(base.iterdir()):
        if not slice_dir.is_dir():
            continue
        slice_name = slice_dir.name
        if slice_name not in ALLOWED_SLICES:
            continue
        # Descend one level into the timestamped wrapper dir(s).
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
                yield ("gl-clic-simplified", slice_name,
                       FIELD_TO_SINGULAR[field], sub)


def repack(results_dir: Path, staging_dir: Path, dry_run: bool = False):
    runs = list(collect_eval_doc(results_dir)) + list(collect_glclic(results_dir))
    runs.sort()
    print(f"Found {len(runs)} ablation source runs")

    if staging_dir.exists() and not dry_run:
        print(f"[warn] wiping existing staging dir {staging_dir}")
        shutil.rmtree(staging_dir)

    seen, dups, unknown = set(), [], []
    by_namespace = {"zero_shot_methods": 0, "tuned_on_new_data": 0}

    for detector, slice_name, domain, src in runs:
        placement = DETECTOR_PLACEMENT.get(detector)
        if placement is None:
            unknown.append((detector, slice_name, domain, src))
            continue
        ns, gran = placement
        cell = _cell_name(ns, domain, slice_name)
        dst = staging_dir / ns / gran / detector / cell
        key = (ns, gran, detector, cell)
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
        rc = _load_json(src / "run_config.json")
        rc = _inject_ablation_metadata(rc, detector, slice_name, domain, src)
        _dump_json(rc, dst / "run_config.json")
        by_namespace[ns] += 1
        print(f"  packed: {dst.relative_to(staging_dir)}")

    print(f"\nPacked {len(seen)} cells:")
    for ns, n in by_namespace.items():
        print(f"  {ns}: {n}")
    if dups:
        print(f"[warn] {len(dups)} duplicate sources:")
        for d in dups:
            print(f"   {d}")
    if unknown:
        print(f"[warn] {len(unknown)} sources with unknown detector mapping:")
        for d in unknown:
            print(f"   {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/aes_doc_eval_ablations")
    ap.add_argument("--staging-dir", default="results/aes_doc_eval_ablations_hf_staging")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    repack(Path(args.results_dir), Path(args.staging_dir), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
