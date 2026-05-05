#!/usr/bin/env python3
"""Clean <RESULTS_DATASET_REPO> and upload the final calibrated
per-detector results to tuned_on_new_data/<detector>/.

Deletes:
  - calibration/                     (interim bookkeeping)
  - README_our_results.md            (interim writeup)
  - tuned_on_new_data/<det>/         for every detector we're replacing, so
                                      stale files don't linger

Uploads (from hf_upload_final/tuned_on_new_data/<det>/):
  - damasha, gigacheck, seqxgpt              (our fine-tunes)
  - genai-sentence, genai-sentence-v2        (HF baselines, now with calibrated doc labels)
  - gl-clic, gl-clic-v2                      (HF baselines, now with calibrated doc labels)
"""
import os, sys
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = os.environ.get("HF_RESULTS_REPO", "<RESULTS_DATASET_REPO>")
STAGING = Path("results/hf_upload_final")

DETECTORS = [
    "damasha", "gigacheck", "seqxgpt",
    "genai-sentence", "genai-sentence-v2",
    "gl-clic", "gl-clic-v2",
]


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Set HF_TOKEN", file=sys.stderr); sys.exit(1)
    if REPO_ID == "<RESULTS_DATASET_REPO>":
        print("Set HF_RESULTS_REPO to the target dataset repository", file=sys.stderr)
        sys.exit(1)
    api = HfApi(token=token)

    # 1. Delete stale top-level folders / files
    for target in ["calibration", "README_our_results.md"]:
        try:
            if target.endswith(".md") or "." in Path(target).suffix:
                api.delete_file(path_in_repo=target, repo_id=REPO_ID, repo_type="dataset",
                                commit_message=f"Remove interim {target}")
                print(f"[delete] {target}")
            else:
                api.delete_folder(path_in_repo=target, repo_id=REPO_ID, repo_type="dataset",
                                  commit_message=f"Remove interim {target}/")
                print(f"[delete] {target}/")
        except Exception as e:
            msg = str(e)
            if "not found" in msg.lower() or "404" in msg:
                print(f"[delete] {target}: already absent, skipping")
            else:
                print(f"[delete] {target}: {e}")

    # 2. Delete each detector's folder (so we don't leave stale files)
    for det in DETECTORS:
        try:
            api.delete_folder(
                path_in_repo=f"tuned_on_new_data/{det}",
                repo_id=REPO_ID, repo_type="dataset",
                commit_message=f"Clear stale tuned_on_new_data/{det}/ before calibrated re-upload",
            )
            print(f"[delete] tuned_on_new_data/{det}/")
        except Exception as e:
            msg = str(e)
            if "not found" in msg.lower() or "404" in msg:
                print(f"[delete] tuned_on_new_data/{det}/: already absent")
            else:
                print(f"[delete] tuned_on_new_data/{det}/: {e}")

    # 3. Upload the fresh versions
    for det in DETECTORS:
        src = STAGING / "tuned_on_new_data" / det
        if not src.exists():
            print(f"[upload] {det}: staging missing, skipping")
            continue
        total_bytes = sum(p.stat().st_size for p in src.rglob("*") if p.is_file())
        print(f"[upload] tuned_on_new_data/{det}/   ({total_bytes/1e6:.1f} MB)")
        api.upload_folder(
            folder_path=str(src),
            path_in_repo=f"tuned_on_new_data/{det}",
            repo_id=REPO_ID, repo_type="dataset",
            commit_message=f"Replace {det} with calibrated-threshold predictions on OpAI-Bench test",
        )

    print("\n[done] https://huggingface.co/datasets/" + REPO_ID)


if __name__ == "__main__":
    main()
