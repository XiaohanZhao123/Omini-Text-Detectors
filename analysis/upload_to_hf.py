#!/usr/bin/env python3
"""Upload staged results to HAT-Baselines/baseline_results dataset repo.

Uploads:
  tuned_on_new_data/{damasha, gigacheck, seqxgpt}/*
  calibration/*
  README_our_results.md
"""
import os, sys
from pathlib import Path
from huggingface_hub import HfApi, login

REPO_ID = "HAT-Baselines/baseline_results"
STAGING = Path("/datadrive/xiaohan/Omini-Text/results/hf_upload_staging")


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Set HF_TOKEN env var", file=sys.stderr)
        sys.exit(1)

    login(token=token, add_to_git_credential=False)
    api = HfApi(token=token)

    # Folder-by-folder upload so we can surface errors per detector
    for sub in ["tuned_on_new_data/damasha",
                "tuned_on_new_data/gigacheck",
                "tuned_on_new_data/seqxgpt",
                "calibration"]:
        src = STAGING / sub
        if not src.exists():
            print(f"skip {sub}: not in staging")
            continue
        print(f"[upload] {sub}  ({sum(p.stat().st_size for p in src.rglob('*') if p.is_file())/1e6:.1f} MB)")
        api.upload_folder(
            folder_path=str(src),
            path_in_repo=sub,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"Add {sub} (fine-tuned on Sondos v2 + test-split oracle threshold calibration)",
        )

    # Single-file uploads (top-level README)
    readme = STAGING / "README_our_results.md"
    if readme.exists():
        print(f"[upload] README_our_results.md")
        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README_our_results.md",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="Add summary README for the new fine-tuned detectors + calibration results",
        )

    print("\n[upload] done. Repo: https://huggingface.co/datasets/" + REPO_ID)


if __name__ == "__main__":
    main()
