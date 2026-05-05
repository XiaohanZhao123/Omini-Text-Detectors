#!/usr/bin/env python3
"""Download evaluation data from Google Drive.

Usage:
    python download_gdrive.py "https://drive.google.com/drive/folders/XXXXX"
    python download_gdrive.py "https://drive.google.com/drive/folders/XXXXX" --output_dir data/external/opai_bench
    python download_gdrive.py --inspect                    # inspect already-downloaded files

Requires: pip install gdown
"""

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_OUTPUT_DIR = "data/external/opai_bench"


def download(url: str, output_dir: str = DEFAULT_OUTPUT_DIR):
    """Download a Google Drive folder or file."""
    # Install gdown if needed
    try:
        import gdown
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading to {out.resolve()}...")
    if "folders/" in url:
        gdown.download_folder(url=url, output=str(out), quiet=False, use_cookies=False)
    else:
        gdown.download(url, str(out) + "/", quiet=False)

    # Show what we got
    inspect(output_dir)


def inspect(output_dir: str = DEFAULT_OUTPUT_DIR):
    """Show downloaded files and their structure."""
    out = Path(output_dir)
    if not out.exists():
        print(f"Not found: {output_dir}. Run download first.")
        return

    print(f"\n{'='*60}")
    print(f"Files in {out.resolve()}")
    print(f"{'='*60}")

    for f in sorted(out.rglob("*")):
        if not f.is_file():
            continue
        mb = f.stat().st_size / 1024 / 1024
        print(f"\n  {f.relative_to(out)} ({mb:.1f} MB)")

        # Preview CSV/JSONL
        suffix = f.suffix.lower()
        if suffix == ".csv":
            try:
                import pandas as pd
                df = pd.read_csv(f, nrows=2)
                print(f"    Columns: {list(df.columns)}")
                print(f"    Rows: ~{sum(1 for _ in open(f)) - 1}")
            except Exception as e:
                print(f"    [Cannot read: {e}]")
        elif suffix == ".jsonl":
            try:
                import json
                with open(f) as fh:
                    first = json.loads(fh.readline())
                    print(f"    Keys: {list(first.keys())}")
                    remaining = sum(1 for line in fh if line.strip())
                    print(f"    Lines: {remaining + 1}")
            except Exception as e:
                print(f"    [Cannot read: {e}]")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("url", nargs="?", default=None,
                   help="Google Drive URL (folder or file)")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output dir")
    p.add_argument("--inspect", action="store_true", help="Only inspect (no download)")
    args = p.parse_args()

    if args.inspect:
        inspect(args.output_dir)
    elif args.url:
        download(args.url, args.output_dir)
    else:
        print("ERROR: Provide a Google Drive URL as the first argument.")
        print("  Example: python download_gdrive.py 'https://drive.google.com/drive/folders/XXXXX'")
        print("  Or use --inspect to view already-downloaded files.")
        sys.exit(1)


if __name__ == "__main__":
    main()
