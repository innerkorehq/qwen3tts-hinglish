#!/usr/bin/env python3
"""
Download and extract the HiACC (Hinglish Adult & Children Code-switched Corpus)
from Zenodo: https://zenodo.org/records/15551669

Usage:
    python3 download_hiacc.py --out-dir ./data/raw/hiacc
"""
import argparse
import os
import sys
import zipfile
import tarfile
import requests
from pathlib import Path

from net_utils import HTTP_TIMEOUT, download_with_retry, retry

ZENODO_RECORD_API = "https://zenodo.org/api/records/15551669"


def get_file_list():
    """Query Zenodo API for the record's file list (name + download URL)."""
    def _fetch():
        resp = requests.get(ZENODO_RECORD_API, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    data = retry(_fetch, what="Zenodo record metadata")
    files = []
    for f in data.get("files", []):
        files.append({
            "name": f["key"],
            "url": f["links"]["self"],
            "size": f.get("size", 0),
        })
    return files


def download_file(url, dest_path):
    if dest_path.exists():
        print(f"  already exists, skipping: {dest_path}")
        return
    print(f"  downloading {dest_path.name} ...")
    download_with_retry(url, dest_path)


def extract_archive(path, out_dir):
    print(f"  extracting {path.name} ...")
    if path.suffix == ".zip":
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(out_dir)
    elif path.suffixes[-2:] == [".tar", ".gz"] or path.suffix == ".tgz":
        with tarfile.open(path, "r:gz") as t:
            t.extractall(out_dir)
    elif path.suffix == ".tar":
        with tarfile.open(path, "r:") as t:
            t.extractall(out_dir)
    else:
        print(f"    (not an archive, leaving as-is: {path.name})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "_archives"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Querying Zenodo record metadata ...")
    try:
        files = get_file_list()
    except Exception as e:
        print(f"ERROR: could not query Zenodo API: {e}", file=sys.stderr)
        print("Fallback: visit https://zenodo.org/records/15551669 manually,", file=sys.stderr)
        print("download adult + children audio/transcript archives into:", file=sys.stderr)
        print(f"  {raw_dir}", file=sys.stderr)
        print("then rerun with --skip-extract removed to extract.", file=sys.stderr)
        sys.exit(1)

    if not files:
        print("No files found in Zenodo record — check the record ID is still valid.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} files:")
    for f in files:
        size_mb = f["size"] / (1 << 20)
        print(f"  - {f['name']} ({size_mb:.1f} MB)")

    for f in files:
        dest = raw_dir / f["name"]
        download_file(f["url"], dest)

    if not args.skip_extract:
        for f in files:
            dest = raw_dir / f["name"]
            extract_archive(dest, out_dir)

    print(f"\nDone. Raw archives in {raw_dir}, extracted data in {out_dir}")
    print("Expected structure: adult/ and children/ subdirs with audio + transcript files.")
    print("If the layout differs, inspect manually and adjust build_manifest.py's HiACC loader.")


if __name__ == "__main__":
    main()
