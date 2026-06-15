#!/usr/bin/env python3
"""
Shard a directory into ~N MB tar files for upload to R2, and the reverse
(download manifest + shards, extract back into a directory).

Why shards instead of one big tar: a single multi-GB tar means a flaky
connection wastes the whole multipart upload's (or download's) progress
whenever it gets reset -- on a bufferbloated link this can mean hours of lost
progress. It's also a same-size duplicate of the source directory while both
exist on disk. Splitting into ~1GB shards bounds both problems: a reset wastes
at most one shard's worth of transfer, and only one shard (~1GB) exists on
disk alongside the source at a time instead of a full duplicate. A small
manifest.json (uploaded last, after every shard succeeds) makes the whole set
resumable: re-running `upload` skips shards already present in R2, and the
manifest's existence is what orchestrate.py checks for --resume.

Usage:
    # tar+upload src-dir's contents as <key-prefix>/shard_NNNNN.tar +
    # <key-prefix>/manifest.json (skips shards already present in R2)
    python3 scripts/shard_tar.py upload --src-dir ./data/resampled \\
        --work-dir ./data/_shards --bucket my-bucket \\
        --key-prefix finetune/qwen3tts-hinglish/resampled_shards \\
        --arcname resampled --shard-size-mb 1024

    # download manifest + shards, extract into dest-dir (-> dest-dir/resampled/...)
    python3 scripts/shard_tar.py download --dest-dir ./data \\
        --work-dir ./data/_shards --bucket my-bucket \\
        --key-prefix finetune/qwen3tts-hinglish/resampled_shards
"""
import argparse
import json
import os
import sys
import tarfile
from pathlib import Path

from upload_to_r2 import (
    download_file,
    ensure_bucket,
    get_client,
    object_exists,
    upload_file,
)

MANIFEST_NAME = "manifest.json"


def _list_files(src_dir):
    src_dir = Path(src_dir)
    return [(p, p.relative_to(src_dir)) for p in sorted(src_dir.rglob("*")) if p.is_file()]


def _bin_pack(files, shard_size_bytes):
    """Greedily group (abs_path, rel_path) pairs so each shard's total size is
    <= shard_size_bytes (a single file larger than shard_size_bytes gets its
    own shard)."""
    shards = []
    current = []
    current_size = 0
    for abs_path, rel_path in files:
        size = abs_path.stat().st_size
        if current and current_size + size > shard_size_bytes:
            shards.append(current)
            current = []
            current_size = 0
        current.append((abs_path, rel_path))
        current_size += size
    if current:
        shards.append(current)
    return shards


def upload_shards(client, bucket, key_prefix, src_dir, work_dir, arcname,
                   shard_size_bytes, force=False):
    key_prefix = key_prefix.rstrip("/")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    files = _list_files(src_dir)
    shards = _bin_pack(files, shard_size_bytes)
    print(f"Packed {len(files)} files from {src_dir} into {len(shards)} shards "
          f"(target {shard_size_bytes / (1 << 20):.0f} MB each)")

    shard_names = []
    for i, members in enumerate(shards):
        shard_name = f"shard_{i:05d}.tar"
        shard_names.append(shard_name)
        key = f"{key_prefix}/{shard_name}"

        if not force and object_exists(client, bucket, key):
            print(f"  [{i + 1}/{len(shards)}] {shard_name} already in R2 -- skipping")
            continue

        shard_path = work_dir / shard_name
        print(f"  [{i + 1}/{len(shards)}] creating {shard_path} ({len(members)} files) ...")
        with tarfile.open(shard_path, "w") as tf:
            for abs_path, rel_path in members:
                tf.add(abs_path, arcname=str(Path(arcname) / rel_path))

        upload_file(client, bucket, shard_path, key)
        shard_path.unlink()

    manifest = {"arcname": arcname, "shards": shard_names, "num_files": len(files)}
    manifest_path = work_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2))
    upload_file(client, bucket, manifest_path, f"{key_prefix}/{MANIFEST_NAME}")
    manifest_path.unlink()

    return manifest


def download_shards(client, bucket, key_prefix, dest_dir, work_dir):
    key_prefix = key_prefix.rstrip("/")
    work_dir = Path(work_dir)
    dest_dir = Path(dest_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    dest_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = work_dir / MANIFEST_NAME
    download_file(client, bucket, f"{key_prefix}/{MANIFEST_NAME}", manifest_path)
    manifest = json.loads(manifest_path.read_text())
    manifest_path.unlink()

    shards = manifest["shards"]
    for i, shard_name in enumerate(shards):
        shard_path = work_dir / shard_name
        print(f"  [{i + 1}/{len(shards)}] downloading + extracting {shard_name} ...")
        download_file(client, bucket, f"{key_prefix}/{shard_name}", shard_path)
        with tarfile.open(shard_path, "r") as tf:
            tf.extractall(dest_dir)
        shard_path.unlink()

    return manifest


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("upload", help="tar+upload src-dir as shards + manifest.json")
    up.add_argument("--src-dir", required=True)
    up.add_argument("--work-dir", required=True, help="local scratch dir for shard tars")
    up.add_argument("--bucket", default=os.environ.get("R2_BUCKET"))
    up.add_argument("--key-prefix", required=True)
    up.add_argument("--arcname", required=True,
                    help="top-level directory name inside each shard tar")
    up.add_argument("--shard-size-mb", type=int, default=1024)
    up.add_argument("--force", action="store_true",
                     help="re-upload shards even if already present in R2")

    down = sub.add_parser("download", help="download manifest.json + shards, extract into dest-dir")
    down.add_argument("--dest-dir", required=True)
    down.add_argument("--work-dir", required=True, help="local scratch dir for shard tars")
    down.add_argument("--bucket", default=os.environ.get("R2_BUCKET"))
    down.add_argument("--key-prefix", required=True)

    args = ap.parse_args()

    if not args.bucket:
        print("ERROR: --bucket required (or set R2_BUCKET env var)", file=sys.stderr)
        sys.exit(1)

    client = get_client()
    ensure_bucket(client, args.bucket)

    if args.cmd == "upload":
        upload_shards(client, args.bucket, args.key_prefix, args.src_dir,
                       args.work_dir, args.arcname, args.shard_size_mb * (1 << 20),
                       force=args.force)
    else:
        download_shards(client, args.bucket, args.key_prefix, args.dest_dir, args.work_dir)

    print("Done.")


if __name__ == "__main__":
    main()
