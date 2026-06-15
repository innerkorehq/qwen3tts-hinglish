#!/usr/bin/env python3
"""
One-time migration: convert R2's existing resampled audio backup -- either a
single `resampled.tar` or (older still) individual `resampled/*` objects --
into the sharded `resampled_shards/` scheme orchestrate.py now uses for
--resume (see scripts/shard_tar.py).

Run this on your LOCAL machine (not the Vast.ai instance) against an R2
bucket/prefix that still has resampled audio in one of the older formats:

    python3 scripts/migrate_resampled_to_shards.py --bucket my-bucket \\
        --key-prefix finetune/qwen3tts-hinglish

Steps: download resampled.tar (or individual resampled/* objects) from R2 ->
extract locally -> shard + upload as resampled_shards/ -> delete the
superseded objects from R2.
"""
import argparse
import os
import shutil
import sys
import tarfile
from pathlib import Path

from shard_tar import upload_shards
from upload_to_r2 import (
    delete_object,
    delete_prefix,
    download_dir,
    download_file,
    ensure_bucket,
    get_client,
    object_exists,
    prefix_exists,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default=os.environ.get("R2_BUCKET"))
    ap.add_argument("--key-prefix", default="finetune/qwen3tts-hinglish",
                    help="R2 key prefix containing resampled.tar / resampled/ "
                         "(default: finetune/qwen3tts-hinglish)")
    ap.add_argument("--work-dir", default="./_migrate_resampled",
                    help="local scratch dir for download + sharding (default: ./_migrate_resampled)")
    ap.add_argument("--shard-size-mb", type=int, default=1024,
                    help="target shard size in MB (default: 1024)")
    ap.add_argument("--concurrency", type=int, default=16,
                    help="parallel downloads for the individual-files fallback (default: 16)")
    ap.add_argument("--keep-local", action="store_true",
                    help="don't delete the local scratch dir after upload")
    ap.add_argument("--force", action="store_true",
                    help="proceed even if resampled_shards/manifest.json already exists in R2 (overwrite it)")
    args = ap.parse_args()

    if not args.bucket:
        print("ERROR: --bucket required (or set R2_BUCKET env var)", file=sys.stderr)
        sys.exit(1)

    client = get_client()
    ensure_bucket(client, args.bucket)

    key_prefix = args.key_prefix.rstrip("/")
    tar_key = f"{key_prefix}/resampled.tar"
    resampled_prefix = f"{key_prefix}/resampled"
    shards_prefix = f"{key_prefix}/resampled_shards"
    manifest_key = f"{shards_prefix}/manifest.json"

    if not args.force and object_exists(client, args.bucket, manifest_key):
        print(f"s3://{args.bucket}/{manifest_key} already exists -- pass --force to overwrite. Aborting.")
        sys.exit(1)

    work_dir = Path(args.work_dir)
    local_resampled = work_dir / "resampled"
    shard_work_dir = work_dir / "_shards"
    work_dir.mkdir(parents=True, exist_ok=True)

    if object_exists(client, args.bucket, tar_key):
        tar_path = work_dir / "resampled.tar"
        print(f"Downloading s3://{args.bucket}/{tar_key} -> {tar_path} ...")
        download_file(client, args.bucket, tar_key, tar_path)
        print(f"Extracting {tar_path} -> {local_resampled} ...")
        with tarfile.open(tar_path, "r") as tf:
            tf.extractall(work_dir)
        tar_path.unlink()
        source = "tar"
    elif prefix_exists(client, args.bucket, resampled_prefix):
        print(f"Downloading s3://{args.bucket}/{resampled_prefix}/ -> {local_resampled} ...")
        download_dir(client, args.bucket, resampled_prefix, local_resampled, concurrency=args.concurrency)
        source = "files"
    else:
        print(f"Found neither s3://{args.bucket}/{tar_key} nor "
              f"s3://{args.bucket}/{resampled_prefix}/ -- nothing to migrate.")
        sys.exit(1)

    print(f"Sharding {local_resampled} -> s3://{args.bucket}/{shards_prefix}/ ...")
    upload_shards(client, args.bucket, shards_prefix, local_resampled, shard_work_dir,
                   arcname="resampled", shard_size_bytes=args.shard_size_mb * (1 << 20),
                   force=args.force)

    if source == "tar":
        print(f"Deleting s3://{args.bucket}/{tar_key} ...")
        delete_object(client, args.bucket, tar_key)
    else:
        print(f"Deleting individual objects under s3://{args.bucket}/{resampled_prefix}/ ...")
        deleted = delete_prefix(client, args.bucket, resampled_prefix)
        print(f"Deleted {deleted} objects.")

    if args.keep_local:
        print(f"Local scratch left at {work_dir} (--keep-local)")
    else:
        print(f"Removing local scratch dir {work_dir} ...")
        shutil.rmtree(work_dir, ignore_errors=True)

    print("Done.")


if __name__ == "__main__":
    main()
