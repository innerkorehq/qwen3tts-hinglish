#!/usr/bin/env python3
"""
Upload/download files or directories to/from Cloudflare R2 (S3-compatible API).

Requires env vars:
    R2_ACCOUNT_ID
    R2_ACCESS_KEY_ID
    R2_SECRET_ACCESS_KEY
    R2_BUCKET (optional, can also pass --bucket)

Usage (upload):
    python3 upload_to_r2.py --file ./data/train_with_codes.jsonl \
        --bucket my-bucket --key finetune/train_with_codes.jsonl

Usage (upload directory recursively):
    python3 upload_to_r2.py --file ./checkpoints/final \
        --bucket my-bucket --key finetune/checkpoints/final --recursive

Usage (download):
    python3 upload_to_r2.py --download --key finetune/train_with_codes.jsonl \
        --bucket my-bucket --file ./data/train_with_codes.jsonl

Usage (existence check, for resume -- exit 0 if present, 1 if missing):
    python3 upload_to_r2.py --exists --key finetune/train_with_codes.jsonl --bucket my-bucket
    python3 upload_to_r2.py --exists --key finetune/checkpoints/final --bucket my-bucket --recursive
"""
import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from net_utils import r2_boto_config, retry

DEFAULT_CONCURRENCY = 16

_thread_local = threading.local()


def get_client():
    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    if not all([account_id, access_key, secret_key]):
        print("ERROR: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY must be set", file=sys.stderr)
        sys.exit(1)

    endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=r2_boto_config(),
        region_name="auto",
    )


def ensure_bucket(client, bucket):
    """Create the bucket if it doesn't exist yet (best-effort, idempotent).

    Without this, the *first* run against a fresh R2 account would fail every
    upload with NoSuchBucket -- including the final checkpoint upload after a
    full training run -- which is exactly the kind of late, expensive failure
    the onstart script's retry/self-destruct logic is meant to avoid.
    """
    def _head():
        try:
            client.head_bucket(Bucket=bucket)
            return True
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchBucket", "NotFound"):
                return False
            raise  # transient -- let retry() handle it

    if retry(_head, what=f"head bucket {bucket}"):
        return

    try:
        client.create_bucket(Bucket=bucket)
        print(f"Created R2 bucket '{bucket}' (didn't exist yet)")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            print(f"WARNING: could not create bucket '{bucket}': {e}", file=sys.stderr)


def upload_file(client, bucket, local_path, key, quiet=False):
    if not quiet:
        size_mb = os.path.getsize(local_path) / (1 << 20)
        print(f"Uploading {local_path} ({size_mb:.1f} MB) -> s3://{bucket}/{key}")
    retry(client.upload_file, str(local_path), bucket, key,
          what=f"upload {key}")


def _thread_client():
    """One boto3 client per worker thread -- clients aren't guaranteed safe
    to share across threads, and a fresh connection pool per thread avoids
    contention under concurrent transfers."""
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = get_client()
        _thread_local.client = client
    return client


def upload_dir(client, bucket, local_dir, key_prefix, concurrency=DEFAULT_CONCURRENCY):
    local_dir = Path(local_dir)
    files = list(local_dir.rglob("*"))
    files = [f for f in files if f.is_file()]
    print(f"Uploading {len(files)} files from {local_dir} -> s3://{bucket}/{key_prefix}/ "
          f"({concurrency} parallel) ...")

    def _upload_one(f):
        rel = f.relative_to(local_dir)
        key = f"{key_prefix}/{rel.as_posix()}"
        upload_file(_thread_client(), bucket, f, key, quiet=True)
        return rel, key

    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(_upload_one, f) for f in files]
        for fut in as_completed(futures):
            rel, key = fut.result()
            done += 1
            print(f"  [{done}/{len(files)}] {rel} -> {key}")


def download_file(client, bucket, key, local_path, quiet=False):
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(f"Downloading s3://{bucket}/{key} -> {local_path}")
    retry(client.download_file, bucket, key, str(local_path),
          what=f"download {key}")


def download_dir(client, bucket, key_prefix, local_dir, concurrency=DEFAULT_CONCURRENCY):
    """Recursively download all objects under key_prefix to local_dir, preserving structure."""
    local_dir = Path(local_dir)
    prefix = key_prefix.rstrip("/") + "/"
    paginator = client.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):]
            if not rel:  # skip the "directory marker" object itself
                continue
            keys.append((key, local_dir / rel))

    print(f"Downloading {len(keys)} files from s3://{bucket}/{prefix} -> {local_dir} "
          f"({concurrency} parallel) ...")

    def _download_one(key, dest):
        download_file(_thread_client(), bucket, key, dest, quiet=True)
        return key, dest

    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(_download_one, key, dest) for key, dest in keys]
        for fut in as_completed(futures):
            key, dest = fut.result()
            done += 1
            print(f"  [{done}/{len(keys)}] {key} -> {dest}")

    print(f"Downloaded {len(keys)} files from s3://{bucket}/{prefix} -> {local_dir}")


def object_exists(client, bucket, key):
    def _head():
        try:
            client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise  # transient (network/5xx) -- let retry() handle it

    return retry(_head, what=f"head {key}")


def prefix_exists(client, bucket, key_prefix):
    """True if any object exists under key_prefix/ (for directory uploads)."""
    prefix = key_prefix.rstrip("/") + "/"
    resp = retry(client.list_objects_v2, Bucket=bucket, Prefix=prefix, MaxKeys=1,
                  what=f"list {prefix}")
    return resp.get("KeyCount", 0) > 0


def delete_prefix(client, bucket, key_prefix):
    """Delete all objects under key_prefix/ (best-effort, used to clean up
    objects uploaded under a now-superseded scheme, e.g. the individual
    per-file resampled/ objects after resampled.tar replaces them)."""
    prefix = key_prefix.rstrip("/") + "/"
    paginator = client.get_paginator("list_objects_v2")
    deleted = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if not keys:
            continue
        retry(client.delete_objects, Bucket=bucket, Delete={"Objects": keys},
              what=f"delete {len(keys)} objects under {prefix}")
        deleted += len(keys)
    return deleted


def count_prefix(client, bucket, key_prefix):
    """Count objects under key_prefix/ (excluding the directory-marker object
    itself, if any) -- used for resume completeness checks where a single
    `prefix_exists` (MaxKeys=1) would wrongly treat a partially-uploaded
    directory as complete."""
    prefix = key_prefix.rstrip("/") + "/"
    paginator = client.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"] != prefix:
                count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=False, help="local path (upload source / download dest)")
    ap.add_argument("--key", required=True, help="remote object key (or prefix if --recursive)")
    ap.add_argument("--bucket", default=os.environ.get("R2_BUCKET"))
    ap.add_argument("--recursive", action="store_true", help="upload/check a directory recursively")
    ap.add_argument("--download", action="store_true", help="download instead of upload")
    ap.add_argument("--exists", action="store_true",
                    help="check whether --key (or --key prefix if --recursive) exists; "
                         "prints EXISTS/MISSING and exits 0/1, no --file needed")
    ap.add_argument("--count", action="store_true",
                    help="print the number of objects under --key prefix (requires "
                         "--recursive) -- for resume completeness checks, where "
                         "--exists (MaxKeys=1) can't tell a partial upload from a "
                         "complete one; no --file needed")
    ap.add_argument("--delete", action="store_true",
                    help="delete all objects under --key prefix (requires --recursive); "
                         "best-effort cleanup, no --file needed")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                    help=f"parallel transfers for --recursive (default {DEFAULT_CONCURRENCY})")
    args = ap.parse_args()

    if not args.bucket:
        print("ERROR: --bucket required (or set R2_BUCKET env var)", file=sys.stderr)
        sys.exit(1)

    client = get_client()
    ensure_bucket(client, args.bucket)

    if args.count:
        print(count_prefix(client, args.bucket, args.key))
        return

    if args.delete:
        if not args.recursive:
            print("ERROR: --delete requires --recursive", file=sys.stderr)
            sys.exit(1)
        deleted = delete_prefix(client, args.bucket, args.key)
        print(f"Deleted {deleted} objects under s3://{args.bucket}/{args.key.rstrip('/')}/")
        return

    if args.exists:
        found = prefix_exists(client, args.bucket, args.key) if args.recursive \
            else object_exists(client, args.bucket, args.key)
        print("EXISTS" if found else "MISSING")
        sys.exit(0 if found else 1)

    if not args.file:
        print("ERROR: --file required (unless --exists)", file=sys.stderr)
        sys.exit(1)

    if args.download:
        if args.recursive:
            download_dir(client, args.bucket, args.key, args.file, concurrency=args.concurrency)
        else:
            download_file(client, args.bucket, args.key, args.file)
    elif args.recursive:
        upload_dir(client, args.bucket, args.file, args.key, concurrency=args.concurrency)
    else:
        upload_file(client, args.bucket, args.file, args.key)

    print("Done.")


if __name__ == "__main__":
    main()
