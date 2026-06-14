#!/usr/bin/env python3
"""
Shared networking helpers: retries + timeouts for HTTP downloads and R2
(boto3/S3) calls.

Network is the single biggest source of "stuck for hours, then self-destruct
after burning the timeout" failures on a rented Vast.ai instance: a stalled
CDN connection (no bytes, connection not closed) hangs forever without a
read-timeout, and a single dropped connection mid-upload/download otherwise
kills the whole pipeline run. Every helper here:

  - sets an explicit read-timeout so a silent stall raises instead of hanging,
  - retries with exponential backoff on transient failures,
  - resumes partial downloads (HTTP Range / boto3 multipart) where possible,

so a flaky connection costs a bit of wall-clock time, not the whole run.
"""
import time
from pathlib import Path

import requests
from botocore.config import Config


# (connect_timeout, read_timeout) in seconds for requests.get(stream=True, ...).
# read_timeout bounds the gap between chunks -- a connection that stops
# sending bytes without closing raises ReadTimeout after this long.
HTTP_TIMEOUT = (15, 60)


def r2_boto_config():
    """Config for boto3 R2 clients: bounded connect/read timeouts plus
    botocore's own adaptive retry (handles throttling + transient 5xx/
    connection errors at the request level)."""
    return Config(
        signature_version="s3v4",
        connect_timeout=10,
        read_timeout=120,
        retries={"max_attempts": 6, "mode": "adaptive"},
    )


def retry(fn, *args, max_attempts=5, backoff_base=5, what="operation", **kwargs):
    """Call fn(*args, **kwargs), retrying on any Exception with exponential
    backoff (backoff_base, 2x, 4x, ... capped at 60s). Re-raises the last
    exception if all attempts fail."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == max_attempts:
                raise
            backoff = min(60, backoff_base * 2 ** (attempt - 1))
            print(f"  {what}: attempt {attempt}/{max_attempts} failed ({e}); "
                  f"retrying in {backoff}s ...")
            time.sleep(backoff)


def download_with_retry(url, dest_path, max_attempts=6, timeout=HTTP_TIMEOUT,
                         chunk_size=1 << 20, headers=None, progress=True):
    """Download url -> dest_path with resumable Range requests + retries.

    Writes to dest_path + ".part" and resumes from however many bytes were
    already written on retry (falls back to a full restart if the server
    doesn't honor Range). A stalled connection (no bytes for timeout[1]
    seconds) raises ReadTimeout, which is caught and retried.
    """
    dest_path = Path(dest_path)
    if dest_path.exists():
        print(f"  already exists, skipping: {dest_path}")
        return

    part_path = dest_path.with_name(dest_path.name + ".part")
    base_headers = dict(headers or {})

    last_exc = None
    for attempt in range(1, max_attempts + 1):
        resume_pos = part_path.stat().st_size if part_path.exists() else 0
        req_headers = dict(base_headers)
        if resume_pos:
            req_headers["Range"] = f"bytes={resume_pos}-"

        try:
            with requests.get(url, stream=True, timeout=timeout, headers=req_headers) as r:
                if resume_pos and r.status_code == 200:
                    # server ignored Range -- restart from scratch
                    resume_pos = 0
                    part_path.unlink()
                elif r.status_code == 416:
                    # Range Not Satisfiable -- we already have it all
                    part_path.rename(dest_path)
                    return
                r.raise_for_status()

                total = resume_pos
                cl = r.headers.get("content-length")
                if cl:
                    total += int(cl)

                mode = "ab" if resume_pos else "wb"
                downloaded = resume_pos
                with open(part_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress and total:
                            pct = downloaded / total * 100
                            print(f"\r    {pct:5.1f}% "
                                  f"({downloaded / (1 << 20):.0f}/{total / (1 << 20):.0f} MB)",
                                  end="", flush=True)
            if progress:
                print()
            part_path.rename(dest_path)
            return
        except (requests.exceptions.RequestException, OSError) as e:
            last_exc = e
            if progress:
                print()
            if attempt == max_attempts:
                break
            backoff = min(60, 5 * 2 ** (attempt - 1))
            have = part_path.stat().st_size if part_path.exists() else 0
            print(f"  download attempt {attempt}/{max_attempts} failed ({e}); "
                  f"retrying in {backoff}s (resuming from {have / (1 << 20):.0f} MB) ...")
            time.sleep(backoff)

    raise RuntimeError(f"download failed after {max_attempts} attempts: {url}") from last_exc
