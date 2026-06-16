#!/usr/bin/env python3
"""
Rewrite absolute audio/ref_audio paths in codes JSONL files so they point to
the actual resampled audio location on the current machine.

Matches anything containing /resampled/<subpath> and replaces the prefix with
--resampled-root.  Safe to run multiple times (idempotent).

Usage (on Vast.ai instance after downloading codes from R2):
    python3 rewrite_paths.py \
        ./data/train_with_codes.jsonl ./data/eval_with_codes.jsonl \
        --resampled-root /root/work/data/resampled
"""
import argparse
import json
import re
import sys
from pathlib import Path


def rewrite_file(path: str, resampled_root: str):
    src = Path(path)
    if not src.exists():
        print(f"  WARNING: {path} not found, skipping", file=sys.stderr)
        return

    resampled_root = resampled_root.rstrip("/")
    lines_out = []
    n_rewritten = 0

    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for key in ("ref_audio", "audio"):
                val = rec.get(key, "")
                if not val:
                    continue
                m = re.search(r"/resampled/(.+)$", val)
                if m and not val.startswith(resampled_root):
                    rec[key] = f"{resampled_root}/{m.group(1)}"
                    n_rewritten += 1
            lines_out.append(json.dumps(rec, ensure_ascii=False))

    with open(src, "w") as f:
        f.write("\n".join(lines_out) + "\n")

    print(f"  {src.name}: {len(lines_out)} records, {n_rewritten} paths rewritten")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="JSONL files to rewrite in-place")
    ap.add_argument("--resampled-root", required=True,
                    help="absolute path to the resampled audio root on this machine")
    args = ap.parse_args()

    for f in args.files:
        rewrite_file(f, args.resampled_root)


if __name__ == "__main__":
    main()
