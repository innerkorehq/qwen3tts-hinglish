#!/usr/bin/env python3
"""
Merge per-shard encode_codes.py outputs into a single JSONL file.

Usage:
    python3 merge_shards.py --out ./data/train_with_codes.jsonl --num-shards 4
    python3 merge_shards.py --out ./data/eval_with_codes.jsonl  --num-shards 4
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="final merged output path")
    ap.add_argument("--num-shards", type=int, required=True)
    args = ap.parse_args()

    out_path = Path(args.out)
    shard_paths = [out_path.with_suffix(f".shard{i}.jsonl") for i in range(args.num_shards)]

    missing = [str(p) for p in shard_paths if not p.exists()]
    if missing:
        raise SystemExit(f"ERROR: missing shard files:\n" + "\n".join(f"  {p}" for p in missing))

    total = 0
    with open(out_path, "w") as f:
        for p in shard_paths:
            count = 0
            with open(p) as sf:
                for line in sf:
                    line = line.strip()
                    if line:
                        f.write(line + "\n")
                        count += 1
            print(f"  {p.name}: {count} records")
            total += count

    print(f"\nMerged {total} records -> {out_path}")

    for p in shard_paths:
        p.unlink()
    print("Shard files removed.")


if __name__ == "__main__":
    main()
