#!/usr/bin/env python3
"""
Encode audio files into discrete codes using Qwen3-TTS-Tokenizer-12Hz.
Restartable: checkpoints progress, skips already-encoded entries on rerun.

Follows the official prepare_data.py batched encode API
(https://github.com/QwenLM/Qwen3-TTS/blob/main/finetuning/prepare_data.py):
Qwen3TTSTokenizer.from_pretrained(path, device_map=...).encode(list_of_paths)
-> enc_res.audio_codes (list of (T, 16) tensors).

Usage:
    python3 encode_codes.py \
        --manifest ./data/manifest_raw.jsonl \
        --out ./data/train_with_codes.jsonl \
        --tokenizer-model Qwen/Qwen3-TTS-Tokenizer-12Hz \
        --device cuda \
        --batch-size 32
"""
import argparse
import json
import os
import sys
from pathlib import Path


def load_done_keys(out_path: Path):
    """Read existing output (and .partial) to find already-processed audio paths."""
    done = set()
    for p in (out_path, out_path.with_suffix(out_path.suffix + ".partial")):
        if p.exists():
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        done.add(rec["audio"])
                    except Exception:
                        continue
    return done


def encode_batch(tokenizer, records):
    """Encode a batch of records, falling back to per-item encoding on batch failure."""
    paths = [r["audio"] for r in records]
    try:
        enc_res = tokenizer.encode(paths)
        out = []
        for code, rec in zip(enc_res.audio_codes, records):
            rec_out = dict(rec)
            rec_out["audio_codes"] = code.cpu().tolist()
            out.append(rec_out)
        return out, []
    except Exception as e:
        print(f"  batch encode failed ({e}), falling back to per-item encoding")
        out = []
        failed = []
        for rec in records:
            try:
                enc_res = tokenizer.encode([rec["audio"]])
                rec_out = dict(rec)
                rec_out["audio_codes"] = enc_res.audio_codes[0].cpu().tolist()
                out.append(rec_out)
            except Exception as e2:
                print(f"    skip (encode error): {rec['audio']}: {e2}")
                failed.append(rec)
        return out, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer-model", default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    ap.add_argument("--device", default="cuda", choices=["mps", "cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--checkpoint-every", type=int, default=200,
                    help="flush partial progress to disk every N samples")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="total number of parallel worker shards")
    ap.add_argument("--shard-idx", type=int, default=0,
                    help="index of this shard (0-based, must be < --num-shards)")
    args = ap.parse_args()

    if args.num_shards > 1:
        out_path = Path(args.out).with_suffix(f".shard{args.shard_idx}.jsonl")
    else:
        out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = out_path.with_suffix(out_path.suffix + ".partial")

    device = args.device
    if device == "mps":
        import torch
        if not torch.backends.mps.is_available():
            print("WARNING: MPS not available, falling back to CPU. This will be slow.")
            device = "cpu"
    print(f"Using device: {device}")

    print(f"Loading tokenizer model: {args.tokenizer_model} ...")
    try:
        from qwen_tts import Qwen3TTSTokenizer
    except ImportError:
        print("ERROR: `qwen_tts` package not found. Install with: pip install qwen-tts", file=sys.stderr)
        sys.exit(1)

    tokenizer = Qwen3TTSTokenizer.from_pretrained(args.tokenizer_model, device_map=device)

    done_keys = load_done_keys(out_path)
    print(f"Found {len(done_keys)} already-encoded entries, will skip these.")

    records = []
    with open(args.manifest) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    # Shard: each worker handles every Nth record starting at shard_idx.
    # Records are assigned by position so shards are interleaved (not contiguous)
    # giving each worker a representative mix of durations/sources.
    if args.num_shards > 1:
        records = [r for i, r in enumerate(records) if i % args.num_shards == args.shard_idx]
        print(f"Shard {args.shard_idx}/{args.num_shards}: {len(records)} records assigned to this worker")

    todo = [r for r in records if r["audio"] not in done_keys]
    print(f"Total records: {len(records)}, remaining to encode: {len(todo)}")

    pf = open(partial_path, "a")
    processed_since_flush = 0
    total_failed = 0

    for i in range(0, len(todo), args.batch_size):
        batch = todo[i:i + args.batch_size]
        encoded, failed = encode_batch(tokenizer, batch)
        total_failed += len(failed)
        for rec_out in encoded:
            pf.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
        processed_since_flush += len(encoded)

        done_so_far = min(i + args.batch_size, len(todo))
        print(f"  [{done_so_far}/{len(todo)}] encoded (failed so far: {total_failed})")

        if processed_since_flush >= args.checkpoint_every:
            pf.flush()
            os.fsync(pf.fileno())
            processed_since_flush = 0

    pf.close()

    if total_failed > 0:
        fail_pct = total_failed / len(todo) * 100
        msg = f"encode_codes: {total_failed}/{len(todo)} files failed to encode ({fail_pct:.1f}%)"
        if fail_pct > 5:
            print(f"ERROR: {msg} -- too many failures, refusing to produce corrupt training data", file=sys.stderr)
            sys.exit(1)
        print(f"WARNING: {msg} -- these clips will be absent from training data")

    # Merge partial into final output (append-only -- safe to rerun)
    print("Merging partial results into final output ...")
    with open(out_path, "a") as out_f, open(partial_path) as in_f:
        for line in in_f:
            out_f.write(line)

    print(f"Done. Output: {out_path} ({len(todo) - total_failed} encoded, {total_failed} skipped)")
    print("You can safely delete the .partial file now, or keep it as a backup:")
    print(f"  {partial_path}")


if __name__ == "__main__":
    main()
