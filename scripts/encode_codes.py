#!/usr/bin/env python3
"""
Encode audio files into discrete codes using Qwen3-TTS-Tokenizer-12Hz.
Restartable: checkpoints progress, skips already-encoded entries on rerun.

Usage:
    python3 encode_codes.py \
        --manifest ./data/manifest_raw.jsonl \
        --out ./data/train_with_codes.jsonl \
        --tokenizer-model Qwen/Qwen3-TTS-Tokenizer-12Hz \
        --device mps \
        --batch-size 8
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import soundfile as sf


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer-model", default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--checkpoint-every", type=int, default=200,
                    help="flush partial progress to disk every N samples")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = out_path.with_suffix(out_path.suffix + ".partial")

    # Device check / fallback
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("WARNING: MPS not available, falling back to CPU. This will be slow.")
        device = "cpu"
    print(f"Using device: {device}")

    print(f"Loading tokenizer model: {args.tokenizer_model} ...")
    try:
        from qwen_tts import Qwen3TTSTokenizer
    except ImportError:
        print("ERROR: `qwen_tts` package not found. Install with: pip install qwen-tts", file=sys.stderr)
        sys.exit(1)

    tokenizer = Qwen3TTSTokenizer.from_pretrained(args.tokenizer_model)
    tokenizer = tokenizer.to(device)
    tokenizer.eval()

    done_keys = load_done_keys(out_path)
    print(f"Found {len(done_keys)} already-encoded entries, will skip these.")

    records = []
    with open(args.manifest) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    todo = [r for r in records if r["audio"] not in done_keys]
    print(f"Total records: {len(records)}, remaining to encode: {len(todo)}")

    pf = open(partial_path, "a")
    processed_since_flush = 0

    with torch.no_grad():
        for i, rec in enumerate(todo):
            audio_path = rec["audio"]
            try:
                wav, sr = sf.read(audio_path)
            except Exception as e:
                print(f"  [{i}] skip (read error): {audio_path}: {e}")
                continue

            try:
                wav_tensor = torch.tensor(wav, dtype=torch.float32, device=device)
                if wav_tensor.ndim > 1:
                    wav_tensor = wav_tensor.mean(dim=-1)
                codes = tokenizer.encode(wav_tensor.unsqueeze(0), sample_rate=sr)
                # codes expected shape: [num_codebooks, T] or similar; convert to nested list
                codes_list = codes.cpu().tolist()
            except Exception as e:
                print(f"  [{i}] skip (encode error): {audio_path}: {e}")
                continue

            rec_out = dict(rec)
            rec_out["audio_codes"] = codes_list
            pf.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            processed_since_flush += 1

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(todo)}] encoded")

            if processed_since_flush >= args.checkpoint_every:
                pf.flush()
                os.fsync(pf.fileno())
                processed_since_flush = 0

    pf.close()

    # Merge partial into final output (append-only — safe to rerun)
    print("Merging partial results into final output ...")
    with open(out_path, "a") as out_f, open(partial_path) as in_f:
        for line in in_f:
            out_f.write(line)

    print(f"Done. Output: {out_path}")
    print("You can safely delete the .partial file now, or keep it as a backup:")
    print(f"  {partial_path}")


if __name__ == "__main__":
    main()
