#!/usr/bin/env python3
"""
Fix a LoRA checkpoint saved with un-merged PEFT keys.

train.py's save_checkpoint() was calling merge_adapter() but then reading
base_model.model.state_dict() which still returns PEFT-wrapped key names
(q_proj.base_layer.weight / q_proj.lora_A.default.weight / ...).
qwen_tts expects plain keys (q_proj.weight), so the model was loading with
randomly-initialized talker weights.

This script reads the broken safetensors file, computes the true merged
weight (W = W_base + lora_B @ lora_A * scale) for each LoRA'd layer, and
writes a clean safetensors file with standard key names.

Usage:
    python3 scripts/merge_lora_checkpoint.py \
        --input  ~/.cache/qwen3tts-hinglish/checkpoints/best/model.safetensors \
        --output ~/.cache/qwen3tts-hinglish/checkpoints/best/model.safetensors \
        --lora-r 16 --lora-alpha 32

    # To fix all checkpoints in one go:
    python3 scripts/merge_lora_checkpoint.py --fix-r2-cache
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def load_tensors(path: Path) -> dict:
    out = {}
    with safe_open(str(path), framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def merge_lora(tensors: dict, lora_r: int, lora_alpha: float) -> dict:
    """Return a clean state dict with LoRA adapters merged into base weights."""
    scale = lora_alpha / lora_r

    # Group by the stem before .base_layer / .lora_A / .lora_B
    # e.g. "talker.model.layers.0.self_attn.q_proj"
    lora_a_re = re.compile(r"^(.+)\.lora_A\.default\.(weight|bias)$")
    lora_b_re = re.compile(r"^(.+)\.lora_B\.default\.(weight|bias)$")
    base_re   = re.compile(r"^(.+)\.base_layer\.(weight|bias)$")

    # Collect stems that have lora_A entries
    lora_stems = set()
    for k in tensors:
        m = lora_a_re.match(k)
        if m:
            lora_stems.add((m.group(1), m.group(2)))  # (stem, weight|bias)

    out = {}
    visited = set()

    for k, v in tensors.items():
        # Skip lora_A / lora_B / lora_embedding — handled below
        if ".lora_A." in k or ".lora_B." in k or ".lora_embedding" in k:
            continue

        m = base_re.match(k)
        if m:
            stem, suffix = m.group(1), m.group(2)
            if (stem, suffix) in lora_stems:
                # Merge: W = W_base + lora_B @ lora_A * scale
                lora_a_key = f"{stem}.lora_A.default.{suffix}"
                lora_b_key = f"{stem}.lora_B.default.{suffix}"
                w_base = v.float()
                lora_a = tensors[lora_a_key].float()
                lora_b = tensors[lora_b_key].float()
                delta = lora_b @ lora_a * scale
                merged = (w_base + delta).to(v.dtype)
                clean_k = f"{stem}.{suffix}"
                out[clean_k] = merged
            else:
                # base_layer with no LoRA — just rename
                clean_k = f"{stem}.{suffix}"
                out[clean_k] = v
            visited.add(k)
        else:
            # Non-LoRA key — keep as-is
            out[k] = v

    return out


def fix_checkpoint(input_path: Path, output_path: Path, lora_r: int, lora_alpha: float):
    print(f"Loading {input_path} ...")
    tensors = load_tensors(input_path)

    # Quick check: does this file have base_layer keys?
    has_peft = any(".base_layer." in k for k in tensors)
    if not has_peft:
        print("  No .base_layer. keys found — checkpoint looks clean already. Skipping.")
        return

    n_lora = sum(1 for k in tensors if ".lora_A." in k)
    print(f"  Found {n_lora} lora_A entries. Merging (r={lora_r}, alpha={lora_alpha}, scale={lora_alpha/lora_r}) ...")
    merged = merge_lora(tensors, lora_r=lora_r, lora_alpha=lora_alpha)

    # Sanity: verify no PEFT keys remain
    bad = [k for k in merged if ".base_layer." in k or ".lora_A." in k or ".lora_B." in k]
    if bad:
        raise RuntimeError(f"BUG: PEFT keys still present after merge: {bad[:3]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Saving {len(merged)} tensors → {output_path}")
    save_file(merged, str(output_path))
    print("  Done.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input",  type=Path, help="path to broken model.safetensors")
    ap.add_argument("--output", type=Path, help="path to write fixed model.safetensors (can be same as --input)")
    ap.add_argument("--lora-r",     type=int,   default=16,   help="LoRA rank (default: 16)")
    ap.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA alpha (default: 32)")
    ap.add_argument("--fix-r2-cache", action="store_true",
                    help="scan ~/.cache/qwen3tts-hinglish/checkpoints/ and fix all checkpoints found")
    args = ap.parse_args()

    if args.fix_r2_cache:
        cache_root = Path.home() / ".cache" / "qwen3tts-hinglish" / "checkpoints"
        safetensors_files = sorted(cache_root.rglob("model.safetensors"))
        if not safetensors_files:
            print(f"No model.safetensors found under {cache_root}")
            return
        for sf in safetensors_files:
            fix_checkpoint(sf, sf, lora_r=args.lora_r, lora_alpha=args.lora_alpha)
        return

    if not args.input or not args.output:
        ap.error("--input and --output are required (or use --fix-r2-cache)")

    fix_checkpoint(args.input, args.output, lora_r=args.lora_r, lora_alpha=args.lora_alpha)


if __name__ == "__main__":
    main()
