#!/usr/bin/env python3
"""
Convert the fp32 master checkpoint of the fine-tuned Qwen3-TTS model into
other formats:
  - fp16  : broadly portable, good default for MPS (M4) inference
  - bf16  : already produced by train.py, this script can re-derive it too
  - GGUF  : attempted via llama.cpp's convert_hf_to_gguf.py for llama.cpp/MPS
            runtimes. Qwen3-TTS is NOT a standard text-only Qwen3ForCausalLM —
            it's a TTS architecture. llama.cpp's converter only supports a
            fixed list of named architectures. This step is LIKELY TO FAIL
            with "Model <Arch> is not supported" unless either:
              (a) Qwen3-TTS's architecture has since been added to llama.cpp, or
              (b) only the underlying LM "trunk" (if Qwen3ForCausalLM-compatible)
                  is being converted, separately from the audio codec.
            This script attempts the conversion and reports the failure clearly
            rather than silently producing a broken file.

Usage:
    # fp16 conversion (always works, simple dtype cast):
    python3 convert_model.py --master ./checkpoints/final_fp32 \
        --to fp16 --out ./checkpoints/final_fp16

    # Attempt GGUF conversion (may fail — see above):
    python3 convert_model.py --master ./checkpoints/final_fp32 \
        --to gguf --out ./checkpoints/final_gguf \
        --llama-cpp-dir ./llama.cpp --gguf-outtype f16
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch


def convert_dtype(master_dir: Path, out_dir: Path, dtype_name: str):
    """Load fp32 master, cast to target dtype, save."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    if dtype_name not in dtype_map:
        raise ValueError(f"unsupported dtype: {dtype_name}")
    target_dtype = dtype_map[dtype_name]

    print(f"Loading fp32 master from {master_dir} ...")
    model = AutoModelForCausalLM.from_pretrained(master_dir, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(master_dir)

    print(f"Casting to {dtype_name} ...")
    model = model.to(target_dtype)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving {dtype_name} model to {out_dir} ...")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"Done. {dtype_name} model at {out_dir}")
    if dtype_name == "fp16":
        print("Note: fp16 has the broadest cross-device support (CUDA, MPS, CPU).")
        print("If MPS inference with fp16 has issues on your torch version, fall")
        print("back to fp32 (the master) — slower but maximally compatible.")


def convert_gguf(master_dir: Path, out_dir: Path, llama_cpp_dir: Path, outtype: str, quantize: str = None):
    """
    Attempt GGUF conversion via llama.cpp's convert_hf_to_gguf.py.

    This is expected to potentially fail for Qwen3-TTS — see module docstring.
    Failure is reported clearly with the actual error and next steps, rather
    than crashing uninformatively.
    """
    converter = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not converter.exists():
        print(f"ERROR: {converter} not found.", file=sys.stderr)
        print("Clone llama.cpp first:", file=sys.stderr)
        print(f"  git clone https://github.com/ggml-org/llama.cpp {llama_cpp_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"model-{outtype}.gguf"

    # Check the architecture in config.json up front so we can give a clear
    # message instead of relying on llama.cpp's stderr alone.
    config_path = master_dir / "config.json"
    arch = None
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        archs = cfg.get("architectures", [])
        arch = archs[0] if archs else None
        print(f"Model architecture in config.json: {arch}")
        if arch and "Qwen3" not in arch:
            print(f"NOTE: architecture '{arch}' may not be a standard Qwen3ForCausalLM.")
            print("llama.cpp's converter supports a fixed list of named architectures;")
            print("TTS-specific architectures are commonly NOT in that list.")

    cmd = [
        sys.executable, str(converter),
        str(master_dir),
        "--outfile", str(out_file),
        "--outtype", outtype,
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        print("\n=== GGUF conversion FAILED ===", file=sys.stderr)
        if "is not supported" in result.stderr or "not supported" in result.stdout:
            print(file=sys.stderr)
            print(f"This is the expected failure mode for Qwen3-TTS (architecture: {arch}).", file=sys.stderr)
            print("llama.cpp's convert_hf_to_gguf.py only supports a fixed set of named", file=sys.stderr)
            print("architectures, and TTS models with custom audio codec heads typically", file=sys.stderr)
            print("aren't among them.", file=sys.stderr)
            print(file=sys.stderr)
            print("Options:", file=sys.stderr)
            print("  1. Check if a newer llama.cpp checkout adds support for this", file=sys.stderr)
            print("     architecture (`git pull` in llama.cpp dir, retry).", file=sys.stderr)
            print("  2. Skip GGUF for now — use the fp16 checkpoint for MPS inference", file=sys.stderr)
            print("     via standard `transformers`, which doesn't have this constraint.", file=sys.stderr)
            print("  3. If Qwen3-TTS separates into a Qwen3ForCausalLM 'trunk' + a separate", file=sys.stderr)
            print("     audio codec module, it may be possible to GGUF-convert only the", file=sys.stderr)
            print("     trunk (for text/token-level inference) while keeping the codec on", file=sys.stderr)
            print("     `transformers`/PyTorch — but this requires the architecture to be", file=sys.stderr)
            print("     split into separate save directories, which isn't done by default.", file=sys.stderr)
        sys.exit(1)

    print(f"\nGGUF file written: {out_file}")

    if quantize:
        quantize_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"
        if not quantize_bin.exists():
            quantize_bin = llama_cpp_dir / "llama-quantize"
        if not quantize_bin.exists():
            print(f"WARNING: llama-quantize binary not found at expected paths under {llama_cpp_dir}.")
            print("Build llama.cpp first (cmake -B build && cmake --build build) to enable quantization.")
            return
        quant_out = out_dir / f"model-{quantize}.gguf"
        qcmd = [str(quantize_bin), str(out_file), str(quant_out), quantize]
        print(f"Running: {' '.join(qcmd)}")
        qresult = subprocess.run(qcmd, capture_output=True, text=True)
        print(qresult.stdout)
        if qresult.returncode != 0:
            print(qresult.stderr, file=sys.stderr)
            print("Quantization step failed (GGUF f16/f32 file above is still usable).", file=sys.stderr)
        else:
            print(f"Quantized GGUF written: {quant_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True, help="path to fp32 master checkpoint dir")
    ap.add_argument("--to", required=True, choices=["fp16", "bf16", "gguf"])
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--llama-cpp-dir", default="./llama.cpp",
                    help="path to a cloned llama.cpp repo (for --to gguf)")
    ap.add_argument("--gguf-outtype", default="f16", choices=["f32", "f16", "bf16", "q8_0"],
                    help="GGUF base output type before optional quantization")
    ap.add_argument("--gguf-quantize", default=None,
                    help="optional further quantization level, e.g. Q4_K_M (requires llama-quantize binary)")
    args = ap.parse_args()

    master_dir = Path(args.master)
    out_dir = Path(args.out)

    if not master_dir.exists():
        print(f"ERROR: master checkpoint not found at {master_dir}", file=sys.stderr)
        sys.exit(1)

    if args.to in ("fp16", "bf16"):
        convert_dtype(master_dir, out_dir, args.to)
    elif args.to == "gguf":
        convert_gguf(master_dir, out_dir, Path(args.llama_cpp_dir), args.gguf_outtype, args.gguf_quantize)


if __name__ == "__main__":
    main()
