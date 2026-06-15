#!/usr/bin/env python3
"""
Generate speech on Mac (M4/MPS) using the fine-tuned Qwen3-TTS Base
checkpoint, voice-cloned from a reference audio file.

The fine-tuned checkpoints (final_fp32/final_bf16/final_fp16, produced by
train.py + convert_model.py and uploaded to R2 under
finetune/qwen3tts-hinglish/) load like the Base model: arbitrary --ref-audio
at inference, via Qwen3TTSModel.generate_voice_clone(). final_fp16 is
recommended for MPS (broadest cross-device support).

Requires the qwen-tts package locally:
    pip install qwen-tts

By default this downloads the fine-tuned checkpoint from R2 (the same bucket
orchestrate.py uploads to) on first run and caches it locally -- subsequent
runs reuse the cached copy. Set R2_ACCOUNT_ID/R2_ACCESS_KEY_ID/
R2_SECRET_ACCESS_KEY/R2_BUCKET as usual, or pass --model to point at an
already-downloaded checkpoint directory instead.

Usage:
    # ICL mode (best quality): provide a transcript of the reference audio
    python3 scripts/generate_speech.py \
        --ref-audio ./my_voice.m4a \
        --ref-text "Yeh meri awaaz ka transcript hai." \
        --text "Yeh naya text hai jo generate karna hai." \
        --out ./out.wav

    # x-vector-only mode (no transcript needed, weaker cloning)
    python3 scripts/generate_speech.py \
        --ref-audio ./my_voice.m4a \
        --x-vector-only \
        --text "Yeh naya text hai jo generate karna hai." \
        --out ./out.wav

    # use an already-downloaded checkpoint directory directly
    python3 scripts/generate_speech.py \
        --model ./checkpoints/final_fp16 \
        --ref-audio ./my_voice.m4a --x-vector-only \
        --text "..." --out ./out.wav
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

# Some ops in the TTS codec/talker may not have MPS kernels yet -- fall back
# to CPU for those instead of crashing. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

R2_PREFIX = "finetune/qwen3tts-hinglish"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "qwen3tts-hinglish" / "checkpoints"


def resolve_checkpoint(checkpoint_name: str, cache_dir: Path) -> str:
    """Return a local checkpoint directory for `checkpoint_name` (e.g.
    "final_fp16"), downloading it from R2 into `cache_dir` on first use."""
    local_dir = cache_dir / checkpoint_name
    if (local_dir / "config.json").exists():
        print(f"Using cached checkpoint at {local_dir}")
        return str(local_dir)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from upload_to_r2 import download_dir, get_client

    bucket = os.environ.get("R2_BUCKET")
    if not bucket:
        sys.exit("ERROR: R2_BUCKET env var not set (needed to download --checkpoint from R2; "
                 "or pass --model <local dir> instead)")

    client = get_client()
    key_prefix = f"{R2_PREFIX}/{checkpoint_name}"
    print(f"Downloading checkpoint {key_prefix} from R2 to {local_dir} (cached for future runs) ...")
    download_dir(client, bucket, key_prefix, local_dir)
    return str(local_dir)


def load_reference_audio(path: str) -> str:
    """Return a path librosa/soundfile can read, converting via pydub/ffmpeg
    first if needed (m4a and other compressed formats often aren't readable
    by soundfile's libsndfile backend directly)."""
    import librosa

    try:
        librosa.load(path, sr=None, duration=0.05)
        return path
    except Exception:
        from pydub import AudioSegment

        audio = AudioSegment.from_file(path)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio.export(tmp.name, format="wav")
        tmp.close()
        return tmp.name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None,
                    help="path to an already-downloaded checkpoint dir (e.g. ./checkpoints/final_fp16). "
                         "If not given, --checkpoint is downloaded from R2 and cached instead.")
    ap.add_argument("--checkpoint", default="final_fp16",
                    help=f"checkpoint variant to download from R2 under {R2_PREFIX}/ and cache "
                         "(e.g. final_fp16, final_bf16, final_fp32, best). Ignored if --model is set. "
                         "default: final_fp16")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR),
                    help=f"local cache dir for downloaded checkpoints (default: {DEFAULT_CACHE_DIR})")
    ap.add_argument("--ref-audio", required=True, help="reference audio for voice cloning (wav, m4a, mp3, ...)")
    ap.add_argument("--ref-text", default=None, help="transcript of --ref-audio (required unless --x-vector-only)")
    ap.add_argument("--x-vector-only", action="store_true",
                    help="clone via speaker embedding only, no --ref-text needed (weaker cloning)")
    ap.add_argument("--text", required=True, help="text to synthesize")
    ap.add_argument("--language", default="Auto", help="language hint (default: Auto)")
    ap.add_argument("--out", required=True, help="output wav path")
    ap.add_argument("--device", default="mps", help="torch device: mps, cpu, cuda (default: mps)")
    ap.add_argument("--dtype", default="float16", choices=list(DTYPE_MAP), help="default: float16")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--repetition-penalty", type=float, default=None)
    args = ap.parse_args()

    if not args.x_vector_only and not args.ref_text:
        ap.error("--ref-text is required unless --x-vector-only is set")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    from qwen_tts import Qwen3TTSModel

    model_path = args.model or resolve_checkpoint(args.checkpoint, Path(args.cache_dir))

    print(f"Loading {model_path} on {args.device} ({args.dtype}) ...")
    tts = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=args.device,
        dtype=DTYPE_MAP[args.dtype],
    )

    ref_audio_path = load_reference_audio(args.ref_audio)

    gen_kwargs = {k: v for k, v in dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    ).items() if v is not None}

    print("Generating ...")
    wavs, sr = tts.generate_voice_clone(
        text=args.text,
        language=args.language,
        ref_audio=ref_audio_path,
        ref_text=args.ref_text,
        x_vector_only_mode=args.x_vector_only,
        **gen_kwargs,
    )

    import soundfile as sf

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), wavs[0], sr)
    print(f"Wrote {out_path} ({len(wavs[0]) / sr:.2f}s @ {sr}Hz)")


if __name__ == "__main__":
    main()
