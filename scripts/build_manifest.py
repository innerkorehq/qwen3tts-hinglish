#!/usr/bin/env python3
"""
Build a single unified JSONL manifest from HiACC (adult + children) and
OpenSLR-104 Hindi-English / Bengali-English code-switched transcript files,
resampling audio to 24kHz mono.

Usage:
    python3 build_manifest.py \
        --hiacc-dir ./data/raw/hiacc \
        --slr104-dir ./data/raw/slr104 \
        --out ./data/manifest_raw.jsonl \
        --eval-frac 0.02
"""
import argparse
import json
import os
import random

# Cap BLAS/OMP threads per process to 1 *before* importing numpy/librosa --
# otherwise each resampling worker spawns its own thread pool and a handful of
# processes can saturate all CPUs while ProcessPoolExecutor (added below)
# still only "shows" a few processes. With this set, parallelism comes purely
# from the process pool, so --workers actually scales across all cores.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import soundfile as sf
import librosa
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

TARGET_SR = 24000
MIN_DUR = 1.0
MAX_DUR = 30.0


def resample_inplace_or_copy(src_path: Path, dst_path: Path):
    """Load audio, resample to TARGET_SR mono, write to dst_path. Returns duration in seconds."""
    y, sr = librosa.load(str(src_path), sr=None, mono=True)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    dur = len(y) / TARGET_SR
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst_path), y, TARGET_SR, subtype="PCM_16")
    return dur


def _resample_task(task):
    """Top-level (picklable) wrapper for ProcessPoolExecutor workers."""
    i, src_str, dst_str = task
    try:
        dur = resample_inplace_or_copy(Path(src_str), Path(dst_str))
    except Exception as e:
        return i, None, str(e)
    return i, dur, None


# HiACC (Zenodo record 15551669, Corpus.zip) layout, verified by inspection:
#   Corpus/adult/audio/{train,val,test}_split/<PID+num>.wav
#   Corpus/adult/transcription/combined_output_changed_{train,val,test}_output.txt
#   Corpus/children/audio/{train,val,test}_split/<PID+num>.wav
#   Corpus/children/transcript/{train,val,test}_output.txt
# Transcript lines are "<filename>.wav, <text>"; speaker_id is the 4-char PID
# prefix of the filename (e.g. "AD09001.wav" -> "AD09", matching
# metadata/speaker_info.csv's PID column).
HIACC_TRANSCRIPT_FILES = {
    "adult": {
        "train": "transcription/combined_output_changed_train_output.txt",
        "val": "transcription/combined_output_changed_val_output.txt",
        "test": "transcription/combined_output_changed_test_output.txt",
    },
    "children": {
        "train": "transcript/train_output.txt",
        "val": "transcript/val_output.txt",
        "test": "transcript/test_output.txt",
    },
}


def find_hiacc_pairs(hiacc_dir: Path):
    """Walk the extracted HiACC Corpus/ directory and pair audio with transcripts."""
    pairs = []

    for category, split_files in HIACC_TRANSCRIPT_FILES.items():
        category_dirs = [p for p in hiacc_dir.rglob("*") if p.is_dir() and p.name.lower() == category]
        for category_dir in category_dirs:
            for split, rel_path in split_files.items():
                transcript_path = category_dir / rel_path
                if not transcript_path.exists():
                    continue
                audio_dir = category_dir / "audio" / f"{split}_split"
                with open(transcript_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or ", " not in line:
                            continue
                        fname, text = line.split(", ", 1)
                        fname = fname.strip()
                        text = text.strip()
                        if not fname or not text:
                            continue
                        audio_path = audio_dir / fname
                        if not audio_path.exists():
                            continue
                        pairs.append({
                            "audio": str(audio_path),
                            "text": text,
                            "lang": "hinglish",
                            "speaker_id": fname[:4],
                            "source": f"hiacc_{category}_{split}",
                        })

    print(f"  found {len(pairs)} HiACC audio/transcript pairs under {hiacc_dir}")
    if not pairs:
        print("  WARNING: no pairs found -- check download_hiacc.py output / Corpus/ layout")

    return pairs


def load_slr104_pairs(slr104_dir: Path):
    """
    Load transcripts.jsonl from each pair subdirectory written by
    download_slr104.py (e.g. hindi-english/, bengali-english/), each
    containing records shaped {"audio", "text", "lang", "speaker_id", "source"}.
    """
    pairs = []
    if not slr104_dir.exists():
        return pairs
    for pair_dir in slr104_dir.iterdir():
        if not pair_dir.is_dir():
            continue
        manifest = pair_dir / "transcripts.jsonl"
        if not manifest.exists():
            continue
        with open(manifest) as f:
            for line in f:
                rec = json.loads(line)
                pairs.append(rec)
        print(f"  loaded {sum(1 for _ in open(manifest))} entries from {pair_dir.name}")
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hiacc-dir", required=True)
    ap.add_argument("--slr104-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resampled-dir", default="./data/resampled",
                    help="where to write 24kHz mono copies")
    ap.add_argument("--eval-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                    help="parallel resampling workers (default: all CPUs)")
    args = ap.parse_args()

    hiacc_dir = Path(args.hiacc_dir)
    slr104_dir = Path(args.slr104_dir)
    resampled_dir = Path(args.resampled_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Scanning HiACC ...")
    hiacc_pairs = find_hiacc_pairs(hiacc_dir)
    print(f"  -> {len(hiacc_pairs)} pairs")

    print("Scanning OpenSLR-104 (Hindi-English / Bengali-English) ...")
    slr104_pairs = load_slr104_pairs(slr104_dir)
    print(f"  -> {len(slr104_pairs)} pairs")

    all_pairs = hiacc_pairs + slr104_pairs
    print(f"\nTotal raw pairs: {len(all_pairs)}")

    if not all_pairs:
        raise SystemExit("No pairs found at all — check input directories before continuing.")

    print("Resampling to 24kHz mono and filtering by duration ...")
    tasks = []
    n_missing = 0
    for i, rec in enumerate(all_pairs):
        src = Path(rec["audio"])
        if not src.exists():
            n_missing += 1
            continue
        dst = resampled_dir / rec["source"] / f"{i:08d}.wav"
        tasks.append((i, str(src), str(dst)))
    if n_missing:
        print(f"  skipping {n_missing} records with missing source audio")

    print(f"  resampling {len(tasks)} files using {args.workers} worker processes ...")

    kept = []
    total_dur = 0.0
    n_done = 0
    n_errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, dur, err in ex.map(_resample_task, tasks, chunksize=32):
            n_done += 1
            if err is not None:
                n_errors += 1
                if n_errors <= 20:
                    print(f"  skipping {all_pairs[i]['audio']} (error: {err})")
                continue
            if dur < MIN_DUR or dur > MAX_DUR:
                continue
            rec = all_pairs[i]
            dst = resampled_dir / rec["source"] / f"{i:08d}.wav"
            rec_out = dict(rec)
            rec_out["audio"] = str(dst.resolve())
            # Self-reference: each clip is its own speaker reference for the
            # Qwen3-TTS speaker encoder (multi-speaker full-FT, no grouping by
            # speaker needed).
            rec_out["ref_audio"] = rec_out["audio"]
            rec_out["duration"] = dur
            kept.append(rec_out)
            total_dur += dur
            if n_done % 1000 == 0:
                print(f"  ... {n_done}/{len(tasks)}, kept {len(kept)}, {total_dur/3600:.2f}hrs")

    print(f"\nKept {len(kept)} clips, total {total_dur/3600:.2f} hours")

    random.seed(args.seed)
    random.shuffle(kept)
    n_eval = max(1, int(len(kept) * args.eval_frac))
    eval_set = kept[:n_eval]
    train_set = kept[n_eval:]

    with open(out_path, "w") as f:
        for rec in train_set:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    eval_path = out_path.with_name(out_path.stem.replace("_raw", "") + "_eval_raw.jsonl")
    with open(eval_path, "w") as f:
        for rec in eval_set:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(train_set)} train records -> {out_path}")
    print(f"Wrote {len(eval_set)} eval records -> {eval_path}")
    print("\nNext: run encode_codes.py on both files.")


if __name__ == "__main__":
    main()
