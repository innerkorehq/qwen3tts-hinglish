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
import soundfile as sf
import librosa
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


def find_hiacc_pairs(hiacc_dir: Path):
    """
    Walk HiACC extracted directory looking for audio files + matching transcripts.
    HiACC's exact layout from Zenodo may vary; this looks for common patterns:
      - a transcripts.csv / .tsv / .json with columns [audio_file, text, ...]
      - or per-file .txt transcript siblings to .wav files
    Adjust this function after inspecting the actual extracted structure.
    """
    pairs = []

    # Pattern 1: look for any csv/tsv/json manifest files
    for meta_file in hiacc_dir.rglob("*"):
        if meta_file.suffix.lower() in (".csv", ".tsv", ".json", ".jsonl"):
            print(f"  found potential metadata file: {meta_file}")

    # Pattern 2: sibling .txt files next to .wav
    wav_files = list(hiacc_dir.rglob("*.wav"))
    print(f"  found {len(wav_files)} .wav files under {hiacc_dir}")
    for wav in wav_files:
        txt = wav.with_suffix(".txt")
        if txt.exists():
            text = txt.read_text(encoding="utf-8").strip()
            subset = "children" if "child" in str(wav).lower() else "adult"
            pairs.append({
                "audio": str(wav),
                "text": text,
                "lang": "hinglish",
                "speaker_id": wav.stem,
                "source": f"hiacc_{subset}",
            })

    if not pairs:
        print("  WARNING: no audio/transcript pairs found via .txt sibling pattern.")
        print("  Inspect the extracted HiACC directory structure manually and either:")
        print("    a) rename/restructure so .wav files have sibling .txt transcripts, or")
        print("    b) extend find_hiacc_pairs() to parse the actual metadata file format")
        print(f"  Extracted root: {hiacc_dir}")

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
    kept = []
    total_dur = 0.0
    for i, rec in enumerate(all_pairs):
        src = Path(rec["audio"])
        if not src.exists():
            continue
        dst = resampled_dir / rec["source"] / f"{i:08d}.wav"
        try:
            dur = resample_inplace_or_copy(src, dst)
        except Exception as e:
            print(f"  skipping {src} (error: {e})")
            continue
        if dur < MIN_DUR or dur > MAX_DUR:
            continue
        rec_out = dict(rec)
        rec_out["audio"] = str(dst.resolve())
        rec_out["duration"] = dur
        kept.append(rec_out)
        total_dur += dur
        if i % 1000 == 0:
            print(f"  ... {i}/{len(all_pairs)}, kept {len(kept)}, {total_dur/3600:.2f}hrs")

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
