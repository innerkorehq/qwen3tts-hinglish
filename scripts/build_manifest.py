#!/usr/bin/env python3
"""
Build a single unified JSONL manifest from HiACC (adult + children),
OpenSLR-104 Hindi-English, and optionally HuggingFace audio datasets,
resampling audio to 24kHz mono.

Usage:
    python3 build_manifest.py \
        --hiacc-dir ./data/raw/hiacc \
        --slr104-dir ./data/raw/slr104 \
        --out ./data/manifest_raw.jsonl \
        --eval-frac 0.02

    # With HF datasets (requires `datasets` package and internet):
    python3 build_manifest.py \
        --hiacc-dir ./data/raw/hiacc \
        --slr104-dir ./data/raw/slr104 \
        --indic-voices \
        --hinglish-casual \
        --hf-raw-dir ./data/raw/hf \
        --out ./data/manifest_raw.jsonl
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

import re
import unicodedata
import numpy as np
import soundfile as sf
import librosa
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

TARGET_SR = 24000
MIN_DUR = 2.0
MAX_DUR = 15.0
SNR_MIN_DB = 20.0

# Strip paraverbal markers like <sigh>, <chuckle>, <laugh> from hinglish-casual.
_PARAVERBAL_RE = re.compile(r"<[^>]+>")

REF_DUR_MIN = 4.0
REF_DUR_MAX = 10.0
REF_DUR_TARGET = 7.0

# Devanagari Unicode block (U+0900–U+097F) + Devanagari Extended (U+A8E0–U+A8FF).
# Also catches Devanagari digits ०-९ (U+0966–U+096F) which are inside the block.
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ꣠-ꣿ]+")
# Zero-width characters sometimes embedded in Devanagari text.
_ZWX_RE = re.compile(r"[​-‍﻿]")


def normalize_hinglish(text: str) -> str:
    """Normalize any mix of Devanagari + Roman Hinglish + English to lowercase Roman.

    Strategy:
    - NFC-normalize so Devanagari matras are composed (prevents split codepoints).
    - Strip zero-width joiners/non-joiners that confuse ITRANS.
    - Replace each Devanagari span (word or mid-word) with its ITRANS transliteration.
    - Lowercase everything: removes ITRANS retroflex caps (T, D, Th, Dh → t, d, th, dh)
      so the text tokenizer sees only familiar lowercase Latin characters throughout.

    Examples:
      "कल मेरा interview था"  -> "kal meraa interview thaa"
      "Kal mera interview tha" -> "kal mera interview tha"
      "कल mera interview था"  -> "kal mera interview thaa"
      "मुझे prepare करना है"  -> "mujhe prepare karanaa hai"
    """
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate as _xlit
    except ImportError:
        # indic_transliteration not installed: pass text through unchanged.
        # bootstrap_vastai.sh installs it; this path only fires in local dev.
        return text.lower()

    text = unicodedata.normalize("NFC", text)
    text = _ZWX_RE.sub("", text)

    def _replace(m):
        return _xlit(m.group(), sanscript.DEVANAGARI, sanscript.ITRANS)

    text = _DEVANAGARI_RE.sub(_replace, text)
    return text.lower()


def estimate_snr(y: np.ndarray, sr: int, frame_ms: int = 20) -> float:
    """Estimate SNR in dB using the quietest 20% of frames as the noise floor.

    Returns 0.0 for near-silent audio (will be caught by duration filter anyway).
    """
    frame_len = int(sr * frame_ms / 1000)
    if len(y) < frame_len:
        return 0.0
    n_frames = len(y) // frame_len
    frames = y[:n_frames * frame_len].reshape(n_frames, frame_len)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    n_noise = max(1, int(n_frames * 0.20))
    noise_rms = np.mean(np.sort(frame_rms)[:n_noise])
    signal_rms = np.sqrt(np.mean(y ** 2))
    if noise_rms < 1e-10:
        return 0.0
    return float(20.0 * np.log10(signal_rms / noise_rms))


def resample_inplace_or_copy(src_path: Path, dst_path: Path):
    """Load audio, resample to TARGET_SR mono, write to dst_path. Returns (y, duration)."""
    y, sr = librosa.load(str(src_path), sr=None, mono=True)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    dur = len(y) / TARGET_SR
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst_path), y, TARGET_SR, subtype="PCM_16")
    return y, dur


def _resample_task(task):
    """Top-level (picklable) wrapper for ProcessPoolExecutor workers.

    Returns (i, dur, snr, err). On resume, loads existing dst to compute SNR.
    """
    i, src_str, dst_str = task
    dst = Path(dst_str)
    if dst.exists() and dst.stat().st_size > 0:
        try:
            y, sr = sf.read(str(dst), dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            dur = len(y) / sr
            snr = estimate_snr(y, sr)
            return i, dur, snr, None
        except Exception:
            pass

    try:
        y, dur = resample_inplace_or_copy(Path(src_str), dst)
        if not dst.exists() or dst.stat().st_size == 0:
            return i, None, None, f"resampled file missing or empty after write: {dst}"
        snr = estimate_snr(y, TARGET_SR)
    except Exception as e:
        return i, None, None, str(e)
    return i, dur, snr, None


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


def load_hf_audio_dataset(
    dataset_name: str,
    raw_dir: Path,
    splits=("train",),
    audio_col: str = "audio",
    text_col: str = "text",
    speaker_col: str = None,
    lang: str = "hinglish",
    source_name: str = None,
    text_filter=None,
    hf_cache_dir: str = None,
    max_rows: int = None,
) -> list:
    """Stream a HuggingFace audio dataset, save raw WAV files, return pairs.

    Each row's audio array is saved as a WAV file under raw_dir/source/.
    A pairs.jsonl cache is written next to the WAVs so re-runs skip the
    download entirely (even across machines if raw_dir is on shared storage).

    Args:
        dataset_name:  HuggingFace dataset id (e.g. "dianavdavidson/indic-voices-...")
        raw_dir:       root dir to store raw WAV files under raw_dir/source/
        splits:        HF splits to load, concatenated in order
        audio_col:     column containing audio (dict with "array"/"sampling_rate")
        text_col:      column containing transcript text
        speaker_col:   column containing speaker id (optional)
        lang:          language tag to write into the manifest record
        source_name:   override the source field (default: last component of dataset_name)
        text_filter:   callable(str) -> str applied to each transcript before saving
        hf_cache_dir:  HF datasets cache directory
        max_rows:      cap total rows across all splits (None = no cap)
    """
    try:
        from datasets import load_dataset, concatenate_datasets
    except ImportError:
        print(f"  WARNING: `datasets` not installed, skipping {dataset_name}")
        return []

    source = source_name or dataset_name.split("/")[-1].replace("-", "_")
    out_dir = raw_dir / source
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume: pairs cache exists → skip download entirely
    pairs_cache = out_dir / "pairs.jsonl"
    if pairs_cache.exists():
        pairs = []
        with open(pairs_cache) as fh:
            for line in fh:
                pairs.append(json.loads(line))
        print(f"  [{source}] resumed {len(pairs)} pairs from cache")
        return pairs

    print(f"  [{source}] loading {dataset_name} splits={splits} ...")
    ds_splits = []
    for sp in splits:
        try:
            ds_splits.append(load_dataset(
                dataset_name, split=sp,
                cache_dir=hf_cache_dir,
                trust_remote_code=True,
            ))
        except Exception as e:
            print(f"  [{source}] WARNING: could not load split '{sp}': {e}")
    if not ds_splits:
        return []
    ds = concatenate_datasets(ds_splits) if len(ds_splits) > 1 else ds_splits[0]

    if max_rows and len(ds) > max_rows:
        ds = ds.select(range(max_rows))
        print(f"  [{source}] capped at {max_rows} rows")

    pairs = []
    n_skip = 0
    for i, row in enumerate(ds):
        audio_info = row.get(audio_col)
        if not audio_info:
            n_skip += 1
            continue

        text = str(row.get(text_col, "") or "").strip()
        if text_filter:
            text = text_filter(text)
        text = text.strip()
        if not text:
            n_skip += 1
            continue

        spk = str(row[speaker_col]) if speaker_col and speaker_col in row else f"{source}_{i:06d}"

        audio_path = out_dir / f"{i:08d}.wav"
        if not audio_path.exists():
            try:
                arr = np.array(audio_info["array"], dtype=np.float32)
                sr = int(audio_info["sampling_rate"])
                sf.write(str(audio_path), arr, sr)
            except Exception as e:
                print(f"  [{source}] skipping row {i}: {e}")
                n_skip += 1
                continue

        pairs.append({
            "audio": str(audio_path),
            "text": text,
            "lang": lang,
            "speaker_id": spk,
            "source": source,
        })

        if (i + 1) % 10000 == 0:
            print(f"  [{source}] {i+1}/{len(ds)} rows, {len(pairs)} kept ...")

    # Write cache so re-runs skip the download
    with open(pairs_cache, "w") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"  [{source}] done: {len(pairs)} pairs ({n_skip} skipped) -> {out_dir}")
    return pairs


def load_indic_voices_pairs(raw_dir: Path, hf_cache_dir: str = None, max_rows: int = 100_000) -> list:
    """Load dianavdavidson/indic-voices-hinglish-nospeakeroverlap-spon.

    Uses 'normalized' text (cleaner than verbatim). Loads train + validation.
    max_rows caps the total across both splits (default 100k).
    """
    return load_hf_audio_dataset(
        dataset_name="dianavdavidson/indic-voices-hinglish-nospeakeroverlap-spon",
        raw_dir=raw_dir,
        splits=("train", "validation"),
        audio_col="audio",
        text_col="normalized",
        speaker_col="speaker_id",
        lang="hinglish",
        source_name="indic_voices",
        hf_cache_dir=hf_cache_dir,
        max_rows=max_rows,
    )


def load_hinglish_casual_pairs(raw_dir: Path, hf_cache_dir: str = None) -> list:
    """Load tiny-aya-translate/hinglish-casual.

    Uses 'utterance_latin' (already romanized). Strips paraverbal markers.
    7 named speakers.
    """
    def _clean(text: str) -> str:
        text = _PARAVERBAL_RE.sub("", text)
        return " ".join(text.split())

    return load_hf_audio_dataset(
        dataset_name="tiny-aya-translate/hinglish-casual",
        raw_dir=raw_dir,
        splits=("train",),
        audio_col="audio",
        text_col="utterance_latin",
        speaker_col="speaker",
        lang="hinglish",
        source_name="hinglish_casual",
        text_filter=_clean,
        hf_cache_dir=hf_cache_dir,
    )


def _stats(values):
    if not values:
        return {}
    a = np.array(values, dtype=np.float64)
    return {
        "count": int(len(a)),
        "mean": float(round(a.mean(), 2)),
        "median": float(round(np.median(a), 2)),
        "std": float(round(a.std(), 2)),
        "min": float(round(a.min(), 2)),
        "max": float(round(a.max(), 2)),
        "p10": float(round(np.percentile(a, 10), 2)),
        "p90": float(round(np.percentile(a, 90), 2)),
    }


def _histogram(values, edges):
    """Return list of (label, count, pct) for each bin defined by edges."""
    a = np.array(values, dtype=np.float64)
    total = len(a)
    buckets = []
    for lo, hi in zip(edges, edges[1:]):
        count = int(((a >= lo) & (a < hi)).sum())
        label = f"{lo}-{hi}"
        buckets.append({"range": label, "count": count,
                         "pct": round(count / total * 100, 1) if total else 0.0})
    # last bucket is inclusive on the right
    count = int((a >= edges[-2]).sum())
    buckets[-1]["count"] = count
    buckets[-1]["pct"] = round(count / total * 100, 1) if total else 0.0
    return buckets


def build_analytics(train_set, eval_set):
    from collections import Counter
    all_recs = train_set + eval_set

    durations = [r["duration"] for r in all_recs]
    snrs = [r["snr"] for r in all_recs]
    txt_lens = [len(r["text"]) for r in all_recs]

    speaker_counts = Counter(r["speaker_id"] for r in all_recs)
    clips_per_spk = list(speaker_counts.values())

    source_stats = {}
    src_counter = Counter(r["source"] for r in all_recs)
    for src in sorted(src_counter):
        recs = [r for r in all_recs if r["source"] == src]
        source_stats[src] = {
            "clips": len(recs),
            "hours": round(sum(r["duration"] for r in recs) / 3600, 2),
        }

    lang_stats = {}
    for r in all_recs:
        lang = r.get("lang", "unknown")
        lang_stats.setdefault(lang, {"clips": 0, "hours": 0.0})
        lang_stats[lang]["clips"] += 1
        lang_stats[lang]["hours"] = round(lang_stats[lang]["hours"] + r["duration"] / 3600, 4)

    return {
        "overall": {
            "total_clips": len(all_recs),
            "total_hours": round(sum(durations) / 3600, 2),
            "train_clips": len(train_set),
            "train_hours": round(sum(r["duration"] for r in train_set) / 3600, 2),
            "eval_clips": len(eval_set),
            "eval_hours": round(sum(r["duration"] for r in eval_set) / 3600, 2),
            "unique_speakers": len(speaker_counts),
        },
        "duration": {
            "stats": _stats(durations),
            "histogram_s": _histogram(durations, [2, 4, 6, 8, 10, 12, 15]),
        },
        "snr": {
            "stats": _stats(snrs),
            "histogram_db": _histogram(snrs, [20, 25, 30, 35, 40, 45, 100]),
        },
        "text_length": {
            "stats": _stats(txt_lens),
            "histogram_chars": _histogram(txt_lens, [0, 50, 100, 150, 200, 300, 10000]),
        },
        "speakers": {
            "stats": _stats(clips_per_spk),
            "single_clip": int(sum(1 for c in clips_per_spk if c == 1)),
            "lt5_clips": int(sum(1 for c in clips_per_spk if c < 5)),
        },
        "source": source_stats,
        "language": lang_stats,
    }


def _bar(pct, width=24):
    filled = int(round(pct / 100 * width))
    return "█" * filled + "░" * (width - filled)


def print_analytics(a):
    o = a["overall"]
    print("\n" + "=" * 60)
    print("  TRAINING DATA ANALYTICS")
    print("=" * 60)
    print(f"  Total   {o['total_clips']:>7,} clips   {o['total_hours']:>7.1f}h")
    print(f"  Train   {o['train_clips']:>7,} clips   {o['train_hours']:>7.1f}h")
    print(f"  Eval    {o['eval_clips']:>7,} clips   {o['eval_hours']:>7.1f}h")
    print(f"  Speakers           {o['unique_speakers']:>6,} unique")

    def section(title, stat, unit, hist, hist_unit):
        s = stat["stats"]
        print(f"\n  {title}")
        print(f"    mean {s['mean']}{unit}  median {s['median']}{unit}  "
              f"std {s['std']}{unit}  p10 {s['p10']}{unit}  p90 {s['p90']}{unit}")
        for b in hist:
            print(f"    {b['range']:>10}{hist_unit}  {_bar(b['pct'])}  "
                  f"{b['count']:>6,}  ({b['pct']:>5.1f}%)")

    section("Duration (s)", a["duration"], "s",
            a["duration"]["histogram_s"], "s")
    section("SNR (dB)", a["snr"], "dB",
            a["snr"]["histogram_db"], "dB")
    section("Transcript length (chars)", a["text_length"], "",
            a["text_length"]["histogram_chars"], "c")

    sp = a["speakers"]
    s = sp["stats"]
    print(f"\n  Clips per speaker")
    print(f"    mean {s['mean']:.1f}  median {s['median']:.1f}  "
          f"min {s['min']:.0f}  max {s['max']:.0f}")
    print(f"    Single-clip speakers : {sp['single_clip']} "
          f"({sp['single_clip']/s['count']*100:.1f}%)")
    print(f"    Speakers with <5 clips: {sp['lt5_clips']} "
          f"({sp['lt5_clips']/s['count']*100:.1f}%)")

    print(f"\n  Source breakdown")
    for src, v in a["source"].items():
        print(f"    {src:<30}  {v['clips']:>6,} clips  {v['hours']:>6.1f}h")

    print(f"\n  Language breakdown")
    for lang, v in a["language"].items():
        print(f"    {lang:<20}  {v['clips']:>6,} clips  {v['hours']:.1f}h")

    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hiacc-dir", default=None,
                    help="extracted HiACC Corpus/ directory (optional if --base-manifest provided)")
    ap.add_argument("--slr104-dir", default=None,
                    help="OpenSLR-104 output directory (optional if --base-manifest provided)")
    ap.add_argument("--base-manifest", default=None,
                    help="existing train manifest JSONL to load as a starting point "
                         "(both train and paired eval file are loaded; only new sources are processed). "
                         "Use when resuming: existing resampled audio is already present, "
                         "only HF datasets need to be added.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resampled-dir", default="./data/resampled",
                    help="where to write 24kHz mono copies")
    ap.add_argument("--eval-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                    help="parallel resampling workers (default: all CPUs)")
    ap.add_argument("--min-snr", type=float, default=SNR_MIN_DB,
                    help="drop clips with estimated SNR below this dB threshold (default: %(default)s)")
    # HuggingFace dataset flags
    ap.add_argument("--indic-voices", action="store_true",
                    help="include dianavdavidson/indic-voices-hinglish-nospeakeroverlap-spon")
    ap.add_argument("--indic-voices-max-rows", type=int, default=100_000,
                    help="cap on indic-voices rows across train+val (default: 100000)")
    ap.add_argument("--hinglish-casual", action="store_true",
                    help="include tiny-aya-translate/hinglish-casual")
    ap.add_argument("--hf-raw-dir", default="./data/raw/hf",
                    help="directory for raw WAV files saved from HF datasets (default: ./data/raw/hf)")
    ap.add_argument("--hf-cache-dir", default=None,
                    help="HuggingFace datasets cache directory (default: ~/.cache/huggingface)")
    args = ap.parse_args()

    hiacc_dir = Path(args.hiacc_dir) if args.hiacc_dir else None
    slr104_dir = Path(args.slr104_dir) if args.slr104_dir else None
    hf_raw_dir = Path(args.hf_raw_dir)
    resampled_dir = Path(args.resampled_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load pre-existing records from a previous run (already resampled + filtered).
    # These are passed through without re-resampling; only new sources are processed below.
    base_kept = []
    if args.base_manifest:
        base_path = Path(args.base_manifest)
        eval_stem = base_path.stem.replace("_raw", "") + "_eval_raw.jsonl"
        eval_path = base_path.with_name(eval_stem)
        for p in [base_path, eval_path]:
            if p.exists():
                with open(p) as fh:
                    for line in fh:
                        base_kept.append(json.loads(line))
        print(f"Loaded {len(base_kept)} records from base manifest (will be kept as-is)")

    all_pairs = []
    if hiacc_dir and hiacc_dir.exists():
        print("Scanning HiACC ...")
        hiacc_pairs = find_hiacc_pairs(hiacc_dir)
        print(f"  -> {len(hiacc_pairs)} pairs")
        all_pairs += hiacc_pairs
    elif not args.base_manifest:
        print("WARNING: --hiacc-dir not provided and no --base-manifest; skipping HiACC")

    if slr104_dir and slr104_dir.exists():
        print("Scanning OpenSLR-104 (Hindi-English) ...")
        slr104_pairs = load_slr104_pairs(slr104_dir)
        print(f"  -> {len(slr104_pairs)} pairs")
        all_pairs += slr104_pairs
    elif not args.base_manifest:
        print("WARNING: --slr104-dir not provided and no --base-manifest; skipping SLR104")

    if args.indic_voices:
        print("Loading indic-voices from HuggingFace ...")
        iv_pairs = load_indic_voices_pairs(
            hf_raw_dir, hf_cache_dir=args.hf_cache_dir,
            max_rows=args.indic_voices_max_rows,
        )
        print(f"  -> {len(iv_pairs)} pairs")
        all_pairs += iv_pairs

    if args.hinglish_casual:
        print("Loading hinglish-casual from HuggingFace ...")
        hc_pairs = load_hinglish_casual_pairs(hf_raw_dir, hf_cache_dir=args.hf_cache_dir)
        print(f"  -> {len(hc_pairs)} pairs")
        all_pairs += hc_pairs

    print(f"\nTotal raw pairs (new sources): {len(all_pairs)}")
    if base_kept:
        print(f"Total base records (from --base-manifest): {len(base_kept)}")

    if not all_pairs and not base_kept:
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
    if not tasks:
        print("  no new pairs to resample (all records from --base-manifest)")

    print(f"  resampling {len(tasks)} files using {args.workers} worker processes "
          f"(files already present in --resampled-dir, e.g. from a --resume R2 "
          f"download, are reused as-is) ...")

    kept = []
    total_dur = 0.0
    n_done = 0
    n_errors = 0
    n_dur_filtered = 0
    n_snr_filtered = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, dur, snr, err in (ex.map(_resample_task, tasks, chunksize=32) if tasks else []):
            n_done += 1
            if err is not None:
                n_errors += 1
                if n_errors <= 20:
                    print(f"  skipping {all_pairs[i]['audio']} (error: {err})")
                continue
            if dur < MIN_DUR or dur > MAX_DUR:
                n_dur_filtered += 1
                continue
            if snr < args.min_snr:
                n_snr_filtered += 1
                continue
            rec = all_pairs[i]
            dst = resampled_dir / rec["source"] / f"{i:08d}.wav"
            rec_out = dict(rec)
            rec_out["audio"] = str(dst.resolve())
            rec_out["duration"] = dur
            rec_out["snr"] = round(snr, 1)
            rec_out["text"] = normalize_hinglish(rec_out["text"])
            kept.append(rec_out)
            total_dur += dur
            if n_done % 1000 == 0:
                print(f"  ... {n_done}/{len(tasks)}, kept {len(kept)}, "
                      f"dur_filtered {n_dur_filtered}, snr_filtered {n_snr_filtered}, "
                      f"{total_dur/3600:.2f}hrs")

    print(f"\nKept {len(kept)} new clips, total {total_dur/3600:.2f} hours")
    print(f"Filtered: {n_dur_filtered} by duration, {n_snr_filtered} by SNR (<{args.min_snr}dB), "
          f"{n_errors} errors")

    # Assign ref_audio for NEW records only (base_kept records already have it set).
    # Use only each new record's own speaker pool to avoid cross-corpus ref_audio.
    from collections import defaultdict
    speaker_clips = defaultdict(list)
    for rec in kept:
        speaker_clips[rec["speaker_id"]].append((rec["audio"], rec["duration"]))

    rng = random.Random(args.seed)
    n_self_ref = 0
    n_in_range = 0
    for rec in kept:
        candidates = [(p, d) for p, d in speaker_clips[rec["speaker_id"]]
                      if p != rec["audio"]]
        if not candidates:
            rec["ref_audio"] = rec["audio"]
            n_self_ref += 1
            continue
        in_range = [(p, d) for p, d in candidates if REF_DUR_MIN <= d <= REF_DUR_MAX]
        if in_range:
            rec["ref_audio"] = rng.choice(in_range)[0]
            n_in_range += 1
        else:
            rec["ref_audio"] = min(candidates, key=lambda x: abs(x[1] - REF_DUR_TARGET))[0]

    if kept:
        print(f"ref_audio (new records): {n_in_range} in {REF_DUR_MIN}-{REF_DUR_MAX}s range, "
              f"{len(kept) - n_self_ref - n_in_range} nearest-to-{REF_DUR_TARGET}s fallback, "
              f"{n_self_ref} self-reference (single-clip speakers)")

    # Merge base records with new records and re-shuffle/re-split.
    # Base records are not re-normalized or re-assigned ref_audio.
    all_kept = base_kept + kept
    random.seed(args.seed)
    random.shuffle(all_kept)
    n_eval = max(1, int(len(all_kept) * args.eval_frac))
    eval_set = all_kept[:n_eval]
    train_set = all_kept[n_eval:]

    with open(out_path, "w") as f:
        for rec in train_set:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    eval_path = out_path.with_name(out_path.stem.replace("_raw", "") + "_eval_raw.jsonl")
    with open(eval_path, "w") as f:
        for rec in eval_set:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(train_set)} train records -> {out_path}")
    print(f"Wrote {len(eval_set)} eval records -> {eval_path}")

    analytics = build_analytics(train_set, eval_set)
    print_analytics(analytics)
    analytics_path = out_path.with_name("manifest_analytics.json")
    with open(analytics_path, "w") as f:
        json.dump(analytics, f, indent=2, ensure_ascii=False)
    print(f"\nAnalytics saved -> {analytics_path}")
    print("\nNext: run encode_codes.py on both files.")


if __name__ == "__main__":
    main()
