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

import numpy as np
import soundfile as sf
import librosa
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

TARGET_SR = 24000
MIN_DUR = 2.0
MAX_DUR = 15.0
SNR_MIN_DB = 20.0

REF_DUR_MIN = 4.0
REF_DUR_MAX = 10.0
REF_DUR_TARGET = 7.0


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
    ap.add_argument("--hiacc-dir", required=True)
    ap.add_argument("--slr104-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resampled-dir", default="./data/resampled",
                    help="where to write 24kHz mono copies")
    ap.add_argument("--eval-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                    help="parallel resampling workers (default: all CPUs)")
    ap.add_argument("--min-snr", type=float, default=SNR_MIN_DB,
                    help="drop clips with estimated SNR below this dB threshold (default: %(default)s)")
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
        for i, dur, snr, err in ex.map(_resample_task, tasks, chunksize=32):
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
            kept.append(rec_out)
            total_dur += dur
            if n_done % 1000 == 0:
                print(f"  ... {n_done}/{len(tasks)}, kept {len(kept)}, "
                      f"dur_filtered {n_dur_filtered}, snr_filtered {n_snr_filtered}, "
                      f"{total_dur/3600:.2f}hrs")

    print(f"\nKept {len(kept)} clips, total {total_dur/3600:.2f} hours")
    print(f"Filtered: {n_dur_filtered} by duration, {n_snr_filtered} by SNR (<{args.min_snr}dB), "
          f"{n_errors} errors")

    # Assign ref_audio: for each clip, pick a different utterance from the same
    # speaker. Prefer clips in REF_DUR_MIN..REF_DUR_MAX (4-10s) so the speaker
    # encoder gets enough phoneme diversity without noise accumulation. If no
    # candidate falls in that range, pick the one closest to REF_DUR_TARGET (7s).
    # Falls back to self-reference only for single-clip speakers.
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

    print(f"ref_audio: {n_in_range} in {REF_DUR_MIN}-{REF_DUR_MAX}s range, "
          f"{len(kept) - n_self_ref - n_in_range} nearest-to-{REF_DUR_TARGET}s fallback, "
          f"{n_self_ref} self-reference (single-clip speakers)")

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

    analytics = build_analytics(train_set, eval_set)
    print_analytics(analytics)
    analytics_path = out_path.with_name("manifest_analytics.json")
    with open(analytics_path, "w") as f:
        json.dump(analytics, f, indent=2, ensure_ascii=False)
    print(f"\nAnalytics saved -> {analytics_path}")
    print("\nNext: run encode_codes.py on both files.")


if __name__ == "__main__":
    main()
