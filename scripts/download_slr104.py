#!/usr/bin/env python3
"""
Download and extract Hindi-English and Bengali-English code-switched speech data
from OpenSLR-104 (MUCS 2021 subtask2): https://www.openslr.org/104/

Hindi-English train: 89.86 hrs, ~7.3GB
Bengali-English train: 46.11 hrs, ~3.9GB
(test sets also available, smaller, optional via --include-test)

Data is Kaldi-style: wav.scp (recording-id -> audio path), segments
(utterance-id, recording-id, start, end), text (utterance-id -> transcript).
This script extracts archives, parses these files, slices per-utterance audio
clips using the segments timestamps, and writes a transcripts.jsonl per
language pair matching the format build_manifest.py expects:
{"audio", "text", "lang", "speaker_id", "source"}.

16kHz 16-bit source audio is resampled to 24kHz mono by build_manifest.py later
(this script just slices segments at native sample rate).

Usage:
    python3 download_slr104.py --out-dir ./data/raw/slr104
    python3 download_slr104.py --out-dir ./data/raw/slr104 --include-test
    python3 download_slr104.py --out-dir ./data/raw/slr104 --pairs hindi-english
"""
import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import soundfile as sf

BASE_URL = "https://openslr.trmal.net/resources/104"

ARCHIVES = {
    "hindi-english": {
        "train": ("Hindi-English_train.tar.gz", "7.3G"),
        "test": ("Hindi-English_test.tar.gz", "443M"),
        "lang": "hi-en",
    },
    "bengali-english": {
        "train": ("Bengali-English_train.tar.gz", "3.9G"),
        "test": ("Bengali-English_test.tar.gz", "606M"),
        "lang": "bn-en",
    },
}


def download_file(url, dest_path):
    if dest_path.exists():
        print(f"  already exists, skipping download: {dest_path.name}")
        return
    print(f"  downloading {dest_path.name} (this is a large file, may take a while) ...")
    # Use curl for resumable download with progress; fall back to requests if unavailable.
    try:
        subprocess.run(
            ["curl", "-fL", "-C", "-", "-o", str(dest_path), url],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        import requests
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)


def extract_archive(path, out_dir):
    print(f"  extracting {path.name} ...")
    with tarfile.open(path, "r:gz") as t:
        t.extractall(out_dir)


def find_kaldi_dir(root: Path):
    """
    Locate the directory containing wav.scp, segments, and text within an
    extracted archive (structure may vary; search a couple levels deep).
    """
    for candidate in root.rglob("wav.scp"):
        d = candidate.parent
        if (d / "segments").exists() and (d / "text").exists():
            return d
    return None


def parse_wav_scp(path: Path):
    """
    wav.scp lines look like either:
      recording-id /path/to/file.wav
    or a pipe command:
      recording-id sox /path/to/file.sph -t wav - |
    Returns dict: recording-id -> resolved wav path (extracting from pipe cmds
    if needed) OR a callable to materialize it. For simplicity, this handles
    the common direct-path case and the `sox ... |` pipe case by running the
    command to produce a temp wav.
    """
    wav_map = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) != 2:
                continue
            rec_id, rest = parts
            rest = rest.strip()
            if rest.endswith("|"):
                # pipe command — store the command, resolve lazily
                wav_map[rec_id] = ("cmd", rest.rstrip("|").strip())
            else:
                wav_map[rec_id] = ("path", rest)
    return wav_map


def resolve_recording(wav_map_entry, base_dir: Path, cache_dir: Path, rec_id: str):
    """Return a Path to a readable wav file for this recording."""
    kind, val = wav_map_entry
    if kind == "path":
        p = Path(val)
        if p.is_absolute():
            return p
        # Relative paths in OpenSLR-104's wav.scp are relative to the split
        # root (e.g. "test/" or "train/"), NOT the "transcripts/" subdirectory
        # that contains wav.scp/segments/text. Try a few candidate bases.
        candidates = [
            base_dir / val,                 # relative to wav.scp's own dir (rare)
            base_dir.parent / val,          # relative to split root (common case)
            base_dir.parent.parent / val,   # one level further up, just in case
        ]
        for c in candidates:
            if c.exists():
                return c
        # None found — return the most likely candidate anyway so the caller's
        # existence check produces a clear "not found" skip rather than a
        # silent wrong path.
        return candidates[1]
    else:
        # pipe command, e.g. "sox /abs/path.sph -t wav -"
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = cache_dir / f"{rec_id}.wav"
        if out_path.exists():
            return out_path
        cmd = val
        # The command typically ends with "-" (stdout); redirect to file.
        full_cmd = f"{cmd} > {out_path}"
        try:
            subprocess.run(full_cmd, shell=True, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"    WARNING: failed to resolve recording {rec_id} via pipe cmd: {e}")
            return None
        return out_path


def parse_segments(path: Path):
    """segments: utterance-id recording-id start-time end-time (seconds)"""
    segs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            utt_id, rec_id, start, end = parts
            segs.append((utt_id, rec_id, float(start), float(end)))
    return segs


def parse_text(path: Path):
    """text: utterance-id transcript..."""
    texts = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) != 2:
                continue
            utt_id, transcript = parts
            texts[utt_id] = transcript
    return texts


def process_pair(pair_name, lang_code, kaldi_dir, out_dir, split):
    print(f"  parsing Kaldi data dir: {kaldi_dir}")
    wav_map = parse_wav_scp(kaldi_dir / "wav.scp")
    segments = parse_segments(kaldi_dir / "segments")
    texts = parse_text(kaldi_dir / "text")

    audio_out_dir = out_dir / "audio"
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "transcripts.jsonl"

    # Cache resolved (decoded) recordings by recording-id so multi-segment
    # recordings don't get decoded repeatedly.
    recording_cache = {}
    cache_dir = out_dir / "_recording_cache"

    total_dur = 0.0
    n_written = 0
    n_skipped = 0

    mode = "a" if manifest_path.exists() else "w"
    with open(manifest_path, mode) as mf:
        for i, (utt_id, rec_id, start, end) in enumerate(segments):
            text = texts.get(utt_id)
            if not text:
                n_skipped += 1
                continue

            if rec_id not in recording_cache:
                entry = wav_map.get(rec_id)
                if entry is None:
                    n_skipped += 1
                    continue
                resolved = resolve_recording(entry, kaldi_dir, cache_dir, rec_id)
                if resolved is None or not resolved.exists():
                    n_skipped += 1
                    continue
                recording_cache[rec_id] = resolved

            rec_path = recording_cache[rec_id]

            try:
                info = sf.info(str(rec_path))
                sr = info.samplerate
                start_frame = int(start * sr)
                end_frame = int(end * sr)
                if end_frame <= start_frame:
                    n_skipped += 1
                    continue
                audio, _ = sf.read(str(rec_path), start=start_frame,
                                    stop=end_frame, dtype="float32")
            except Exception as e:
                n_skipped += 1
                if n_skipped <= 5:
                    print(f"    WARNING: failed to slice {utt_id} from {rec_path}: {e}")
                continue

            out_fname = f"{pair_name}_{split}_{utt_id}.wav"
            out_path = audio_out_dir / out_fname
            sf.write(str(out_path), audio, sr)

            rec = {
                "audio": str(out_path.resolve()),
                "text": text,
                "lang": lang_code,
                "speaker_id": rec_id,
                "source": f"slr104_{pair_name}_{split}",
            }
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            dur = (end_frame - start_frame) / sr
            total_dur += dur
            n_written += 1

            if i % 2000 == 0:
                print(f"    ... {i}/{len(segments)} segments, "
                      f"{n_written} written, {total_dur/3600:.2f}hrs so far")

    # Clean up decoded recording cache (large intermediate sph->wav files)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    print(f"  {pair_name}/{split}: wrote {n_written} clips "
          f"({total_dur/3600:.2f}hrs), skipped {n_skipped}")
    return total_dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pairs", nargs="+", default=["hindi-english", "bengali-english"],
                    choices=list(ARCHIVES.keys()))
    ap.add_argument("--include-test", action="store_true",
                    help="also download+process the (smaller) test split")
    ap.add_argument("--keep-archives", action="store_true",
                    help="don't delete downloaded tar.gz after extraction (uses more disk)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    archives_dir = out_dir / "_archives"
    archives_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train"] + (["test"] if args.include_test else [])

    for pair_name in args.pairs:
        cfg = ARCHIVES[pair_name]
        pair_out_dir = out_dir / pair_name
        pair_out_dir.mkdir(parents=True, exist_ok=True)

        for split in splits:
            fname, size_hint = cfg[split]
            url = f"{BASE_URL}/{fname}"
            archive_path = archives_dir / fname

            print(f"\n=== {pair_name} / {split} ({size_hint}) ===")
            download_file(url, archive_path)

            extract_dir = out_dir / "_extracted" / pair_name / split
            extract_dir.mkdir(parents=True, exist_ok=True)
            extract_archive(archive_path, extract_dir)

            if not args.keep_archives:
                archive_path.unlink()

            kaldi_dir = find_kaldi_dir(extract_dir)
            if kaldi_dir is None:
                print(f"  WARNING: could not find wav.scp+segments+text under {extract_dir}")
                print("  Inspect the extracted structure manually and adjust find_kaldi_dir().")
                continue

            process_pair(pair_name, cfg["lang"], kaldi_dir, pair_out_dir, split)

            # Clean up extracted raw archive contents (we've written what we need
            # to pair_out_dir/audio + transcripts.jsonl)
            shutil.rmtree(extract_dir, ignore_errors=True)

    print("\nDone. Per-pair transcripts.jsonl + audio/ written under:")
    for pair_name in args.pairs:
        print(f"  {out_dir / pair_name}")


if __name__ == "__main__":
    main()
