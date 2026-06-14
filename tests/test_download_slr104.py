#!/usr/bin/env python3
"""
Test download_slr104.py on a local Mac (M4 or any machine with Python + the
project's deps — no GPU needed).

Two modes:

1. OFFLINE (default) — builds a small synthetic Kaldi-format dataset
   (wav.scp + segments + text + a couple of generated .wav files) matching the
   structure download_slr104.py expects, runs the parsing/slicing functions
   against it, and checks the output transcripts.jsonl + audio clips are
   correct. No network access needed. This validates the PARSING LOGIC.

2. LIVE (--live) — runs the real download_slr104.py against the actual
   OpenSLR-104 mirror for ONE pair with --include-test (smaller test split,
   not the multi-GB train split), to confirm find_kaldi_dir() and
   resolve_recording() work against the REAL archive structure. This is the
   piece flagged as unverified in docs/RUNBOOK.md. Downloads ~440MB-600MB
   (a test split tarball) — not huge, but not free; run this once to validate
   before committing to the full pipeline run on Vast.ai.

Usage:
    python3 tests/test_download_slr104.py              # offline only
    python3 tests/test_download_slr104.py --live       # offline + live smoke test
    python3 tests/test_download_slr104.py --live --pair bengali-english
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import download_slr104 as slr104  # noqa: E402


# ---------------------------------------------------------------------------
# OFFLINE TEST: synthetic Kaldi dataset
# ---------------------------------------------------------------------------

def make_synthetic_kaldi_dir(root: Path):
    """
    Build a minimal Kaldi-style data dir:
      - rec1.wav : 6 seconds of a 440Hz tone at 16kHz mono
      - rec2.wav : 4 seconds of a 220Hz tone at 16kHz mono (via a `sox`-less
        pipe command that just cats the file, to exercise the pipe-parsing
        path without requiring sox to be installed)
      - wav.scp  : recording-id -> path (rec1 direct path, rec2 via pipe cmd)
      - segments : 3 utterances cut from rec1 + rec2
      - text     : transcripts for each utterance, including a Hinglish
                    code-switched example
    """
    kaldi_dir = root / "data" / "train"
    kaldi_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = root / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    sr = 16000

    # rec1: 6s tone
    t1 = np.linspace(0, 6, int(6 * sr), endpoint=False)
    rec1 = (0.1 * np.sin(2 * np.pi * 440 * t1)).astype("float32")
    rec1_path = wav_dir / "rec1.wav"
    sf.write(str(rec1_path), rec1, sr)

    # rec2: 4s tone (will be referenced via a pipe command using `cat`)
    t2 = np.linspace(0, 4, int(4 * sr), endpoint=False)
    rec2 = (0.1 * np.sin(2 * np.pi * 220 * t2)).astype("float32")
    rec2_path = wav_dir / "rec2.wav"
    sf.write(str(rec2_path), rec2, sr)

    # wav.scp: rec1 direct path, rec2 via a pipe command.
    # Use `cat` (always available) rather than `sox`, piping the wav bytes
    # through unchanged — exercises resolve_recording()'s pipe-cmd branch
    # without requiring sox.
    with open(kaldi_dir / "wav.scp", "w") as f:
        f.write(f"rec1 {rec1_path.resolve()}\n")
        f.write(f"rec2 cat {rec2_path.resolve()} |\n")

    # segments: utt-id rec-id start end
    with open(kaldi_dir / "segments", "w") as f:
        f.write("utt-001 rec1 0.0 2.0\n")     # 2s clip from rec1
        f.write("utt-002 rec1 2.0 5.5\n")     # 3.5s clip from rec1
        f.write("utt-003 rec2 0.5 3.0\n")     # 2.5s clip from rec2
        f.write("utt-004 rec1 5.5 5.4\n")     # invalid (end < start) -> should be skipped

    # text: utt-id transcript (Hinglish code-switching example for utt-002)
    with open(kaldi_dir / "text", "w") as f:
        f.write("utt-001 yeh ek simple sentence hai\n")
        f.write("utt-002 aaj hum ek नया function implement karenge using python\n")
        f.write("utt-003 thank you for watching this tutorial\n")
        # utt-004 deliberately has no text entry -> tests missing-text skip path
        f.write("utt-005 this utterance id has no segment\n")  # orphan text entry

    return kaldi_dir


def test_find_kaldi_dir(tmp_root: Path):
    print("\n[offline] test_find_kaldi_dir ...")
    extract_root = tmp_root / "extracted"
    extract_root.mkdir()
    # Nest the kaldi dir a couple levels deep, like a real tarball might
    nested = extract_root / "Hindi-English_train" / "kws" / "s5"
    kaldi_dir = make_synthetic_kaldi_dir(nested)

    found = slr104.find_kaldi_dir(extract_root)
    assert found is not None, "find_kaldi_dir() returned None for a valid synthetic layout"
    assert found == kaldi_dir, f"find_kaldi_dir() found {found}, expected {kaldi_dir}"
    print(f"  OK: found kaldi dir at {found}")
    return kaldi_dir


def test_parse_wav_scp(kaldi_dir: Path):
    print("\n[offline] test_parse_wav_scp ...")
    wav_map = slr104.parse_wav_scp(kaldi_dir / "wav.scp")
    assert "rec1" in wav_map and wav_map["rec1"][0] == "path", f"rec1 entry wrong: {wav_map.get('rec1')}"
    assert "rec2" in wav_map and wav_map["rec2"][0] == "cmd", f"rec2 entry wrong: {wav_map.get('rec2')}"
    print(f"  OK: rec1={wav_map['rec1']}")
    print(f"  OK: rec2={wav_map['rec2']}")
    return wav_map


def test_parse_segments_and_text(kaldi_dir: Path):
    print("\n[offline] test_parse_segments / test_parse_text ...")
    segments = slr104.parse_segments(kaldi_dir / "segments")
    texts = slr104.parse_text(kaldi_dir / "text")

    assert len(segments) == 4, f"expected 4 segments, got {len(segments)}"
    seg_ids = {s[0] for s in segments}
    assert seg_ids == {"utt-001", "utt-002", "utt-003", "utt-004"}, seg_ids

    assert texts["utt-002"] == "aaj hum ek नया function implement karenge using python"
    assert "utt-004" not in texts, "utt-004 should have no text entry"
    assert "utt-005" in texts, "utt-005 orphan text entry should still parse"
    print(f"  OK: {len(segments)} segments, {len(texts)} text entries")
    return segments, texts


def test_resolve_recording(kaldi_dir: Path, wav_map, tmp_root: Path):
    print("\n[offline] test_resolve_recording (direct path + pipe cmd) ...")
    cache_dir = tmp_root / "_recording_cache"

    # Direct path
    p1 = slr104.resolve_recording(wav_map["rec1"], kaldi_dir, cache_dir, "rec1")
    assert p1 is not None and p1.exists(), f"rec1 resolution failed: {p1}"
    info1 = sf.info(str(p1))
    assert abs(info1.frames / info1.samplerate - 6.0) < 0.01, "rec1 should be ~6s"
    print(f"  OK: rec1 direct path resolved -> {p1} ({info1.frames/info1.samplerate:.2f}s)")

    # Pipe command (cat)
    p2 = slr104.resolve_recording(wav_map["rec2"], kaldi_dir, cache_dir, "rec2")
    assert p2 is not None and p2.exists(), f"rec2 (pipe cmd) resolution failed: {p2}"
    info2 = sf.info(str(p2))
    assert abs(info2.frames / info2.samplerate - 4.0) < 0.01, "rec2 should be ~4s"
    print(f"  OK: rec2 pipe-cmd resolved -> {p2} ({info2.frames/info2.samplerate:.2f}s)")

    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def test_process_pair_end_to_end(kaldi_dir: Path, tmp_root: Path):
    print("\n[offline] test_process_pair (full slice + manifest write) ...")
    out_dir = tmp_root / "out" / "hindi-english"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_dur = slr104.process_pair("hindi-english", "hi-en", kaldi_dir, out_dir, "train")

    manifest_path = out_dir / "transcripts.jsonl"
    assert manifest_path.exists(), "transcripts.jsonl was not written"

    records = [json.loads(l) for l in open(manifest_path)]
    # utt-001, utt-002, utt-003 should be written; utt-004 (invalid duration,
    # also missing text) should be skipped.
    assert len(records) == 3, f"expected 3 records, got {len(records)}: {records}"

    utt_ids = {Path(r["audio"]).stem.split("_")[-1] for r in records}
    assert utt_ids == {"utt-001", "utt-002", "utt-003"}, utt_ids

    for r in records:
        assert Path(r["audio"]).exists(), f"audio file missing: {r['audio']}"
        assert r["lang"] == "hi-en"
        assert r["source"] == "slr104_hindi-english_train"
        assert len(r["text"]) > 0

    # Check durations roughly match segment lengths (2.0, 3.5, 2.5)
    durations = []
    for r in records:
        info = sf.info(r["audio"])
        durations.append(info.frames / info.samplerate)
    durations.sort()
    expected = sorted([2.0, 3.5, 2.5])
    for got, exp in zip(durations, expected):
        assert abs(got - exp) < 0.05, f"duration mismatch: got {got}, expected {exp}"

    print(f"  OK: {len(records)} records written, total_dur={total_dur:.2f}s "
          f"(expected ~{sum(expected):.2f}s)")
    assert abs(total_dur - sum(expected)) < 0.1

    # Hinglish code-switching content check
    hinglish_rec = [r for r in records if "नया" in r["text"] or "function" in r["text"]]
    assert hinglish_rec, "expected the Hinglish code-switched utterance to be present"
    print(f"  OK: Hinglish code-switched text preserved: {hinglish_rec[0]['text']!r}")


def run_offline_tests():
    print("=" * 60)
    print("OFFLINE TESTS (synthetic Kaldi data, no network)")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        kaldi_dir = test_find_kaldi_dir(tmp_root)
        wav_map = test_parse_wav_scp(kaldi_dir)
        test_parse_segments_and_text(kaldi_dir)
        test_resolve_recording(kaldi_dir, wav_map, tmp_root)
        test_process_pair_end_to_end(kaldi_dir, tmp_root)
    print("\nAll offline tests PASSED.")


# ---------------------------------------------------------------------------
# LIVE TEST: real download against OpenSLR-104 mirror (small test split)
# ---------------------------------------------------------------------------

def run_live_test(pair: str):
    print("\n" + "=" * 60)
    print(f"LIVE TEST: real download_slr104.py run for '{pair}' (--include-test)")
    print("=" * 60)
    print("This downloads the smaller TEST split tarball (hundreds of MB) to")
    print("verify find_kaldi_dir() and resolve_recording() against the REAL")
    print("archive structure. Train tarballs (multi-GB) are NOT downloaded here.")

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "slr104_live"

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "download_slr104.py"),
            "--out-dir", str(out_dir),
            "--pairs", pair,
            "--include-test",
        ]
        print(f"\n$ {' '.join(cmd)}\n")

        # Monkeypatch: we only want the test split, not train. download_slr104.py
        # always does both when --include-test is passed (splits = ["train",
        # "test"]). To avoid downloading the multi-GB train tarball in this
        # smoke test, run a small Python driver that calls the test-split path
        # directly instead of shelling out to main().
        cfg = slr104.ARCHIVES[pair]
        pair_out_dir = out_dir / pair
        pair_out_dir.mkdir(parents=True, exist_ok=True)
        archives_dir = out_dir / "_archives"
        archives_dir.mkdir(parents=True, exist_ok=True)

        fname, size_hint = cfg["test"]
        url = f"{slr104.BASE_URL}/{fname}"
        archive_path = archives_dir / fname

        print(f"Downloading {fname} ({size_hint}) ...")
        slr104.download_file(url, archive_path)

        extract_dir = out_dir / "_extracted" / pair / "test"
        extract_dir.mkdir(parents=True, exist_ok=True)
        slr104.extract_archive(archive_path, extract_dir)
        archive_path.unlink()

        print("\nSearching for Kaldi data dir in extracted archive ...")
        kaldi_dir = slr104.find_kaldi_dir(extract_dir)
        if kaldi_dir is None:
            print("\nFAIL: find_kaldi_dir() returned None.")
            print(f"Extracted contents under {extract_dir}:")
            for p in sorted(extract_dir.rglob("*"))[:50]:
                print(f"  {p.relative_to(extract_dir)}")
            print("\nfind_kaldi_dir() needs adjusting for the real archive layout —")
            print("see docs/RUNBOOK.md 'Before your first real run'.")
            sys.exit(1)

        print(f"OK: found Kaldi dir at {kaldi_dir.relative_to(extract_dir)}")

        wav_map = slr104.parse_wav_scp(kaldi_dir / "wav.scp")
        segments = slr104.parse_segments(kaldi_dir / "segments")
        texts = slr104.parse_text(kaldi_dir / "text")
        print(f"OK: wav.scp has {len(wav_map)} recordings, "
              f"segments has {len(segments)} utterances, "
              f"text has {len(texts)} transcripts")

        # Show a sample of wav.scp entry types (direct path vs pipe cmd)
        kinds = {}
        for v in wav_map.values():
            kinds[v[0]] = kinds.get(v[0], 0) + 1
        print(f"OK: wav.scp entry kinds: {kinds}")

        # Process just the first 5 segments to confirm resolve_recording + slicing
        # works against the REAL extracted directory. Important: don't copy
        # wav.scp/text/segments to an isolated directory — OpenSLR-104's wav.scp
        # paths are relative to the split root (kaldi_dir.parent), so process_pair
        # must run against the real extracted location. Instead, monkeypatch
        # parse_segments to limit the segment count.
        print("\nProcessing first 5 segments as a smoke test ...")
        orig_parse_segments = slr104.parse_segments
        slr104.parse_segments = lambda path, _orig=orig_parse_segments: _orig(path)[:5]

        smoke_out = Path(tmp) / "smoke_out"
        smoke_out.mkdir()
        try:
            total_dur = slr104.process_pair(pair, cfg["lang"], kaldi_dir, smoke_out, "test")
        finally:
            slr104.parse_segments = orig_parse_segments

        manifest = smoke_out / "transcripts.jsonl"
        records = [json.loads(l) for l in open(manifest)] if manifest.exists() else []
        print(f"OK: processed {len(records)}/5 segments, "
              f"total {total_dur:.2f}s of audio")
        for r in records[:2]:
            print(f"  sample: lang={r['lang']} text={r['text'][:60]!r}")

        if len(records) == 0:
            print("\nWARNING: 0 segments processed successfully — resolve_recording()")
            print("may not be handling this archive's wav.scp format. Inspect")
            print(f"{trimmed_dir / 'wav.scp'} manually.")
            sys.exit(1)

        print("\nLive smoke test PASSED — find_kaldi_dir() and resolve_recording()")
        print("work against the real OpenSLR-104 archive structure.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="also run a live smoke test against the real OpenSLR-104 "
                         "mirror (downloads a test-split tarball, hundreds of MB)")
    ap.add_argument("--pair", default="bengali-english",
                    choices=["hindi-english", "bengali-english"],
                    help="which pair's test split to use for the live smoke test "
                         "(bengali-english test is smaller: ~606MB vs 443MB for "
                         "hindi-english — pick whichever, both are 'small' relative "
                         "to train tarballs)")
    args = ap.parse_args()

    run_offline_tests()

    if args.live:
        run_live_test(args.pair)
    else:
        print("\n(Skipping live download test. Run with --live to verify against")
        print("the real OpenSLR-104 archive structure — downloads a test-split")
        print("tarball, several hundred MB.)")


if __name__ == "__main__":
    main()
