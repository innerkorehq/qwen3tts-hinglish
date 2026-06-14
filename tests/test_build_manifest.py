#!/usr/bin/env python3
"""
Test build_manifest.py's find_hiacc_pairs() against a synthetic directory tree
matching HiACC's real layout (verified by downloading and inspecting the
Zenodo Corpus.zip — see docs/RUNBOOK.md):

    Corpus/adult/transcription/combined_output_changed_{train,val,test}_output.txt
    Corpus/adult/audio/{train,val,test}_split/<PID+num>.wav
    Corpus/children/transcript/{train,val,test}_output.txt
    Corpus/children/audio/{train,val,test}_split/<PID+num>.wav

Offline, no network/GPU needed.

Usage:
    python3 tests/test_build_manifest.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import build_manifest  # noqa: E402


def make_synthetic_hiacc(root: Path):
    """Build a minimal HiACC-like Corpus/ tree with one adult and one children pair."""
    adult_audio = root / "Corpus" / "adult" / "audio" / "train_split"
    adult_transcript_dir = root / "Corpus" / "adult" / "transcription"
    children_audio = root / "Corpus" / "children" / "audio" / "train_split"
    children_transcript_dir = root / "Corpus" / "children" / "transcript"

    for d in (adult_audio, adult_transcript_dir, children_audio, children_transcript_dir):
        d.mkdir(parents=True, exist_ok=True)

    sr = 16000
    tone = (0.1 * np.sin(2 * np.pi * 440 * np.arange(sr * 2) / sr)).astype(np.float32)

    sf.write(str(adult_audio / "AD09001.wav"), tone, sr)
    sf.write(str(children_audio / "CH03001.wav"), tone, sr)

    # Adult transcript: includes a code-switched line, plus an entry whose
    # audio file doesn't exist (should be skipped) and a blank line.
    (adult_transcript_dir / "combined_output_changed_train_output.txt").write_text(
        "AD09001.wav, In the heart of a bustling city दो sibling रहते थे\n"
        "AD09999.wav, this file does not exist\n"
        "\n",
        encoding="utf-8",
    )

    # Children transcript: Hindi-English code-switched.
    (children_transcript_dir / "train_output.txt").write_text(
        "CH03001.wav, आपका favourite festival कौन सा है\n",
        encoding="utf-8",
    )


def main():
    tmpdir = Path(tempfile.mkdtemp(prefix="hiacc_test_"))
    try:
        make_synthetic_hiacc(tmpdir)

        pairs = build_manifest.find_hiacc_pairs(tmpdir)

        assert len(pairs) == 2, f"expected 2 pairs, got {len(pairs)}: {pairs}"

        by_speaker = {p["speaker_id"]: p for p in pairs}

        adult = by_speaker["AD09"]
        assert adult["text"] == "In the heart of a bustling city दो sibling रहते थे"
        assert adult["source"] == "hiacc_adult_train"
        assert adult["lang"] == "hinglish"
        assert Path(adult["audio"]).name == "AD09001.wav"
        assert Path(adult["audio"]).exists()

        children = by_speaker["CH03"]
        assert children["text"] == "आपका favourite festival कौन सा है"
        assert children["source"] == "hiacc_children_train"
        assert Path(children["audio"]).name == "CH03001.wav"
        assert Path(children["audio"]).exists()

        # The AD09999 line (missing audio file) must be skipped, not crash.
        assert all(Path(p["audio"]).name != "AD09999.wav" for p in pairs)

        print(f"OK: find_hiacc_pairs() returned {len(pairs)} pairs as expected")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
