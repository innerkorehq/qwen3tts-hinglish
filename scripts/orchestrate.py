#!/usr/bin/env python3
"""
End-to-end orchestrator for the Vast.ai A100 PCIe full pipeline, using a
250GB container disk for all working data.

Flow (everything in one rental, see docs/RUNBOOK.md):
  1. Search vast.ai for A100 PCIe offers with >= --disk GB (cheapest reliable match)
  2. Create an instance with --disk 250 (a single container disk -- no separate
     volume provisioning, no machine-pinning)
  3. Onstart script runs the whole pipeline on /root/work:
     bootstrap -> download HiACC + OpenSLR-104 (Hindi-English, Bengali-English)
     -> build manifest (resample, split) -> backup to R2 -> encode_codes.py
     -> backup codes to R2 -> download base model -> train.py -> convert
     formats -> upload all checkpoint variants + logs to R2 -> write DONE
     marker
  4. Poll instance status + DONE marker
  5. On completion (or failure/timeout), destroy the instance -- container disk
     is destroyed with it, no separate cleanup step

COMPLETE HANDOFF: step 3's onstart script is fully self-contained and
self-cleaning -- an EXIT/TERM/INT trap uploads pipeline.log + a status.txt to
R2 and self-destructs the instance (`vastai destroy instance $CONTAINER_ID
--api-key $CONTAINER_API_KEY -y`) on ANY exit path: SUCCESS, any FAILED_*, or
an unexpected crash. This means the run completes (or fails and cleans up,
stopping billing) entirely on its own even if this orchestrator process or the
local machine running it is offline for the whole run. Steps 4/5 here are then
just an optional live-progress view + an idempotent backstop destroy for cases
the instance itself can't handle (--timeout-hours, a dead/offline instance).
See docs/RUNBOOK.md "Self-destruct and handoff".

Ctrl-C only stops *local polling* -- it does NOT destroy the instance (the
remote pipeline and its self-destruct trap keep running independently). Use
`--attach INSTANCE_ID` to re-attach polling to an already-running instance
(e.g. after Ctrl-C, or from a different machine).

Only final and important intermediary artifacts (raw manifests, resampled audio,
encoded codes, checkpoint variants, logs) go to R2 -- /root/work is scratch space
for the run and disappears with the instance.

Container disk sizing (see docs/RUNBOOK.md "Disk budget" for the full breakdown):
steady-state usage is ~144GB, peak transient (raw + resampled audio coexisting
during build_manifest) is ~160GB, against 250GB -- ~90GB headroom at peak.

Requires:
  - `vastai` CLI installed and authenticated (`vastai set api-key <key>`)
  - R2 env vars: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET
  - REPO_GIT_URL pointing at a git remote with this project (cloned on the instance)
  - (optional) VAST_SSH_PUBKEY_PATH pointing at a local SSH public key (.pub) --
    if set, the key is attached to the instance after creation so you can SSH
    in manually (e.g. `vastai ssh-url <id>`) to check progress or debug.
  - (optional) HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) -- passed through to the
    instance as HF_TOKEN for huggingface_hub. Not required for the current
    public base model repo, but raises HF rate limits (helps avoid CDN stalls)
    and covers the model becoming gated later.

Usage:
  export VAST_API_KEY=...
  export R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_BUCKET=...
  export REPO_GIT_URL=https://github.com/yourname/qwen3tts-finetune.git
  export VAST_SSH_PUBKEY_PATH=~/.ssh/id_ed25519.pub   # optional, for manual SSH access
  python3 orchestrate.py --max-price 1.20 --disk 250 --timeout-hours 30
  python3 orchestrate.py --dry-run     # search + show chosen offer, don't create
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_GIT_URL = os.environ.get(
    "REPO_GIT_URL", ""
)  # set this to your git remote with these scripts
R2_PREFIX = "finetune/qwen3tts-hinglish"


def run(cmd, check=True, capture=True, quiet=False):
    if not quiet:
        print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=capture, text=True)
    if capture and not quiet:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


def search_offers(
    max_price, min_disk, gpu=None, min_reliability=0.95, min_vram_gb=None
):
    """Search vast.ai for offers matching any of `gpu` (e.g. A100_PCIE and/or
    A100_SXM4) with enough disk/VRAM/reliability, sorted by price. If `gpu` is
    None/empty, no gpu_name filter is applied (any GPU)."""
    clauses = [
        "num_gpus=1",
        f"reliability>{min_reliability}",
        f"dph_total<={max_price}",
        f"disk_space>={min_disk}",
        "rentable=true",
    ]
    if gpu:
        if len(gpu) == 1:
            clauses.insert(0, f"gpu_name={gpu[0]}")
        else:
            clauses.insert(0, f"gpu_name in [{','.join(gpu)}]")
    if min_vram_gb is not None:
        clauses.append(f"gpu_ram>={min_vram_gb}")
    query = " ".join(clauses)
    result = run(["vastai", "search", "offers", query, "--raw"], check=True)
    try:
        offers = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(
            "ERROR: could not parse vastai search offers output as JSON",
            file=sys.stderr,
        )
        print(result.stdout, file=sys.stderr)
        sys.exit(1)

    if not offers:
        print(f"No offers found matching: {query}", file=sys.stderr)
        sys.exit(1)

    offers.sort(key=lambda o: o.get("dph_total", 999))
    return offers


def build_onstart_script(slr104_pairs, slr104_include_test, eval_frac, resume=False):
    """
    Generates the bash script that runs entirely on the remote instance via
    --onstart-cmd. Full pipeline, all on the instance's container disk under
    /root/work: clone repo, bootstrap, download HiACC + OpenSLR-104
    (Hindi-English / Bengali-English code-switched), build manifest
    (resample + split), encode codes, train, convert formats, upload results,
    write DONE marker. Errors are captured so the orchestrator can detect failure
    via the marker file content.

    If resume=True, each stage first checks R2 for that stage's output (via
    `upload_to_r2.py --exists`) and downloads it instead of recomputing if
    present -- see docs/RUNBOOK.md "Resuming a failed run". Training resume
    (accel_state/training_state.json) is handled the same way and passed to
    train.py via --resume; train.py itself uploads accel_state/training_state
    (and the best checkpoint) to R2 after each epoch.

    The script is a complete handoff to the instance: an EXIT/TERM/INT trap
    (1) writes /root/DONE if some unexpected error path skipped it, (2) uploads
    /root/pipeline.log (plus a per-failure timestamped copy on non-SUCCESS) and
    a small status.txt to R2 so the run is debuggable even if the local
    orchestrator is offline, and (3) self-destructs the instance via
    `vastai destroy instance $CONTAINER_ID --api-key $CONTAINER_API_KEY -y`.
    This means the instance cleans up and stops billing on its own -- success,
    any FAILED_*, or an unexpected crash -- even if the M4 orchestrator is
    asleep, disconnected, or shut down. See docs/RUNBOOK.md "Self-destruct and
    handoff".
    """
    if not REPO_GIT_URL:
        print(
            "WARNING: REPO_GIT_URL is not set. The onstart script will expect",
            file=sys.stderr,
        )
        print(
            "the qwen3tts-finetune project to already be baked into the image, or",
            file=sys.stderr,
        )
        print(
            "you must edit build_onstart_script() to fetch it another way (e.g. R2).",
            file=sys.stderr,
        )

    if REPO_GIT_URL:
        clone_cmd = (
            f"for i in 1 2 3 4 5; do\n"
            f"  rm -rf repo\n"
            f"  if git clone --depth 1 {REPO_GIT_URL} repo; then break; fi\n"
            f'  echo "git clone attempt $i/5 failed, retrying in $((i*10))s ..."\n'
            f"  sleep $((i*10))\n"
            f"done"
        )
    else:
        clone_cmd = "echo 'skip clone, assuming repo present'"

    SLR104_PAIRS = " ".join(slr104_pairs)
    SLR104_TEST_ARG = "--include-test" if slr104_include_test else ""
    EVAL_FRAC = eval_frac
    RESUME = "1" if resume else "0"

    script = f"""#!/bin/bash
set -uo pipefail
exec > /root/pipeline.log 2>&1

# Unbuffered stdout/stderr -- without this, Python's block-buffering on a
# redirected (non-tty) stream means pipeline.log can appear "stuck" on a
# stale tqdm progress line for a long time even while a download/training
# step is actively progressing.
export PYTHONUNBUFFERED=1

echo "=== START $(date) ==="

cd /root
{clone_cmd}
REPO=/root/repo

# --- env for R2 ---
export R2_ACCOUNT_ID="{os.environ.get("R2_ACCOUNT_ID", "")}"
export R2_ACCESS_KEY_ID="{os.environ.get("R2_ACCESS_KEY_ID", "")}"
export R2_SECRET_ACCESS_KEY="{os.environ.get("R2_SECRET_ACCESS_KEY", "")}"
export R2_BUCKET="{os.environ.get("R2_BUCKET", "")}"

# --- HF_TOKEN (optional): huggingface_hub picks this up automatically for
# snapshot_download/from_pretrained. Not required for the current public
# Qwen3-TTS-12Hz-1.7B-Base repo, but (a) anonymous HF downloads are subject to
# tighter rate limits / throttling -- a likely contributor to CDN stalls --
# and (b) covers the model becoming gated later. Set HF_TOKEN locally before
# running orchestrate.py to pass it through; harmless if empty. ---
export HF_TOKEN="{os.environ.get("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN", ""))}"

# --- Complete handoff: install vastai CLI immediately so the cleanup trap
# below can self-destruct the instance no matter where the pipeline fails.
# Retried -- if this never installs, self-destruct can't run later, so it's
# worth a few attempts against a flaky PyPI mirror. ---
for i in 1 2 3 4 5; do
  pip install -q vastai 2>/dev/null && break
  echo "pip install vastai attempt $i/5 failed, retrying in $((i*5))s ..."
  sleep $((i*5))
done

mark_done() {{
  echo "$1" > /root/DONE
  echo "=== END $(date) status=$1 ==="
}}

# --- Cleanup trap: runs on ANY exit (success, FAILED_*, or an unexpected
# crash/signal). Uploads logs + a status file to R2 (so the run is debuggable
# even with the instance gone) and self-destructs the instance via the
# instance-scoped CONTAINER_ID/CONTAINER_API_KEY -- this is what makes the
# instance independent of the M4 orchestrator: it stops billing on its own
# even if the orchestrator is offline. See docs/RUNBOOK.md "Self-destruct and
# handoff".
CLEANED_UP=0
cleanup() {{
  if [ "$CLEANED_UP" = "1" ]; then return; fi
  CLEANED_UP=1

  if [ ! -f /root/DONE ]; then
    echo "UNKNOWN_EXIT" > /root/DONE
    echo "=== END $(date) status=UNKNOWN_EXIT (unexpected crash/signal) ==="
  fi
  STATUS=$(cat /root/DONE)

  python3 "$REPO/scripts/upload_to_r2.py" --file /root/pipeline.log --bucket "$R2_BUCKET" \\
    --key {R2_PREFIX}/logs/pipeline.log 2>/dev/null || true
  python3 "$REPO/scripts/upload_to_r2.py" --file /root/DONE --bucket "$R2_BUCKET" \\
    --key {R2_PREFIX}/logs/status.txt 2>/dev/null || true
  if [ "$STATUS" != "SUCCESS" ]; then
    python3 "$REPO/scripts/upload_to_r2.py" --file /root/pipeline.log --bucket "$R2_BUCKET" \\
      --key "{R2_PREFIX}/logs/pipeline_failed_${{STATUS}}_$(date +%s).log" 2>/dev/null || true
  fi

  echo "Self-destructing instance $CONTAINER_ID (status=$STATUS) ..."
  vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" -y || true
}}
trap cleanup EXIT
trap 'cleanup; exit 143' TERM
trap 'cleanup; exit 130' INT

# --- All working data lives under /root/work on the instance's container disk ---
# (final and important intermediary artifacts are also pushed to R2 below;
# /root/work disappears with the instance at the end of the run)
mkdir -p /root/work
cd /root/work
mkdir -p data models checkpoints

if [ ! -d "$REPO" ]; then
  mark_done FAILED_BOOTSTRAP; exit 1
fi

# --- 1. bootstrap deps (GPU/disk checks, Python deps, sox, flash-attn) ---
bash "$REPO/scripts/bootstrap_vastai.sh" || {{ mark_done FAILED_BOOTSTRAP; exit 1; }}

# --- 2-4. get train_with_codes.jsonl/eval_with_codes.jsonl -- or, with
#          --resume, reuse a prior run's already-encoded codes from R2
#          directly, skipping manifest/resampled-audio entirely (the encoded
#          codes are the only thing stages 5+ need; raw/resampled audio is
#          only an intermediate for stage 4). See RUNBOOK.md "Resuming a
#          failed run". ---
RESUME_CODES=0
if [ "{RESUME}" = "1" ]; then
  if python3 "$REPO/scripts/upload_to_r2.py" --exists --bucket "$R2_BUCKET" --key {R2_PREFIX}/train_with_codes.jsonl \\
     && python3 "$REPO/scripts/upload_to_r2.py" --exists --bucket "$R2_BUCKET" --key {R2_PREFIX}/eval_with_codes.jsonl; then
    RESUME_CODES=1
  fi
fi

if [ "$RESUME_CODES" = "1" ]; then
  echo "--resume: found encoded codes in R2, downloading instead of rebuilding manifest/resampled audio/codes"
  python3 "$REPO/scripts/upload_to_r2.py" --download --bucket "$R2_BUCKET" \\
    --key {R2_PREFIX}/train_with_codes.jsonl --file ./data/train_with_codes.jsonl \\
    || {{ mark_done FAILED_ENCODE_TRAIN; exit 1; }}
  python3 "$REPO/scripts/upload_to_r2.py" --download --bucket "$R2_BUCKET" \\
    --key {R2_PREFIX}/eval_with_codes.jsonl --file ./data/eval_with_codes.jsonl \\
    || {{ mark_done FAILED_ENCODE_EVAL; exit 1; }}

  # train_with_codes.jsonl has absolute ref_audio paths pointing to
  # ./data/resampled/ -- TTSDataset loads these at training time for
  # per-sample speaker embeddings. Must download even when codes are cached.
  echo "--resume: downloading resampled audio (needed for speaker embeddings at train time) ..."
  python3 "$REPO/scripts/shard_tar.py" download --dest-dir ./data \\
    --work-dir ./data/_shards --bucket "$R2_BUCKET" \\
    --key-prefix {R2_PREFIX}/resampled_shards \\
    || {{ mark_done FAILED_TRAIN; exit 1; }}
else
  # --- 2/3. download datasets + build unified manifest -- or, with --resume,
  #          reuse a prior run's manifest + resampled_shards/ (or older
  #          resampled.tar / resampled/* schemes) from R2. See RUNBOOK.md
  #          "Resuming a failed run" for the sharding rationale. ---
  RESUME_SHARDS=0
  RESUME_TAR=0
  RESUME_FILES=0
  if [ "{RESUME}" = "1" ]; then
    if python3 "$REPO/scripts/upload_to_r2.py" --exists --bucket "$R2_BUCKET" --key {R2_PREFIX}/resampled_shards/manifest.json \\
       && python3 "$REPO/scripts/upload_to_r2.py" --download --bucket "$R2_BUCKET" \\
         --key {R2_PREFIX}/manifest_raw.jsonl --file ./data/manifest_raw.jsonl \\
       && python3 "$REPO/scripts/upload_to_r2.py" --download --bucket "$R2_BUCKET" \\
         --key {R2_PREFIX}/manifest_eval_raw.jsonl --file ./data/manifest_eval_raw.jsonl; then
      RESUME_SHARDS=1
    elif python3 "$REPO/scripts/upload_to_r2.py" --exists --bucket "$R2_BUCKET" --key {R2_PREFIX}/resampled.tar \\
       && python3 "$REPO/scripts/upload_to_r2.py" --download --bucket "$R2_BUCKET" \\
         --key {R2_PREFIX}/manifest_raw.jsonl --file ./data/manifest_raw.jsonl \\
       && python3 "$REPO/scripts/upload_to_r2.py" --download --bucket "$R2_BUCKET" \\
         --key {R2_PREFIX}/manifest_eval_raw.jsonl --file ./data/manifest_eval_raw.jsonl; then
      RESUME_TAR=1
    elif python3 "$REPO/scripts/upload_to_r2.py" --download --bucket "$R2_BUCKET" \\
         --key {R2_PREFIX}/manifest_raw.jsonl --file ./data/manifest_raw.jsonl \\
       && python3 "$REPO/scripts/upload_to_r2.py" --download --bucket "$R2_BUCKET" \\
         --key {R2_PREFIX}/manifest_eval_raw.jsonl --file ./data/manifest_eval_raw.jsonl; then
      RESUME_FILES=1
    fi
  fi

  if [ "$RESUME_SHARDS" = "1" ]; then
    echo "--resume: found resampled_shards/manifest.json in R2, downloading and extracting shards instead of rebuilding"
    python3 "$REPO/scripts/shard_tar.py" download --dest-dir ./data \\
      --work-dir ./data/_shards --bucket "$R2_BUCKET" \\
      --key-prefix {R2_PREFIX}/resampled_shards \\
      || {{ mark_done FAILED_BUILD_MANIFEST; exit 1; }}
  elif [ "$RESUME_TAR" = "1" ]; then
    echo "--resume: no resampled_shards/manifest.json in R2 -- found resampled.tar from an older run, downloading and extracting"
    python3 "$REPO/scripts/upload_to_r2.py" --download --bucket "$R2_BUCKET" \\
      --key {R2_PREFIX}/resampled.tar --file ./data/resampled.tar \\
      || {{ mark_done FAILED_BUILD_MANIFEST; exit 1; }}
    mkdir -p ./data/resampled
    tar -xf ./data/resampled.tar -C ./data \\
      || {{ mark_done FAILED_BUILD_MANIFEST; exit 1; }}
    rm -f ./data/resampled.tar
  else
    if [ "$RESUME_FILES" = "1" ]; then
      echo "--resume: no resampled_shards/manifest.json or resampled.tar in R2 -- downloading individual resampled/ files"
      echo "  from an older run (if any) for build_manifest.py to reuse"
      python3 "$REPO/scripts/upload_to_r2.py" --download --recursive --bucket "$R2_BUCKET" \\
        --key {R2_PREFIX}/resampled --file ./data/resampled || true
    fi

    python3 "$REPO/scripts/download_hiacc.py" --out-dir ./data/raw/hiacc \\
      || {{ mark_done FAILED_DOWNLOAD_HIACC; exit 1; }}

    python3 "$REPO/scripts/download_slr104.py" \\
      --out-dir ./data/raw/slr104 \\
      --pairs {SLR104_PAIRS} \\
      {SLR104_TEST_ARG} \\
      || {{ mark_done FAILED_DOWNLOAD_SLR104; exit 1; }}

    python3 "$REPO/scripts/build_manifest.py" \\
      --hiacc-dir ./data/raw/hiacc \\
      --slr104-dir ./data/raw/slr104 \\
      --out ./data/manifest_raw.jsonl \\
      --resampled-dir ./data/resampled \\
      --eval-frac {EVAL_FRAC} \\
      || {{ mark_done FAILED_BUILD_MANIFEST; exit 1; }}

    # manifest_eval_raw.jsonl is written alongside --out (same stem + _eval_raw)
    if [ ! -f ./data/manifest_eval_raw.jsonl ]; then
      mark_done FAILED_BUILD_MANIFEST; exit 1
    fi

    # Backup raw manifests + resampled audio to R2 as ~1GB tar shards.
    python3 "$REPO/scripts/upload_to_r2.py" --file ./data/manifest_raw.jsonl --bucket "$R2_BUCKET" \\
      --key {R2_PREFIX}/manifest_raw.jsonl
    python3 "$REPO/scripts/upload_to_r2.py" --file ./data/manifest_eval_raw.jsonl --bucket "$R2_BUCKET" \\
      --key {R2_PREFIX}/manifest_eval_raw.jsonl

    # Free disk before sharding -- see RUNBOOK.md "Disk budget".
    rm -rf ./data/raw/slr104/*/audio ./data/raw/hiacc

    python3 "$REPO/scripts/shard_tar.py" upload --src-dir ./data/resampled \\
      --work-dir ./data/_shards --bucket "$R2_BUCKET" \\
      --key-prefix {R2_PREFIX}/resampled_shards --arcname resampled --shard-size-mb 1024 \\
      || {{ mark_done FAILED_BUILD_MANIFEST; exit 1; }}

    # Clean up objects from older schemes -- resampled_shards/ is now canonical.
    python3 "$REPO/scripts/upload_to_r2.py" --delete --bucket "$R2_BUCKET" \\
      --key {R2_PREFIX}/resampled.tar || true
    python3 "$REPO/scripts/upload_to_r2.py" --delete --recursive --bucket "$R2_BUCKET" \\
      --key {R2_PREFIX}/resampled || true
  fi

  # --- 4. encode audio -> codes ---
  python3 "$REPO/scripts/encode_codes.py" \\
    --manifest ./data/manifest_raw.jsonl \\
    --out ./data/train_with_codes.jsonl \\
    --device cuda --batch-size 32 \\
    || {{ mark_done FAILED_ENCODE_TRAIN; exit 1; }}

  python3 "$REPO/scripts/encode_codes.py" \\
    --manifest ./data/manifest_eval_raw.jsonl \\
    --out ./data/eval_with_codes.jsonl \\
    --device cuda --batch-size 32 \\
    || {{ mark_done FAILED_ENCODE_EVAL; exit 1; }}

  # Backup encoded codes to R2 (important intermediary artifact -- small files)
  python3 "$REPO/scripts/upload_to_r2.py" --file ./data/train_with_codes.jsonl --bucket "$R2_BUCKET" \\
    --key {R2_PREFIX}/train_with_codes.jsonl
  python3 "$REPO/scripts/upload_to_r2.py" --file ./data/eval_with_codes.jsonl --bucket "$R2_BUCKET" \\
    --key {R2_PREFIX}/eval_with_codes.jsonl
fi

# --- 4.5. download base model (init_model_path for train.py) -- done here,
#          after resampling/encoding and the raw-data cleanup in step 2/3,
#          rather than during bootstrap, so its ~6.8GB isn't sitting on disk
#          during build_manifest's peak-transient window. See docs/RUNBOOK.md
#          "Disk budget". ---
bash "$REPO/scripts/download_base_model.sh" || {{ mark_done FAILED_DOWNLOAD_MODEL; exit 1; }}

# --- 5. train (produces final_fp32, final_bf16, final aliased to fp32) -- or,
#        with --resume, reuse a prior run's accel_state/training_state.json
#        from R2 and continue from there ---
TRAIN_RESUME_FLAG=""
if [ "{RESUME}" = "1" ]; then
  if python3 "$REPO/scripts/upload_to_r2.py" --exists --recursive --bucket "$R2_BUCKET" --key {R2_PREFIX}/accel_state \\
     && python3 "$REPO/scripts/upload_to_r2.py" --exists --bucket "$R2_BUCKET" --key {R2_PREFIX}/training_state.json; then
    echo "--resume: found training checkpoint in R2, downloading"
    python3 "$REPO/scripts/upload_to_r2.py" --download --recursive --bucket "$R2_BUCKET" \\
      --key {R2_PREFIX}/accel_state --file ./checkpoints/accel_state
    python3 "$REPO/scripts/upload_to_r2.py" --download --bucket "$R2_BUCKET" \\
      --key {R2_PREFIX}/training_state.json --file ./checkpoints/training_state.json
    TRAIN_RESUME_FLAG="--resume"
  fi
fi

# train_config.yaml paths (init_model_path, output_model_path, data files) are
# relative to cwd, which is /root/work here -- see configs/train_config.yaml comment.
# train.py itself uploads accel_state/training_state.json/best to R2 after each
# epoch (if R2 env vars are set, which they are above) -- this is what makes
# FAILED_TRAIN recoverable via --resume on a fresh instance.
python3 "$REPO/scripts/train.py" --config "$REPO/configs/train_config.yaml" $TRAIN_RESUME_FLAG \\
  || {{ mark_done FAILED_TRAIN; exit 1; }}

# --- 6. upload the core checkpoints (final artifacts -> R2) ---
# Trained checkpoints exist only on this instance's container disk, and the
# cleanup trap self-destructs the instance regardless of exit status -- so
# upload final_fp32/final_bf16 *before* attempting any post-processing
# (fp16/GGUF conversion below). A conversion failure must never risk losing
# a completed training run. Retry each upload a few times before giving up.
upload_with_retry() {{
  for attempt in 1 2 3; do
    if python3 "$REPO/scripts/upload_to_r2.py" --file "$1" --bucket "$R2_BUCKET" --key "$2" --recursive; then
      return 0
    fi
    echo "upload attempt $attempt for $2 failed, retrying in 30s ..."
    sleep 30
  done
  return 1
}}

upload_with_retry ./checkpoints/final_fp32 {R2_PREFIX}/final_fp32 \\
  || {{ mark_done FAILED_UPLOAD_MODEL; exit 1; }}

upload_with_retry ./checkpoints/final_bf16 {R2_PREFIX}/final_bf16 \\
  || {{ mark_done FAILED_UPLOAD_MODEL; exit 1; }}

# --- 7. convert fp32 master -> fp16 (for MPS/M4 inference). Failure here is
#        NOT fatal -- final_fp32/final_bf16 are already safely uploaded above
#        and are usable on their own. ---
python3 "$REPO/scripts/convert_model.py" \\
  --master ./checkpoints/final_fp32 \\
  --to fp16 --out ./checkpoints/final_fp16 \\
  && echo "fp16 conversion succeeded" \\
  || echo "fp16 conversion failed (final_fp32/final_bf16 still usable). Continuing."

if [ -f ./checkpoints/final_fp16/model.safetensors ]; then
  upload_with_retry ./checkpoints/final_fp16 {R2_PREFIX}/final_fp16 || true
else
  echo "WARNING: skipping final_fp16 upload -- model.safetensors missing (conversion failed or incomplete)"
fi

# --- 8. attempt GGUF conversion (may fail for Qwen3-TTS -- see convert_model.py
#        docstring; failure here does NOT abort the pipeline, since fp32/fp16
#        checkpoints are already usable for cross-device inference) ---
for i in 1 2 3; do
  rm -rf ./llama.cpp
  git clone --depth 1 https://github.com/ggml-org/llama.cpp ./llama.cpp 2>/dev/null && break
  echo "llama.cpp clone attempt $i/3 failed, retrying in $((i*10))s ..."
  sleep $((i*10))
done
if [ -f ./llama.cpp/convert_hf_to_gguf.py ]; then
  python3 "$REPO/scripts/convert_model.py" \\
    --master ./checkpoints/final_fp32 \\
    --to gguf --out ./checkpoints/final_gguf \\
    --llama-cpp-dir ./llama.cpp --gguf-outtype f16 \\
    && echo "GGUF conversion succeeded" \\
    || echo "GGUF conversion failed (expected for Qwen3-TTS -- fp32/fp16 checkpoints still usable). Continuing."
else
  echo "llama.cpp clone failed or convert script missing -- skipping GGUF, continuing."
fi

# Upload GGUF only if it was produced (best-effort, doesn't fail the run)
if [ -d ./checkpoints/final_gguf ]; then
  upload_with_retry ./checkpoints/final_gguf {R2_PREFIX}/final_gguf || true
fi

mark_done SUCCESS
"""
    return script


def create_instance(
    offer_id,
    disk_gb,
    onstart_script_path,
    image="pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
):
    """Create an instance with a single --disk GB container disk (no separate volume)."""
    result = run(
        [
            "vastai",
            "create",
            "instance",
            str(offer_id),
            "--image",
            image,
            "--disk",
            str(disk_gb),
            "--onstart",
            str(onstart_script_path),
            "--ssh",
            "--direct",
            "--raw",
        ]
    )
    data = json.loads(result.stdout)
    if not data.get("success"):
        raise RuntimeError(f"create instance failed: {data}")
    return data["new_contract"]


def get_instance_status(instance_id):
    result = run(["vastai", "show", "instance", str(instance_id), "--raw"], check=False)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def _clear_tmp(path):
    """Remove a leftover tmp path, whether it's a file, an empty dir, or a dir
    `vastai copy`/rsync sometimes leaves behind when the remote source doesn't
    exist yet (plain Path.unlink() raises EPERM/IsADirectoryError on a dir)."""
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def download_r2_status():
    """Best-effort: download the status.txt the instance's cleanup trap uploads
    to R2 just before self-destructing (see build_onstart_script). Returns the
    status string, or None if not present -- e.g. the instance is still
    running, hasn't reached the cleanup trap yet, or vastai/pip install failed
    on the instance before it could upload anything."""
    tmp = Path("/tmp/r2_status_check.txt")
    _clear_tmp(tmp)
    run(
        [
            "python3",
            str(Path(__file__).resolve().parent / "upload_to_r2.py"),
            "--download",
            "--bucket",
            os.environ["R2_BUCKET"],
            "--key",
            f"{R2_PREFIX}/logs/status.txt",
            "--file",
            str(tmp),
        ],
        check=False,
        quiet=True,
    )
    if tmp.is_file():
        content = tmp.read_text().strip()
        _clear_tmp(tmp)
        return content
    _clear_tmp(tmp)
    return None


def check_done_marker(instance_id):
    """SSH-free check via `vastai copy` of the DONE marker file, if it exists.

    Before the pipeline finishes, /root/DONE doesn't exist yet and `vastai copy`
    prints an "Invalid src_full_path" error to stderr for the missing remote
    file -- this is expected/benign on every poll until the marker is written,
    so the command runs quietly and absence is detected by the local tmp file
    never being created.
    """
    tmp = Path("/tmp/DONE_marker_check")
    _clear_tmp(tmp)
    run(
        [
            "vastai",
            "copy",
            f"{instance_id}:/root/DONE",
            f"local:{tmp}",
        ],
        check=False,
        quiet=True,
    )
    if tmp.is_file():
        content = tmp.read_text().strip()
        _clear_tmp(tmp)
        return content
    _clear_tmp(tmp)
    return None


def attach_ssh_key(instance_id, ssh_pubkey_path):
    """Attach a public key to the instance for manual `vastai ssh-url` access."""
    pubkey = Path(ssh_pubkey_path).expanduser().read_text().strip()
    run(["vastai", "attach", "ssh", str(instance_id), pubkey], check=False)


def destroy_instance(instance_id):
    # -y skips the interactive "Are you sure?" confirmation prompt, which
    # would otherwise hang (or abort on a bare Enter) in non-interactive use.
    run(["vastai", "destroy", "instance", str(instance_id), "-y"], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max-price", type=float, default=1.20, help="max $/hr for A100 PCIe"
    )
    ap.add_argument(
        "--disk",
        type=int,
        default=250,
        help="container disk size in GB (default 250 -- see "
        "docs/RUNBOOK.md 'Disk budget' for the breakdown; "
        "steady-state ~144GB, peak transient ~160GB)",
    )
    ap.add_argument(
        "--gpu",
        nargs="+",
        default=None,
        help="vast.ai gpu_name(s) to search, e.g. --gpu A100_PCIE A100_SXM4 "
        "(default: no filter, any GPU; cheapest matching offer across "
        "all given wins)",
    )
    ap.add_argument(
        "--min-vram",
        type=int,
        default=None,
        metavar="GB",
        help="minimum GPU VRAM in GB (e.g. 80 to require the 80GB A100 "
        "variant, excluding 40GB offers)",
    )
    ap.add_argument(
        "--min-reliability",
        type=float,
        default=0.95,
        help="minimum vast.ai reliability score, 0-1 (default 0.95)",
    )
    ap.add_argument(
        "--poll-interval",
        type=int,
        default=300,
        help="seconds between status checks (default 300 for long runs)",
    )
    ap.add_argument(
        "--timeout-hours",
        type=float,
        default=30.0,
        help="safety net, not expected runtime (default 30)",
    )
    ap.add_argument(
        "--slr104-pairs",
        nargs="+",
        default=["hindi-english", "bengali-english"],
        choices=["hindi-english", "bengali-english"],
        help="OpenSLR-104 code-switched pairs to download",
    )
    ap.add_argument(
        "--slr104-include-test",
        action="store_true",
        help="also download the (smaller) OpenSLR-104 test splits",
    )
    ap.add_argument(
        "--eval-frac",
        type=float,
        default=0.02,
        help="fraction of combined dataset held out for eval",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="resume a previously failed run: each pipeline stage checks R2 "
        "for that stage's output first and downloads it instead of "
        "recomputing if present (see docs/RUNBOOK.md 'Resuming a failed run')",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="search and show offer only, don't create",
    )
    ap.add_argument(
        "--attach",
        type=int,
        default=None,
        metavar="INSTANCE_ID",
        help="re-attach polling to an already-running instance instead of "
        "creating a new one -- use this after Ctrl-C'ing a previous run "
        "(Ctrl-C only stops local polling; the remote pipeline and its "
        "self-destruct trap keep running independently)",
    )
    args = ap.parse_args()

    for var in (
        "R2_ACCOUNT_ID",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET",
    ):
        if not os.environ.get(var):
            print(f"ERROR: {var} not set", file=sys.stderr)
            sys.exit(1)

    instance_id = None
    final_status = None
    self_destructed = False

    if args.attach is not None:
        instance_id = args.attach
        print(
            f"Re-attaching to instance {instance_id} -- skipping offer search/creation."
        )
        print("Polling for completion ...")
    else:
        gpu_msg = "/".join(args.gpu) if args.gpu else "any GPU"
        vram_msg = f", >= {args.min_vram}GB VRAM" if args.min_vram else ""
        print(
            f"Searching for {gpu_msg} offers under ${args.max_price}/hr "
            f"with >= {args.disk}GB disk{vram_msg}, "
            f"reliability > {args.min_reliability} ..."
        )
        offers = search_offers(
            args.max_price,
            args.disk,
            gpu=args.gpu,
            min_reliability=args.min_reliability,
            min_vram_gb=args.min_vram,
        )
        best = offers[0]
        print(f"\\nTop {min(5, len(offers))} offers:")
        for o in offers[:5]:
            print(
                f"  id={o['id']} ${o['dph_total']:.3f}/hr "
                f"reliability={o.get('reliability', 0):.3f} "
                f"disk={o.get('disk_space')}GB "
                f"loc={o.get('geolocation', '?')}"
            )

        print(f"\\nSelected offer id={best['id']} at ${best['dph_total']:.3f}/hr")

        if args.dry_run:
            print("\\n--dry-run set, not creating instance.")
            return

        onstart_path = Path("/tmp/onstart.sh")
        onstart_path.write_text(
            build_onstart_script(
                slr104_pairs=args.slr104_pairs,
                slr104_include_test=args.slr104_include_test,
                eval_frac=args.eval_frac,
                resume=args.resume,
            )
        )
        print(f"\\nGenerated onstart script -> {onstart_path}")
        if args.resume:
            print(
                "--resume set: each stage will check R2 for prior outputs before recomputing."
            )

        print("\\nCreating instance ...")

    try:
        if args.attach is None:
            instance_id = create_instance(best["id"], args.disk, onstart_path)
            print(f"Instance created: id={instance_id}")

            ssh_pubkey_path = os.environ.get("VAST_SSH_PUBKEY_PATH")
            if ssh_pubkey_path:
                print(f"Attaching SSH key from {ssh_pubkey_path} ...")
                attach_ssh_key(instance_id, ssh_pubkey_path)
                run(["vastai", "ssh-url", str(instance_id)], check=False)
                print(
                    "(use the URL above to SSH in manually and inspect /root/pipeline.log)"
                )
            else:
                print(
                    "VAST_SSH_PUBKEY_PATH not set -- skipping manual SSH access setup."
                )
            print("Pipeline is now running in the background via onstart script.")
            print("Polling for completion ...")

        start = time.time()
        timeout_sec = args.timeout_hours * 3600

        self_destructed = False

        while True:
            elapsed = time.time() - start
            if elapsed > timeout_sec:
                print(f"\\nTIMEOUT after {args.timeout_hours}h -- destroying instance.")
                final_status = "TIMEOUT"
                break

            info = get_instance_status(instance_id)
            actual_status = info.get("actual_status") if info else None
            print(f"  [{elapsed / 60:.1f}min] instance status: {actual_status}")

            if not info:
                # The instance may already be gone -- its cleanup trap
                # self-destructs it on completion/failure regardless of
                # whether we're polling (see build_onstart_script). Check R2
                # for the status it uploads just before destroying itself.
                r2_status = download_r2_status()
                if r2_status:
                    print(
                        f"\\nInstance is gone; self-reported status from R2: {r2_status}"
                    )
                    final_status = r2_status
                    self_destructed = True
                    break

            if actual_status in ("exited", "offline"):
                print(f"\\nInstance reached terminal bad state: {actual_status}")
                final_status = f"INSTANCE_{actual_status.upper()}"
                break

            marker = check_done_marker(instance_id)
            if marker:
                print(f"\\nDONE marker found: {marker}")
                final_status = marker
                # The instance's own cleanup trap is uploading logs/status and
                # self-destructing right about now -- the destroy_instance()
                # call below is a harmless, idempotent backstop.
                self_destructed = True
                break

            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        # Total handoff: Ctrl-C only stops *local polling*. The remote
        # pipeline (and its own cleanup trap / self-destruct) keeps running
        # independently via the onstart script -- do NOT destroy the
        # instance here, or the whole point of "fire and forget" is lost.
        print("\\n\\nInterrupted (Ctrl-C) -- this only stops local polling.")
        print("The remote pipeline keeps running and will self-destruct on its own")
        print(
            "(success or failure) via its cleanup trap. The instance is NOT being destroyed."
        )
        if instance_id is not None:
            print(f"\\nRe-attach polling later with:")
            print(f"  python3 {sys.argv[0]} --attach {instance_id}")
            print(f"\\nOr check status without polling:")
            print(f"  vastai show instance {instance_id} --raw")
            print(f"  (or check {R2_PREFIX}/logs/status.txt in R2 once it finishes)")
        return

    except RuntimeError as e:
        print(f"\\nERROR: {e}")
        final_status = "ERROR_PROVISIONING"

    print(f"\\nFinal status: {final_status}")

    if final_status == "SUCCESS":
        print(
            f"Results uploaded to R2 under {R2_PREFIX}/ "
            f"(final_fp32/, final_bf16/, final_fp16/, manifest/codes backups, logs/)"
        )
    elif final_status not in (None, "SUCCESS"):
        print("Pipeline did not succeed. Pulling log for debugging ...")
        if self_destructed or instance_id is None:
            run(
                [
                    "python3",
                    str(Path(__file__).resolve().parent / "upload_to_r2.py"),
                    "--download",
                    "--bucket",
                    os.environ["R2_BUCKET"],
                    "--key",
                    f"{R2_PREFIX}/logs/pipeline.log",
                    "--file",
                    "./pipeline_failed.log",
                ],
                check=False,
            )
        else:
            run(
                [
                    "vastai",
                    "copy",
                    f"{instance_id}:/root/pipeline.log",
                    "local:./pipeline_failed.log",
                ],
                check=False,
            )
        print("Saved to ./pipeline_failed.log (if found) -- inspect before retrying.")
        print(
            f"Full status + per-failure log copies are also under "
            f"{R2_PREFIX}/logs/ in R2 (status.txt, pipeline_failed_<status>_<ts>.log)."
        )

    if instance_id is not None:
        # Idempotent backstop: the instance's own cleanup trap already
        # self-destructs on every exit path (success, FAILED_*, timeout-from-
        # the-instance's-perspective, or an unexpected crash/signal) -- see
        # build_onstart_script's "Self-destruct and handoff". This call covers
        # the remaining cases where the orchestrator itself decides to give up
        # (--timeout-hours, INSTANCE_EXITED/OFFLINE, provisioning error). Note:
        # Ctrl-C (KeyboardInterrupt) returns earlier above and never reaches here.
        print(
            f"\\nDestroying instance {instance_id} (if not already self-destructed) ..."
        )
        destroy_instance(instance_id)
        print("Done. Billing stopped, container disk destroyed with the instance.")
    else:
        print("Done. (No instance was created, or it already self-destructed.)")


if __name__ == "__main__":
    main()
