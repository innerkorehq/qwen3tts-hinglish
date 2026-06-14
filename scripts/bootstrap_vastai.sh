#!/usr/bin/env bash
# Bootstrap script for Vast.ai A100 instance.
# Called automatically by orchestrate.py's onstart script (no manual SSH needed).
# Handles environment setup only: GPU/disk checks, Python deps, base model
# download. Dataset download, manifest building, encoding, and training are
# separate steps in orchestrate.py's onstart pipeline — see docs/RUNBOOK.md.
#
# Usage (called from onstart script with cwd=/root/work, on the instance's
# 250GB container disk):
#   bash /root/repo/scripts/bootstrap_vastai.sh

set -euo pipefail

echo "=== 1/4: Checking GPU ==="
nvidia-smi || { echo "ERROR: no GPU visible"; exit 1; }

echo "=== 2/4: Checking container disk space ==="
df -h .
FREE_GB=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
echo "Free space: ${FREE_GB}GB"
if [ "$FREE_GB" -lt 100 ]; then
  echo "WARNING: less than 100GB free. Dataset downloads + resampled audio +"
  echo "1.7B model checkpoints (fp32/bf16/fp16 variants) may not fit."
  echo "Consider increasing --disk on orchestrate.py (default 250, see"
  echo "docs/RUNBOOK.md 'Disk budget')."
fi

echo "=== 3/4: Installing Python deps ==="
# Retry pip installs -- PyPI/index hiccups are common and otherwise abort the
# whole run via `set -e` before training even starts.
pip_retry() {
  for attempt in 1 2 3 4 5; do
    if pip install -q "$@"; then return 0; fi
    if [ "$attempt" = "5" ]; then return 1; fi
    echo "pip install $* : attempt $attempt/5 failed, retrying in $((attempt*10))s ..."
    sleep $((attempt*10))
  done
}
pip_retry -U pip
pip_retry qwen-tts transformers accelerate deepspeed boto3 soundfile librosa \
  datasets huggingface_hub pydub requests pyyaml

echo "=== 4/4: Downloading base model ==="
mkdir -p ./data ./models

# huggingface_hub's streaming download has no read-timeout by default, so a
# CDN connection that silently stalls (stops sending bytes without closing)
# can hang forever -- HF_HUB_DOWNLOAD_TIMEOUT bounds that, and the retry loop
# below resumes from the partial *.incomplete files (range requests) on the
# next attempt. Stale *.lock files from a killed attempt are cleared first.
export HF_HUB_DOWNLOAD_TIMEOUT=120
MODEL_DIR=./models/Qwen3-TTS-12Hz-1.7B-Base
for attempt in 1 2 3 4 5; do
  echo "--- snapshot_download attempt $attempt/5 ---"
  find "$MODEL_DIR" -name "*.lock" -delete 2>/dev/null || true
  if timeout -k 10 900 python3 - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    local_dir="$MODEL_DIR",
)
EOF
  then
    break
  fi
  if [ "$attempt" = "5" ]; then
    echo "ERROR: snapshot_download failed after 5 attempts"
    exit 1
  fi
  echo "attempt $attempt timed out or failed -- retrying (resumes from partial files) ..."
done

echo ""
echo "=== Bootstrap complete ==="
