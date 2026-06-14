#!/usr/bin/env bash
# Bootstrap script for Vast.ai A100 instance.
# Called automatically by orchestrate.py's onstart script (no manual SSH needed).
# Handles environment setup only: GPU/disk checks, Python deps. The base model
# download, dataset download, manifest building, encoding, and training are
# separate steps in orchestrate.py's onstart pipeline — see docs/RUNBOOK.md.
#
# Usage (called from onstart script with cwd=/root/work, on the instance's
# 250GB container disk):
#   bash /root/repo/scripts/bootstrap_vastai.sh

set -euo pipefail

echo "=== 1/3: Checking GPU ==="
nvidia-smi || { echo "ERROR: no GPU visible"; exit 1; }

echo "=== 2/3: Checking container disk space ==="
df -h .
FREE_GB=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
echo "Free space: ${FREE_GB}GB"
if [ "$FREE_GB" -lt 100 ]; then
  echo "WARNING: less than 100GB free. Dataset downloads + resampled audio +"
  echo "1.7B model checkpoints (fp32/bf16/fp16 variants) may not fit."
  echo "Consider increasing --disk on orchestrate.py (default 250, see"
  echo "docs/RUNBOOK.md 'Disk budget')."
fi

echo "=== 3/3: Installing Python deps ==="
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

echo "=== 3a/3: Installing sox ==="
# librosa/audioread fall back to sox (via subprocess) when soundfile can't
# read a file -- without it that fallback path errors with "SoX could not be
# found". Best-effort: not fatal if apt is unavailable/offline.
if command -v sox >/dev/null 2>&1; then
  echo "sox already present"
else
  (apt-get update -qq && apt-get install -y -qq sox libsox-fmt-all) > /dev/null 2>&1 \
    && echo "sox installed" \
    || echo "WARNING: sox install failed -- continuing (only used as a librosa/audioread fallback backend)"
fi

echo "=== 3b/3: Installing flash-attn (optional, faster attention) ==="
# Builds a CUDA extension from source -- can take 10-20+ minutes and needs
# nvcc; MAX_JOBS bounds parallel compile jobs to avoid OOM on many-core boxes.
# Best-effort and non-fatal: without it, qwen_tts just falls back to the
# manual PyTorch attention path (slower, but already works).
pip_retry ninja packaging
export MAX_JOBS=4
if timeout 1800 pip install -q flash-attn --no-build-isolation; then
  echo "flash-attn installed"
else
  echo "WARNING: flash-attn install failed or timed out -- continuing without it (slower attention, non-fatal)"
fi

mkdir -p ./data ./models

echo ""
echo "=== Bootstrap complete ==="
