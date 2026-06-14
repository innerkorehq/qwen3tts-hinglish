#!/usr/bin/env bash
# Bootstrap script for Vast.ai A100 instance.
# Called automatically by orchestrate.py's onstart script (no manual SSH needed).
# Handles environment setup only: GPU/disk checks, Python deps, base model
# download. Dataset download, manifest building, encoding, and training are
# separate steps in orchestrate.py's onstart pipeline — see docs/RUNBOOK.md.
#
# Usage (called from onstart script with cwd=/root/work, on the instance's
# 200GB container disk):
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
  echo "Consider increasing --disk on orchestrate.py (default 200, see"
  echo "docs/RUNBOOK.md 'Disk budget')."
fi

echo "=== 3/4: Installing Python deps ==="
pip install -q -U pip
pip install -q qwen-tts transformers accelerate deepspeed boto3 soundfile librosa \
  datasets huggingface_hub pydub requests pyyaml

echo "=== 4/4: Downloading base model ==="
mkdir -p ./data ./models
python3 - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    local_dir="./models/Qwen3-TTS-12Hz-1.7B-Base",
)
EOF

echo ""
echo "=== Bootstrap complete ==="
