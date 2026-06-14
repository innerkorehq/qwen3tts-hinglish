#!/usr/bin/env bash
# Downloads the Qwen3-TTS-12Hz-1.7B-Base checkpoint into ./models/.
#
# Called by orchestrate.py's onstart script *after* the dataset
# download/resampling stage (and its raw-data cleanup) and *before*
# train.py -- not during bootstrap_vastai.sh. This keeps the model's ~6.8GB
# off disk during build_manifest's peak-transient window (raw + resampled
# audio coexisting), which is the highest disk-usage point in the run. See
# docs/RUNBOOK.md "Disk budget".
#
# Usage (cwd=/root/work, ./models already created by the onstart script):
#   bash /root/repo/scripts/download_base_model.sh

set -euo pipefail

echo "=== Downloading base model (Qwen3-TTS-12Hz-1.7B-Base) ==="
mkdir -p ./models

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

echo "=== Base model download complete ==="
