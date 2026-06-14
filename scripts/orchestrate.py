#!/usr/bin/env python3
"""
End-to-end orchestrator for the Vast.ai A100 PCIe full pipeline, using a
200GB container disk for all working data.

Flow (everything in one rental, see docs/RUNBOOK.md):
  1. Search vast.ai for A100 PCIe offers with >= --disk GB (cheapest reliable match)
  2. Create an instance with --disk 200 (a single container disk -- no separate
     volume provisioning, no machine-pinning)
  3. Onstart script runs the whole pipeline on /root/work:
     bootstrap -> download HiACC + OpenSLR-104 (Hindi-English, Bengali-English)
     -> build manifest (resample, split) -> backup to R2 -> encode_codes.py
     -> backup codes to R2 -> train.py -> convert formats -> upload all
     checkpoint variants + logs to R2 -> write DONE marker
  4. Poll instance status + DONE marker
  5. On completion (or failure/timeout), destroy the instance -- container disk
     is destroyed with it, no separate cleanup step

Only final and important intermediary artifacts (raw manifests, resampled audio,
encoded codes, checkpoint variants, logs) go to R2 -- /root/work is scratch space
for the run and disappears with the instance.

Container disk sizing (see docs/RUNBOOK.md "Disk budget" for the full breakdown):
steady-state usage is ~144GB, peak transient (raw + resampled audio coexisting
during build_manifest) is ~160GB, against 200GB -- ~40GB headroom at peak.

Requires:
  - `vastai` CLI installed and authenticated (`vastai set api-key <key>`)
  - R2 env vars: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET
  - REPO_GIT_URL pointing at a git remote with this project (cloned on the instance)

Usage:
  export VAST_API_KEY=...
  export R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_BUCKET=...
  export REPO_GIT_URL=https://github.com/yourname/qwen3tts-finetune.git
  python3 orchestrate.py --max-price 1.20 --disk 200 --timeout-hours 30
  python3 orchestrate.py --dry-run     # search + show chosen offer, don't create
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_GIT_URL = os.environ.get("REPO_GIT_URL", "")  # set this to your git remote with these scripts
R2_PREFIX = "finetune/qwen3tts-hinglish"


def run(cmd, check=True, capture=True):
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=capture, text=True)
    if capture:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


def search_offers(max_price, min_disk, gpu="A100_PCIE"):
    """Search vast.ai for A100 PCIe offers with enough disk, sorted by price."""
    query = (
        f"gpu_name={gpu} "
        f"num_gpus=1 "
        f"reliability>0.95 "
        f"dph_total<={max_price} "
        f"disk_space>={min_disk} "
        f"rentable=true"
    )
    result = run(["vastai", "search", "offers", query, "--raw"], check=True)
    try:
        offers = json.loads(result.stdout)
    except json.JSONDecodeError:
        print("ERROR: could not parse vastai search offers output as JSON", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        sys.exit(1)

    if not offers:
        print(f"No offers found matching: {query}", file=sys.stderr)
        sys.exit(1)

    offers.sort(key=lambda o: o.get("dph_total", 999))
    return offers


def build_onstart_script(slr104_pairs, slr104_include_test, eval_frac):
    """
    Generates the bash script that runs entirely on the remote instance via
    --onstart-cmd. Full pipeline, all on the instance's container disk under
    /root/work: clone repo, bootstrap, download HiACC + OpenSLR-104
    (Hindi-English / Bengali-English code-switched), build manifest
    (resample + split), encode codes, train, convert formats, upload results,
    write DONE marker. Errors are captured so the orchestrator can detect failure
    via the marker file content.
    """
    if not REPO_GIT_URL:
        print("WARNING: REPO_GIT_URL is not set. The onstart script will expect", file=sys.stderr)
        print("the qwen3tts-finetune project to already be baked into the image, or", file=sys.stderr)
        print("you must edit build_onstart_script() to fetch it another way (e.g. R2).", file=sys.stderr)

    clone_cmd = f"git clone {REPO_GIT_URL} repo" if REPO_GIT_URL else "echo 'skip clone, assuming repo present'"

    SLR104_PAIRS = " ".join(slr104_pairs)
    SLR104_TEST_ARG = "--include-test" if slr104_include_test else ""
    EVAL_FRAC = eval_frac

    script = f"""#!/bin/bash
set -uo pipefail
exec > /root/pipeline.log 2>&1

echo "=== START $(date) ==="

cd /root
{clone_cmd}

# --- env for R2 ---
export R2_ACCOUNT_ID="{os.environ.get('R2_ACCOUNT_ID','')}"
export R2_ACCESS_KEY_ID="{os.environ.get('R2_ACCESS_KEY_ID','')}"
export R2_SECRET_ACCESS_KEY="{os.environ.get('R2_SECRET_ACCESS_KEY','')}"
export R2_BUCKET="{os.environ.get('R2_BUCKET','')}"

mark_done() {{
  echo "$1" > /root/DONE
  echo "=== END $(date) status=$1 ==="
}}

# --- All working data lives under /root/work on the instance's container disk ---
# (final and important intermediary artifacts are also pushed to R2 below;
# /root/work disappears with the instance at the end of the run)
REPO=/root/repo
mkdir -p /root/work
cd /root/work
mkdir -p data models checkpoints

if [ ! -d "$REPO" ]; then
  mark_done FAILED_BOOTSTRAP; exit 1
fi

# --- 1. bootstrap deps + base model (downloads base model into /root/work/models) ---
bash "$REPO/scripts/bootstrap_vastai.sh" || {{ mark_done FAILED_BOOTSTRAP; exit 1; }}

# --- 2. download datasets directly onto the container disk ---
python3 "$REPO/scripts/download_hiacc.py" --out-dir ./data/raw/hiacc \\
  || {{ mark_done FAILED_DOWNLOAD_HIACC; exit 1; }}

python3 "$REPO/scripts/download_slr104.py" \\
  --out-dir ./data/raw/slr104 \\
  --pairs {SLR104_PAIRS} \\
  {SLR104_TEST_ARG} \\
  || {{ mark_done FAILED_DOWNLOAD_SLR104; exit 1; }}

# --- 3. build unified manifest (resample to 24kHz mono, filter, train/eval split) ---
python3 "$REPO/scripts/build_manifest.py" \\
  --hiacc-dir ./data/raw/hiacc \\
  --slr104-dir ./data/raw/slr104 \\
  --out ./data/manifest_raw.jsonl \\
  --resampled-dir ./data/resampled \\
  --eval-frac {EVAL_FRAC} \\
  || {{ mark_done FAILED_BUILD_MANIFEST; exit 1; }}

# build_manifest.py writes the eval split as manifest_eval_raw.jsonl alongside
# the requested --out path (same stem with _eval_raw suffix)
if [ ! -f ./data/manifest_eval_raw.jsonl ]; then
  mark_done FAILED_BUILD_MANIFEST; exit 1
fi

# Backup raw manifests + resampled audio to R2 (important intermediary artifacts --
# if encoding or training fails later, you don't have to re-download/re-resample
# on a fresh rental)
python3 "$REPO/scripts/upload_to_r2.py" --file ./data/manifest_raw.jsonl --bucket "$R2_BUCKET" \\
  --key {R2_PREFIX}/manifest_raw.jsonl
python3 "$REPO/scripts/upload_to_r2.py" --file ./data/manifest_eval_raw.jsonl --bucket "$R2_BUCKET" \\
  --key {R2_PREFIX}/manifest_eval_raw.jsonl
python3 "$REPO/scripts/upload_to_r2.py" --file ./data/resampled --bucket "$R2_BUCKET" \\
  --key {R2_PREFIX}/resampled --recursive

# Free up disk: raw sliced audio (16kHz, from download_slr104.py / download_hiacc.py)
# is no longer needed once resampled/ exists. This matters at our disk budget --
# see docs/RUNBOOK.md "Disk budget".
rm -rf ./data/raw/slr104/*/audio ./data/raw/hiacc

# --- 4. encode audio -> codes ---
python3 "$REPO/scripts/encode_codes.py" \\
  --manifest ./data/manifest_raw.jsonl \\
  --out ./data/train_with_codes.jsonl \\
  --device cuda --batch-size 16 \\
  || {{ mark_done FAILED_ENCODE_TRAIN; exit 1; }}

python3 "$REPO/scripts/encode_codes.py" \\
  --manifest ./data/manifest_eval_raw.jsonl \\
  --out ./data/eval_with_codes.jsonl \\
  --device cuda --batch-size 16 \\
  || {{ mark_done FAILED_ENCODE_EVAL; exit 1; }}

# Backup encoded codes to R2 (important intermediary artifact -- small files)
python3 "$REPO/scripts/upload_to_r2.py" --file ./data/train_with_codes.jsonl --bucket "$R2_BUCKET" \\
  --key {R2_PREFIX}/train_with_codes.jsonl
python3 "$REPO/scripts/upload_to_r2.py" --file ./data/eval_with_codes.jsonl --bucket "$R2_BUCKET" \\
  --key {R2_PREFIX}/eval_with_codes.jsonl

# --- 5. train (produces final_fp32, final_bf16, final aliased to fp32) ---
# train_config.yaml paths (base_model_path, output_dir, data files) are
# relative to cwd, which is /root/work here -- see configs/train_config.yaml comment.
python3 "$REPO/scripts/train.py" --config "$REPO/configs/train_config.yaml" \\
  || {{ mark_done FAILED_TRAIN; exit 1; }}

# --- 6. convert fp32 master -> fp16 (for MPS/M4 inference) ---
python3 "$REPO/scripts/convert_model.py" \\
  --master ./checkpoints/final_fp32 \\
  --to fp16 --out ./checkpoints/final_fp16 \\
  || {{ mark_done FAILED_CONVERT_FP16; exit 1; }}

# --- 7. attempt GGUF conversion (may fail for Qwen3-TTS -- see convert_model.py
#        docstring; failure here does NOT abort the pipeline, since fp32/fp16
#        checkpoints are already usable for cross-device inference) ---
git clone --depth 1 https://github.com/ggml-org/llama.cpp ./llama.cpp 2>/dev/null || true
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

# --- 8. upload all checkpoint variants + logs (final artifacts -> R2) ---
python3 "$REPO/scripts/upload_to_r2.py" --file ./checkpoints/final_fp32 --bucket "$R2_BUCKET" \\
  --key {R2_PREFIX}/final_fp32 --recursive \\
  || {{ mark_done FAILED_UPLOAD_MODEL; exit 1; }}

python3 "$REPO/scripts/upload_to_r2.py" --file ./checkpoints/final_bf16 --bucket "$R2_BUCKET" \\
  --key {R2_PREFIX}/final_bf16 --recursive \\
  || {{ mark_done FAILED_UPLOAD_MODEL; exit 1; }}

python3 "$REPO/scripts/upload_to_r2.py" --file ./checkpoints/final_fp16 --bucket "$R2_BUCKET" \\
  --key {R2_PREFIX}/final_fp16 --recursive \\
  || {{ mark_done FAILED_UPLOAD_MODEL; exit 1; }}

# Upload GGUF only if it was produced (best-effort, doesn't fail the run)
if [ -d ./checkpoints/final_gguf ]; then
  python3 "$REPO/scripts/upload_to_r2.py" --file ./checkpoints/final_gguf --bucket "$R2_BUCKET" \\
    --key {R2_PREFIX}/final_gguf --recursive || true
fi

python3 "$REPO/scripts/upload_to_r2.py" --file /root/pipeline.log --bucket "$R2_BUCKET" \\
  --key {R2_PREFIX}/logs/pipeline.log

mark_done SUCCESS
"""
    return script


def create_instance(offer_id, disk_gb, onstart_script_path,
                     image="pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime"):
    """Create an instance with a single --disk GB container disk (no separate volume)."""
    result = run([
        "vastai", "create", "instance", str(offer_id),
        "--image", image,
        "--disk", str(disk_gb),
        "--onstart", str(onstart_script_path),
        "--ssh", "--direct",
        "--raw",
    ])
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


def check_done_marker(instance_id):
    """SSH-free check via `vastai copy` of the DONE marker file, if it exists."""
    tmp = Path("/tmp/DONE_marker_check")
    if tmp.exists():
        tmp.unlink()
    result = run([
        "vastai", "copy",
        f"{instance_id}:/root/DONE",
        f"local:{tmp}",
    ], check=False)
    if tmp.exists():
        content = tmp.read_text().strip()
        tmp.unlink()
        return content
    return None


def destroy_instance(instance_id):
    run(["vastai", "destroy", "instance", str(instance_id)], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-price", type=float, default=1.20, help="max $/hr for A100 PCIe")
    ap.add_argument("--disk", type=int, default=200,
                    help="container disk size in GB (default 200 -- see "
                         "docs/RUNBOOK.md 'Disk budget' for the breakdown; "
                         "steady-state ~144GB, peak transient ~160GB)")
    ap.add_argument("--gpu", default="A100_PCIE")
    ap.add_argument("--poll-interval", type=int, default=300,
                    help="seconds between status checks (default 300 for long runs)")
    ap.add_argument("--timeout-hours", type=float, default=30.0,
                    help="safety net, not expected runtime (default 30)")
    ap.add_argument("--slr104-pairs", nargs="+", default=["hindi-english", "bengali-english"],
                    choices=["hindi-english", "bengali-english"],
                    help="OpenSLR-104 code-switched pairs to download")
    ap.add_argument("--slr104-include-test", action="store_true",
                    help="also download the (smaller) OpenSLR-104 test splits")
    ap.add_argument("--eval-frac", type=float, default=0.02,
                    help="fraction of combined dataset held out for eval")
    ap.add_argument("--dry-run", action="store_true", help="search and show offer only, don't create")
    args = ap.parse_args()

    for var in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
        if not os.environ.get(var):
            print(f"ERROR: {var} not set", file=sys.stderr)
            sys.exit(1)

    print(f"Searching for {args.gpu} offers under ${args.max_price}/hr "
          f"with >= {args.disk}GB disk ...")
    offers = search_offers(args.max_price, args.disk, gpu=args.gpu)
    best = offers[0]
    print(f"\\nTop {min(5, len(offers))} offers:")
    for o in offers[:5]:
        print(f"  id={o['id']} ${o['dph_total']:.3f}/hr "
              f"reliability={o.get('reliability', 0):.3f} "
              f"disk={o.get('disk_space')}GB "
              f"loc={o.get('geolocation','?')}")

    print(f"\\nSelected offer id={best['id']} at ${best['dph_total']:.3f}/hr")

    if args.dry_run:
        print("\\n--dry-run set, not creating instance.")
        return

    onstart_path = Path("/tmp/onstart.sh")
    onstart_path.write_text(build_onstart_script(
        slr104_pairs=args.slr104_pairs,
        slr104_include_test=args.slr104_include_test,
        eval_frac=args.eval_frac,
    ))
    print(f"\\nGenerated onstart script -> {onstart_path}")

    print("\\nCreating instance ...")
    instance_id = None
    final_status = None

    try:
        instance_id = create_instance(best["id"], args.disk, onstart_path)
        print(f"Instance created: id={instance_id}")
        print("Pipeline is now running in the background via onstart script.")
        print("Polling for completion ...")

        start = time.time()
        timeout_sec = args.timeout_hours * 3600

        while True:
            elapsed = time.time() - start
            if elapsed > timeout_sec:
                print(f"\\nTIMEOUT after {args.timeout_hours}h -- destroying instance.")
                final_status = "TIMEOUT"
                break

            info = get_instance_status(instance_id)
            actual_status = info.get("actual_status") if info else None
            print(f"  [{elapsed/60:.1f}min] instance status: {actual_status}")

            if actual_status in ("exited", "offline"):
                print(f"\\nInstance reached terminal bad state: {actual_status}")
                final_status = f"INSTANCE_{actual_status.upper()}"
                break

            marker = check_done_marker(instance_id)
            if marker:
                print(f"\\nDONE marker found: {marker}")
                final_status = marker
                break

            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print("\\nInterrupted by user.")
        final_status = "INTERRUPTED"

    except RuntimeError as e:
        print(f"\\nERROR: {e}")
        final_status = "ERROR_PROVISIONING"

    print(f"\\nFinal status: {final_status}")

    if final_status == "SUCCESS":
        print(f"Results uploaded to R2 under {R2_PREFIX}/ "
              f"(final_fp32/, final_bf16/, final_fp16/, manifest/codes backups, logs/)")
    elif final_status and final_status.startswith("FAILED") and instance_id is not None:
        print("Pipeline failed mid-run. Pulling log before destroying instance ...")
        run(["vastai", "copy", f"{instance_id}:/root/pipeline.log", "local:./pipeline_failed.log"], check=False)
        print("Saved to ./pipeline_failed.log -- inspect before retrying.")

    if instance_id is not None:
        print(f"\\nDestroying instance {instance_id} ...")
        destroy_instance(instance_id)
        print("Done. Billing stopped, container disk destroyed with the instance.")
    else:
        print("Done. (No instance was created.)")


if __name__ == "__main__":
    main()
