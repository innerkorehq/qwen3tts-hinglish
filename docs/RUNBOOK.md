# Qwen3-TTS 1.7B Full Fine-Tune — Hinglish (HiACC + OpenSLR-104 Hindi-English/Bengali-English)

## Overview

Goal: full fine-tune of `Qwen/Qwen3-TTS-12Hz-1.7B-Base` on a combined manifest of:
- **HiACC** (Hinglish adult + children code-switched corpus, ~5.24 hrs)
- **OpenSLR-104** (MUCS 2021 subtask2): Hindi-English code-switched train
  (89.86 hrs) and Bengali-English code-switched train (46.11 hrs) — genuinely
  code-switched speech from spoken technical tutorials, CC BY-SA 4.0,
  https://www.openslr.org/104/

Combined dataset is roughly **140 hours** (5.24 + 89.86 + 46.11), a known
quantity up front — unlike the earlier IndicVoices-R-based plan, no dataset-size
discovery step is needed.

**Strategy: everything runs in a single Vast.ai A100 PCIe rental with a 250GB
container disk, orchestrated by `scripts/orchestrate.py`.** From your M4, one
command provisions the instance, clones this repo, and runs the entire pipeline
end-to-end on `/root/work` (the instance's container disk):

```
download HiACC + OpenSLR-104 (Hindi-English, Bengali-English)
  -> build unified manifest (resample to 24kHz mono, filter, train/eval split)
  -> backup raw manifests + resampled audio (as ~1GB resampled_shards/ tar shards) to R2
  -> free disk: remove raw sliced audio now that resampled/ exists
  -> encode audio -> discrete codes (Qwen3-TTS-Tokenizer-12Hz, on GPU)
  -> backup encoded codes to R2
  -> download base model (Qwen3-TTS-12Hz-1.7B-Base)
  -> full fine-tune (1.7B, single combined manifest)
  -> save fp32 master + bf16 + fp16 checkpoints
  -> attempt GGUF conversion (best-effort, non-fatal)
  -> upload all checkpoint variants + logs to R2
  -> write /root/DONE
```

The orchestrator polls for `/root/DONE` and **destroys the instance
automatically** on success, failure, or timeout — no manual provisioning or
teardown step, no local preprocessing on M4, and no separate volume to clean up
(the container disk is destroyed with the instance).

### Container disk architecture

A single `vastai create instance ... --disk 250` provisions a 250GB container
disk for the instance — no separate volume provisioning, no machine-pinning,
no two-phase search. All pipeline data (`/root/work/data/`,
`/root/work/models/`, `/root/work/checkpoints/`) lives on this disk. At the
end of the run, **only final and important intermediary artifacts are pushed
to R2** (raw manifests, resampled audio, encoded codes, all checkpoint
variants, logs) — `/root/work` disappears with the instance when destroyed, so
nothing left only there survives. Note (from Vast.ai docs): container disk
size is **fixed at instance creation** and can't be resized later — see "Disk
budget" below for why 250GB is sized with headroom.

### Disk budget (250GB container disk)

Steady-state and peak-transient usage, assuming the full ~141hr dataset
(HiACC 5.24hrs + SLR104 Hindi-English 89.86hrs + Bengali-English 46.11hrs):

| Component | Size |
|---|---|
| OS / pip / torch / CUDA baseline | ~15 GB |
| Raw dataset tarballs (during download, deleted after extract) | ~11.8 GB |
| Resampled audio (24kHz mono PCM16, ~141hrs) | ~24.4 GB |
| Encoded codes (jsonl) | ~2 GB |
| Base model download (Qwen3-TTS-12Hz-1.7B-Base) | ~6.8 GB |
| Checkpoints (3 recent + 1 best, ~17GB each) | ~68 GB |
| Final saves (fp32 + bf16 + fp16) | ~13.6 GB |
| llama.cpp clone (GGUF attempt) | ~2 GB |
| **Steady-state total** | **~144 GB** |
| Raw sliced audio (16kHz, before resample) — transient | +~16.3 GB |
| **Peak transient total** (during build_manifest, before cleanup) | **~153.2 GB** |

Against 250GB, that's **~97GB headroom at peak**. The onstart script removes
raw sliced audio (`./data/raw/slr104/*/audio`, `./data/raw/hiacc`) immediately
after `resampled/` is built and backed up to R2, bringing usage back down
before the base model is downloaded.

The base model download happens *after* the build_manifest/encode stages
(just before `train.py`, via `scripts/download_base_model.sh`) rather than
during bootstrap — this keeps its ~6.8GB off disk during the highest-disk-usage
window (raw + resampled audio coexisting), at the cost of a few extra minutes
of download time right before training starts.

If `FAILED_BUILD_MANIFEST`, `FAILED_ENCODE_*`, or `FAILED_TRAIN` shows a disk
space error in `pipeline_failed.log`, increase `--disk` (e.g. to 300) — remember
this can't be changed after instance creation, so a failed run on insufficient
disk requires a fresh instance.

**Why this is the simplest option despite higher A100 cost**: download + resample +
encode on M4 base could take many hours to days depending on MPS throughput.
Running everything on the A100 instead means the entire project — including
dataset acquisition — completes in one bounded rental window, at the cost of
paying A100 rates for I/O-bound steps (downloads ~11GB total for SLR104 train
tarballs + HiACC, resampling ~140hrs of audio) that don't strictly need a GPU.
Given A100 PCIe pricing (~$1.20/hr) and that these steps are likely 1-3 hours
total, this adds roughly **$1-4** to the run versus doing them for free on M4 — a
reasonable tradeoff for "press one button and walk away."

---

## One-time setup (M4 / local machine)

```bash
pip install vastai
vastai set api-key <your-vast-api-key>
```

Push this project to a git remote `orchestrate.py` can clone on the instance:
```bash
cd qwen3tts-finetune
git init && git add -A && git commit -m "initial"
git remote add origin <your-repo-url>
git push -u origin main
```

Set required environment variables:
```bash
export VAST_API_KEY=...                 # if not using `vastai set api-key`
export R2_ACCOUNT_ID=...
export R2_ACCESS_KEY_ID=...
export R2_SECRET_ACCESS_KEY=...
export R2_BUCKET=qwen3tts-finetune-data
export REPO_GIT_URL=https://github.com/yourname/qwen3tts-finetune.git
```

Optionally, set this to enable manual SSH access to the instance (e.g. to
`tail -f /root/pipeline.log` or poke around while the pipeline is running):
```bash
export VAST_SSH_PUBKEY_PATH=~/.ssh/id_ed25519.pub
```
If set, `orchestrate.py` attaches this key to the instance right after
creation and prints `vastai ssh-url <id>` so you can connect with
`ssh -i ~/.ssh/id_ed25519 $(vastai ssh-url <id> | sed 's#ssh://##')` (or just
run `vastai ssh-url <id>` again later to get the connection string).

Optionally, set a Hugging Face token to pass through to the instance:
```bash
export HF_TOKEN=hf_...
```
Not required for the current public `Qwen/Qwen3-TTS-12Hz-1.7B-Base` repo, but
recommended: anonymous HF downloads are subject to tighter rate limits, which
can contribute to the kind of stalled-CDN-connection hang described in
"Network robustness" below, and this also covers the model becoming gated
later. `orchestrate.py` passes `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) through
to the instance as `HF_TOKEN`, which `huggingface_hub` picks up automatically;
harmless if left unset.

No dataset auth tokens needed — OpenSLR-104 is a direct, ungated download
(CC BY-SA 4.0), and HiACC is from Zenodo (also ungated).

---

## Running the full pipeline

### Dry run (no cost)

```bash
python3 scripts/orchestrate.py --dry-run --max-price 1.20
```

Shows the top 5 matching A100 PCIe offers by price/reliability without creating
anything. Use this to sanity-check pricing before committing.

### Real run

```bash
python3 scripts/orchestrate.py \
  --max-price 1.20 \
  --disk 250 \
  --timeout-hours 30 \
  --slr104-pairs hindi-english bengali-english \
  --eval-frac 0.02
```

Flags:
- `--max-price` — max $/hr for A100 PCIe (searches `reliability>0.95` offers, picks cheapest)
- `--disk` — container disk size in GB (default 250 — see "Disk budget" above;
  steady-state ~144GB, peak transient ~160GB, ~90GB headroom). Fixed at
  instance creation, can't be resized later.
- `--timeout-hours` — safety net, not expected runtime (default 30)
- `--slr104-pairs` — which OpenSLR-104 code-switched pairs to download
  (default both: `hindi-english bengali-english`)
- `--slr104-include-test` — also download the smaller test splits (Hindi-English
  5.18hrs, Bengali-English 7.02hrs) for additional eval data; off by default
- `--eval-frac` — fraction of the combined dataset held out for eval (default 0.02)
- `--poll-interval` — seconds between status checks (default 300 for long runs)

### What happens

1. `search_offers()` finds the cheapest A100 PCIe offer with `disk_space >= --disk`
   and `reliability > 0.95`
2. `create_instance()` launches it with `--disk 250` and an onstart script
   containing the full pipeline
3. The instance immediately starts running in the background (no SSH needed) —
   and is **fully self-sufficient from here**: see "Self-destruct and handoff"
   below. It will finish (or fail) and shut itself down without any further
   help from the orchestrator or your local machine.
4. The orchestrator (if still running) polls instance status + `/root/DONE`
   via `vastai copy` every `--poll-interval` seconds, purely to show progress
   and to fetch the final status/log
5. On `SUCCESS`, `FAILED_*`, `TIMEOUT`, or a bad instance state (`exited`/`offline`),
   the instance is **destroyed** (by itself, or as an idempotent backstop by
   the orchestrator) — billing stops, and the container disk is destroyed with
   it (no separate cleanup step)

### `/root/DONE` status codes

- `SUCCESS` — all checkpoint variants + logs uploaded to R2 (see Output formats below)
- `FAILED_BOOTSTRAP` — dependency install (GPU/disk checks, Python deps) failed
- `FAILED_DOWNLOAD_HIACC` — HiACC zenodo download/extract failed
- `FAILED_DOWNLOAD_SLR104` — OpenSLR-104 download/extract/parse failed (e.g. mirror
  unreachable, or Kaldi data dir structure inside the tarball didn't match
  `find_kaldi_dir()`'s expectations)
- `FAILED_BUILD_MANIFEST` — manifest building/resampling failed, or produced no
  eval split
- `FAILED_ENCODE_TRAIN` / `FAILED_ENCODE_EVAL` — audio-to-codes tokenization failed
- `FAILED_DOWNLOAD_MODEL` — base model (`Qwen3-TTS-12Hz-1.7B-Base`) download failed
- `FAILED_TRAIN` — training itself failed
- `FAILED_CONVERT_FP16` — fp32-to-fp16 conversion failed (training succeeded,
  fp32/bf16 checkpoints likely still present — check pipeline_failed.log)
- `FAILED_UPLOAD_MODEL` — checkpoint upload to R2 failed (after 3 retries with
  30s backoff) despite training succeeding — see "Self-destruct and handoff"
  for why this is no longer the unrecoverable worst case it used to be
- `UNKNOWN_EXIT` — the script exited via an unexpected crash or signal that
  skipped `mark_done` entirely (the cleanup trap writes this as a fallback so
  the instance still self-destructs and uploads logs)
- `ERROR_PROVISIONING` — failed before the pipeline even started, e.g.
  `create_instance()` itself failed (offer no longer available, account
  issue, etc.). No instance to destroy in this case.
- `INSTANCE_EXITED` / `INSTANCE_OFFLINE` — the instance itself reached a bad
  state (container crashed or host disconnected) before writing `/root/DONE`

On any non-`SUCCESS` status, the orchestrator pulls `/root/pipeline.log` (via
`vastai copy`, or from R2 if the instance already self-destructed) to
`./pipeline_failed.log` locally, so you can debug without re-renting. The same
log, plus `status.txt` and a timestamped `pipeline_failed_<status>_<ts>.log`,
are always in R2 under `finetune/qwen3tts-hinglish/logs/` regardless of
whether the orchestrator was even running — see "Self-destruct and handoff".

**GGUF conversion failure does NOT set a FAILED status** — it's expected to likely
fail for Qwen3-TTS (see Output formats below) and the pipeline continues regardless.

---

## Self-destruct and handoff

Once `vastai create instance` returns, **the instance is on its own** — the
onstart script is a complete handoff. You can close your laptop, lose your
internet connection, or `Ctrl-C` the orchestrator, and the run still completes
(or fails) and stops billing without further input.

This is implemented with a bash `trap` near the top of the onstart script that
runs `cleanup()` on **every** exit path — normal completion, any
`mark_done FAILED_*; exit 1`, or an unexpected crash/signal (`EXIT`/`TERM`/`INT`).
`cleanup()`:

1. If `/root/DONE` was never written (an unexpected crash), writes `UNKNOWN_EXIT`
   so the run still has a final status.
2. Uploads `/root/pipeline.log` to R2 at `finetune/qwen3tts-hinglish/logs/pipeline.log`
   (always overwritten — latest run) and the contents of `/root/DONE` to
   `finetune/qwen3tts-hinglish/logs/status.txt`.
3. On any non-`SUCCESS` status, additionally uploads a timestamped copy to
   `finetune/qwen3tts-hinglish/logs/pipeline_failed_<status>_<ts>.log` — so
   repeated failed attempts don't clobber each other's logs ("create an R2
   artifact for the error").
4. Self-destructs the instance:
   `vastai destroy instance $CONTAINER_ID --api-key $CONTAINER_API_KEY -y`.
   `CONTAINER_ID` and `CONTAINER_API_KEY` are injected automatically by Vast.ai
   into every instance — no extra credentials need to be passed in. `vastai`
   itself is `pip install`ed as the very first step of the onstart script (before
   the repo clone or bootstrap), so this works even if `FAILED_BOOTSTRAP` fires.

All of this is **idempotent and best-effort** (`|| true` everywhere except the
final destroy): if R2 is unreachable, the instance still self-destructs (cost
control wins); if `vastai` itself somehow isn't usable, the orchestrator's own
`destroy_instance()` call is a harmless backstop — destroying an
already-destroyed instance just fails quietly.

### Ctrl-C / re-attaching

`Ctrl-C` (or closing the terminal) only stops **local polling** — the
orchestrator does **not** destroy the instance in this case, since that would
defeat the point of the handoff. The remote pipeline and its `cleanup()` trap
above keep running and will self-destruct on their own when the run finishes.

To resume watching an existing run (from this machine or another), use:

```bash
python3 scripts/orchestrate.py --attach INSTANCE_ID
```

This skips offer search/creation and goes straight to polling
`vastai show instance` / the `/root/DONE` marker for that instance, with the
same `--timeout-hours`/`--poll-interval` and the same end-of-run backstop
destroy for `TIMEOUT`/`INSTANCE_EXITED`/`INSTANCE_OFFLINE`. Find the instance
id from the `Instance created: id=...` line printed at creation time, or via
`vastai show instances-v1 --raw`.

Only the orchestrator's own give-up paths (`--timeout-hours`,
`INSTANCE_EXITED`/`INSTANCE_OFFLINE`, provisioning errors) trigger the
backstop `destroy_instance()` call — Ctrl-C never does.

**Checkpoint uploads get retries, not silent loss**: `FAILED_UPLOAD_MODEL`
(step 8) retries each of `final_fp32`/`final_bf16`/`final_fp16` up to 3 times
with 30s backoff before giving up — a completed training run is the most
expensive thing to lose, so it gets a few chances against a transient R2 hiccup
before the instance (correctly) self-destructs anyway.

**If the orchestrator is offline when the run finishes**: when you next run
`orchestrate.py` (or just check manually), `get_instance_status()` will find
the instance gone. The orchestrator then downloads
`finetune/qwen3tts-hinglish/logs/status.txt` from R2 to learn the final status,
and pulls `pipeline.log` from R2 instead of `vastai copy`. If `SUCCESS`,
download the checkpoint variants as usual (see "Verify and download results"
below). If `FAILED_*`, fix the issue and re-run with `--resume` (see "Resuming
a failed run") — every stage that completed before the failure (manifest,
codes, training checkpoints) was already pushed to R2 incrementally, so
`--resume` picks up from there rather than restarting.

---

## Network robustness

A rented A100 instance's network can stall or drop mid-transfer at any point —
model download, dataset download, or R2 upload/download. Without bounds, a
stalled connection that never closes can hang **forever** (no read-timeout),
burning cost until the 30h `--timeout-hours` backstop. Every network call in
the pipeline now has an explicit read-timeout + retry/backoff + resume:

- **`scripts/net_utils.py`** — shared helpers:
  - `download_with_retry(url, dest)`: resumable HTTP download (Range requests
    against a `.part` file), `HTTP_TIMEOUT=(15, 60)` read-timeout, up to 6
    attempts with exponential backoff. Used by `download_hiacc.py` and as the
    fallback path in `download_slr104.py`.
  - `r2_boto_config()`: boto3 `Config` with `connect_timeout=10`,
    `read_timeout=120`, and `retries={"max_attempts": 6, "mode": "adaptive"}`.
    Used by `upload_to_r2.py`'s `get_client()` and `train.py`'s
    `_r2_client_and_bucket()`.
  - `retry(fn, ...)`: generic exponential-backoff retry, used to wrap every
    `upload_file`/`download_file`/`head_object`/`list_objects_v2` call in
    `upload_to_r2.py` and the Zenodo metadata fetch in `download_hiacc.py`.
- **`download_base_model.sh`**: `HF_HUB_DOWNLOAD_TIMEOUT=120` bounds
  `snapshot_download`'s per-chunk read; a 5-attempt retry loop (each capped at
  15min via `timeout -k 10 900`) clears stale `*.lock` files and resumes from
  `*.incomplete` files. `pip install` (deps, in `bootstrap_vastai.sh`) also
  retries 5x with backoff.
- **`download_slr104.py`**: `curl -C -` (resume) with `--connect-timeout 15
  --speed-limit 1024 --speed-time 60` (abort+retry if throughput drops below
  1KB/s for 60s — the exact "stalled but not closed" case), 6 attempts, then
  falls back to `download_with_retry`.
- **Onstart script (`orchestrate.py`)**: the repo `git clone` and the
  `llama.cpp` clone (step 7, GGUF) each retry up to 5x/3x with backoff
  (`rm -rf` + re-clone, since `git clone` fails into a non-empty dir). The
  `pip install vastai` step (needed for self-destruct, see above) retries 5x —
  if this silently failed permanently, the cleanup trap's self-destruct
  wouldn't have a `vastai` binary to call.

Net effect: a single dropped connection or a multi-minute CDN stall costs a
few minutes of retries, not the whole run.

---

## M4 test for download_slr104.py

`tests/test_download_slr104.py` runs locally on a Mac (M4 or any machine with
the project's Python deps — no GPU needed):

```bash
pip install -r requirements.txt
python3 tests/test_download_slr104.py            # offline only, seconds
python3 tests/test_download_slr104.py --live      # + live smoke test, ~30s + ~580MB download
```

**Offline mode** builds a synthetic Kaldi-format dataset (wav.scp + segments +
text + generated tones, including a Hinglish code-switched transcript and a
`cat`-based pipe-command `wav.scp` entry) and runs `find_kaldi_dir()`,
`parse_wav_scp()`, `parse_segments()`, `parse_text()`, `resolve_recording()`, and
`process_pair()` against it. No network needed, runs in seconds.

**Live mode** additionally downloads the Bengali-English *test* split (606MB —
not the multi-GB train tarballs) from the real OpenSLR-104 mirror and runs
`find_kaldi_dir()` + `process_pair()` (limited to 5 segments) against the actual
archive structure.

**This live test already caught and fixed a real bug**: OpenSLR-104's `wav.scp`
contains paths relative to the *split root* (e.g. `test/`), not relative to the
`transcripts/` subdirectory that contains `wav.scp`/`segments`/`text` itself.
`resolve_recording()` originally resolved relative paths against `kaldi_dir`
(= `transcripts/`), which silently resolved to non-existent files and caused
`process_pair()` to skip 100% of segments. It now tries `kaldi_dir`,
`kaldi_dir.parent`, and `kaldi_dir.parent.parent` as candidate bases. Verified
against the real Bengali-English test split: `find_kaldi_dir()` correctly found
`test/transcripts`, and 5/5 sample segments processed with Bengali-English
code-switched text intact.

---

## Before your first real run — remaining things to verify

The live test above verified `find_kaldi_dir()` and `resolve_recording()` against
the Bengali-English **test** split. The following are still unverified before
committing to the full run:

- **Train tarball structure**: only the (smaller) test-split tarball was checked
  live. The train tarballs (7.3GB Hindi-English, 3.9GB Bengali-English) are much
  larger and weren't downloaded for verification — their internal layout is
  likely the same (`<split>/transcripts/{wav.scp,segments,text}` with audio files
  in `<split>/`), but if `FAILED_DOWNLOAD_SLR104` occurs on the real run, check
  `pipeline.log` for "could not find wav.scp+segments+text" first. **This is the
  one remaining unverified item** — see the small-pair smoke test below.
- **`wav.scp` pipe-command entries**: the Bengali-English test split's `wav.scp`
  had 40/40 direct-path entries (no `sox ... |` pipes). Train tarballs may differ;
  `resolve_recording()`'s pipe-command branch (using `cat`/`sox` via shell) is
  exercised by the offline test but not against real pipe-command data.
- ~~**HiACC archive structure**~~ — resolved. `find_hiacc_pairs()` has been
  rewritten against the real downloaded `Corpus.zip` layout — see "HiACC corpus
  structure" below.
- ~~**`encode_codes.py`/`train.py` placeholders**~~ — resolved. Both now use the
  real `qwen_tts` finetuning API (`Qwen3TTSTokenizer`, `Qwen3TTSModel` +
  `accelerate`) — see "Pipeline internals: data format, encoding, and training"
  below.
- **`train_config.yaml` batch size**: `training.batch_size: 4` /
  `gradient_accumulation_steps: 8` (effective batch 32) is the current best guess
  for A100 80GB with a 1.7B full fine-tune. If `FAILED_TRAIN` shows an OOM, drop
  `batch_size` to 2 and raise `gradient_accumulation_steps` to 16 (same effective
  batch size).

To validate the remaining items cheaply before the full ~140hr run, run with a
single, smaller pair first:
```bash
python3 scripts/orchestrate.py --slr104-pairs bengali-english --timeout-hours 6 --resume
```
Bengali-English train alone is ~46hrs (3.9GB download) — a faster, cheaper
(~$5-10) end-to-end pass that exercises every pipeline stage (including the new
encode/train code and `--resume` path) and surfaces the train-tarball structure
issue above before committing to the full Hindi-English + Bengali-English run.

---

## Output formats and cross-device usage

`train.py` produces three checkpoint variants, all uploaded to R2 under
`finetune/qwen3tts-hinglish/`:

| Variant | R2 key | Use case |
|---|---|---|
| `final_fp32` | `final_fp32/` | Source-of-truth master. Use as input for any further conversion. Largest (~6.8GB for 1.7B params). |
| `final_bf16` | `final_bf16/` | Matches training dtype, ready to load directly on CUDA (Hetzner prod). |
| `final_fp16` | `final_fp16/` | Recommended for MPS (M4 dev) inference via standard `transformers`. Broadest cross-device support. |
| `final_gguf` | `final_gguf/` (if present) | Attempted llama.cpp GGUF conversion — see caveat below. |

Raw manifests, resampled audio, and encoded codes are also backed up to R2
(`manifest_raw.jsonl`, `manifest_eval_raw.jsonl`, `resampled_shards/` (~1GB
tar shards + `manifest.json`), `train_with_codes.jsonl`, `eval_with_codes.jsonl`)
— a failed run after the manifest/encode stage doesn't require re-downloading
or re-resampling on a fresh rental: `orchestrate.py --resume` downloads and
reuses these (see "Resuming a failed run").

### GGUF caveat

`convert_model.py --to gguf` attempts conversion via llama.cpp's
`convert_hf_to_gguf.py`. **This is likely to fail for Qwen3-TTS** because
llama.cpp's converter only supports a fixed list of named model architectures
(standard `Qwen3ForCausalLM` text models are supported, but Qwen3-TTS is a
TTS-specific architecture with an audio codec component that almost certainly
isn't in that list).

The pipeline does **not** treat GGUF failure as fatal — it logs the failure and
continues, since `final_fp32`/`final_fp16` are already usable cross-device via
standard `transformers` on MPS, CUDA, or CPU.

If GGUF/llama.cpp support matters later:
1. Check for updated llama.cpp architecture support (`git pull` in `./llama.cpp`,
   rerun `convert_model.py --to gguf`)
2. Use `final_fp16` with standard `transformers` on MPS instead — no GGUF needed
3. If Qwen3-TTS's LM "trunk" is separable from its audio codec and the trunk is a
   standard `Qwen3ForCausalLM`, only that portion could potentially be
   GGUF-converted — but this requires splitting the checkpoint into separate save
   directories, which `train.py` does not currently do

### Loading on different devices

Load `final_fp16` (MPS/CPU) or `final_bf16` (CUDA) via standard
`transformers.AutoModelForCausalLM.from_pretrained(path, torch_dtype=...)`.
`config.json` in each variant has had `attn_implementation`, `device_map`, and
`quantization_config` stripped by `_clean_config_for_portability()`, so set
`attn_implementation` explicitly at load time based on the runtime device
(`"sdpa"` is a safe default for both MPS and CUDA; avoid `"flash_attention_2"`
outside CUDA).

---

## HiACC corpus structure

`download_hiacc.py` pulls `Corpus.zip` from Zenodo record 15551669. Its real
layout (verified by downloading and inspecting the archive):

| Path | Contents |
|---|---|
| `Corpus/adult/audio/{train,val,test}_split/<PID+num>.wav` | Adult speaker audio |
| `Corpus/adult/transcription/combined_output_changed_{train,val,test}_output.txt` | Adult transcripts |
| `Corpus/children/audio/{train,val,test}_split/<PID+num>.wav` | Child speaker audio |
| `Corpus/children/transcript/{train,val,test}_output.txt` | Child transcripts |

Each transcript line is `<filename>.wav, <text>` (split on the first `, `).
`find_hiacc_pairs()` (in `build_manifest.py`):
- locates the `adult`/`children` directories under the extracted corpus
  (case-insensitive `rglob`)
- for each split's transcript file, resolves the audio path at
  `<category>/audio/<split>_split/<filename>.wav`, skipping lines whose audio
  file doesn't exist
- sets `speaker_id` to the first 4 characters of the filename (matches the
  `speaker_info.csv` PIDs, e.g. `AD09001.wav` -> `AD09`, `CH03001.wav` -> `CH03`)
- sets `source` to `hiacc_{adult|children}_{split}` and `lang` to `"hinglish"`

Covered by the offline test `tests/test_build_manifest.py`.

---

## Pipeline internals: data format, encoding, and training

### Manifest schema

Every record produced by `build_manifest.py` (HiACC + OpenSLR-104) has, after
resampling:
```json
{"audio": "<path to 24kHz mono wav>", "ref_audio": "<same path>", "text": "...",
 "lang": "hinglish", "speaker_id": "...", "source": "hiacc_adult_train"}
```
`ref_audio` is a **self-reference** — each clip is its own speaker reference for
`model.speaker_encoder` during training. This is the simplest correct choice for
a multi-speaker corpus: it avoids grouping clips by speaker, and every record
already carries its own reference audio.

### Encoding (`encode_codes.py`)

Uses the real `qwen_tts` tokenizer API:
```python
tokenizer = Qwen3TTSTokenizer.from_pretrained(args.tokenizer_model, device_map=device)
enc_res = tokenizer.encode([rec["audio"] for rec in batch])   # enc_res.audio_codes: list of (T,16) tensors
```
Batched (`--batch-size 32` default) with per-item fallback on batch failure.
Output records add an `audio_codes` field (nested list, `(T,16)`) to each
manifest record, written to `train_with_codes.jsonl` / `eval_with_codes.jsonl`.
Restartable via `.partial` files + `load_done_keys()` + periodic checkpoint
flush, same pattern as before.

### Training (`train.py`)

Not HF `Trainer` — a hand-rolled `accelerate` loop over `Qwen3TTSModel`, using
`scripts/qwen_tts_dataset.py` (vendored verbatim from the official
[`finetuning/dataset.py`](https://github.com/QwenLM/Qwen3-TTS/blob/main/finetuning/dataset.py),
since it's not part of the `qwen-tts` pip package). The forward pass and loss
follow [`finetuning/sft_12hz.py`](https://github.com/QwenLM/Qwen3-TTS/blob/main/finetuning/sft_12hz.py)
exactly: text + codec embeddings, per-codebook embeddings for codebooks 1-15,
`model.talker(...)` loss plus `0.3 * sub_talker_loss` from
`forward_sub_talker_finetune()`.

**Multi-speaker design decision**: the official script captures a *single*
`target_speaker_embedding` from the first batch and bakes it into
`codec_embedding.weight[3000]` plus `talker_config.spk_id`/`custom_voice` —
a single-target-voice adaptation. This project is a general multi-speaker
corpus, so `train.py` keeps the **per-sample** speaker embedding (from each
record's own `ref_audio` via `model.speaker_encoder`) for training
conditioning, but does **not** perform that bake-in at save time. Checkpoints
stay loadable exactly like the Base model, with arbitrary `--ref_audio` at
inference.

Other details: cosine LR schedule with warmup
(`get_cosine_schedule_with_warmup`), eval loop (mean loss over `eval_jsonl`)
after each epoch, `--max-steps` for dry-run timing.

**Checkpoint layout** under `./checkpoints/` (= `output_model_path`):

| Path | Contents |
|---|---|
| `accel_state/` | `accelerator.save_state()` — model+optimizer+scheduler+RNG, overwritten each epoch |
| `training_state.json` | `{epoch, global_step, best_eval_loss, best_epoch}` |
| `best/` | Full HF-loadable checkpoint from the best-eval-loss epoch |
| `final_fp32/`, `final_bf16/`, `final/` (alias of `final_fp32`) | Produced once at the end |

---

## Resuming a failed run

`python3 scripts/orchestrate.py --resume ...` (pass the same other flags as the
failed run) makes each pipeline stage check R2 for a prior run's output before
recomputing it:

- **Encode** (checked first): if `train_with_codes.jsonl` and
  `eval_with_codes.jsonl` exist in R2, downloads them directly and skips
  download/manifest/resampling entirely -- raw and resampled audio are only
  an intermediate for producing the encoded codes, so once the codes exist
  there's nothing left for those stages to produce.
- **Download + manifest** (only if encoded codes aren't in R2 yet): tries
  three resume tiers, newest first, by checking R2 for each scheme's output
  plus `manifest_raw.jsonl`/`manifest_eval_raw.jsonl`:
  - **`RESUME_SHARDS`** (current scheme): if `resampled_shards/manifest.json`
    exists in R2 and both manifests download successfully, downloads +
    extracts each `resampled_shards/shard_NNNNN.tar` via
    `shard_tar.py download` into `./data/resampled/`, skipping
    HiACC/OpenSLR-104 and `build_manifest.py` entirely (the large, slow steps).
  - **`RESUME_TAR`** (older scheme): if there's no shards manifest but a single
    `resampled.tar` exists in R2 and both manifests download successfully,
    downloads `resampled.tar` and extracts it to `./data/resampled/` the same
    way.
  - **`RESUME_FILES`** (oldest scheme): if neither of the above is present but
    the manifests still download successfully, downloads whatever individual
    `resampled/*` objects exist in R2 (best-effort) for `build_manifest.py` to
    reuse via its per-file resumability (for each expected output file, if it
    already exists non-empty under `--resampled-dir`, it reuses it -- reading
    the duration from the file header -- instead of re-decoding/re-resampling),
    then re-downloads HiACC/OpenSLR-104 and reruns `build_manifest.py`,
    resampling only what's missing.
  - If none of the above apply (e.g. the manifests themselves aren't in R2),
    runs the full download + `build_manifest.py` pipeline from scratch.
  - In any non-`RESUME_SHARDS` case, once `resampled/` exists (either freshly
    built or extracted from the older `resampled.tar`/`resampled/*` schemes),
    it's packed into `resampled_shards/` via `shard_tar.py upload`: ~1GB tar
    shards (`shard_NNNNN.tar`) plus a `manifest.json` uploaded last, once every
    shard has succeeded. Sharding bounds both the disk overhead (one ~1GB
    shard alongside `resampled/` at a time, not a full-size duplicate) and how
    much a flaky connection can waste on a reset (one shard, not the whole
    archive); a plain `manifest.json` existence check is unambiguous for
    resume (no partial-upload ambiguity like individual files had). Any
    `resampled.tar` or individual `resampled/*` objects left in R2 from an
    older run are deleted once `resampled_shards/` uploads successfully.

  If your R2 bucket still has a `resampled.tar` or individual `resampled/*`
  objects from a run before the sharded scheme existed, run
  `scripts/migrate_resampled_to_shards.py` locally (not on the instance) to
  convert it once, ahead of time -- it downloads + extracts whichever of the
  two older schemes is present, shards + uploads `resampled_shards/`, and
  deletes the superseded objects.
- **Train**: if `accel_state/` and `training_state.json` exist in R2, downloads
  them into `./checkpoints/` and passes `--resume` to `train.py`, which calls
  `accelerator.load_state()` and reads `training_state.json` to skip already-
  completed epochs.

`train.py` itself uploads `accel_state/` + `training_state.json` (and `best/`
if it improved) to R2 **after every epoch**, so a `FAILED_TRAIN` run is
recoverable on a fresh instance via `--resume` without losing more than the
current epoch's progress.

**Caveat**: resume reuses R2 state as-is. If you change `train_config.yaml`
(e.g. batch size, LR schedule) between runs, the restored optimizer/scheduler
state in `accel_state/` may not match the new config — for substantial config
changes, start a fresh run without `--resume` instead.

---

## Verify and download results

After `orchestrate.py` reports `SUCCESS`:

```bash
python3 scripts/upload_to_r2.py --download --bucket $R2_BUCKET \
  --key finetune/qwen3tts-hinglish/logs/pipeline.log --file ./pipeline.log
less ./pipeline.log   # check loss curve, eval samples, dataset hour counts found
```

Download the variant you need:
```bash
# For M4/MPS inference:
python3 scripts/upload_to_r2.py --download --recursive --bucket $R2_BUCKET \
  --key finetune/qwen3tts-hinglish/final_fp16 --file ./final_fp16

# For Hetzner/CUDA production:
python3 scripts/upload_to_r2.py --download --recursive --bucket $R2_BUCKET \
  --key finetune/qwen3tts-hinglish/final_bf16 --file ./final_bf16
```

---

## Cost summary

| Step | Where | Cost |
|---|---|---|
| Everything (download, resample, encode, train, convert, upload) | Vast.ai A100 PCIe + 250GB container disk via `orchestrate.py` | ~$1.20/hr x actual runtime (disk is included in `dph_total` for most offers) -- container destroyed on completion, no separate cleanup |
| R2 storage (raw manifests, resampled audio, codes, 3-4 checkpoint variants) | Cloudflare R2 | ~$0.015/GB/month, negligible |

Total cost = (A100 hourly rate) x (download + resample + encode + train + convert
time). The combined dataset (~140hrs: HiACC 5.24 + SLR104 Hindi-English 89.86 +
SLR104 Bengali-English 46.11) is a known quantity, so total cost is more
predictable than the earlier IndicVoices-R-based plan. Run a small first pass
(`--slr104-pairs bengali-english --timeout-hours 6`, ~$5-10) to validate the
pipeline and get a real per-phase timing breakdown from `pipeline.log` before
committing to the full two-pair run.

---

## Troubleshooting

- **`vastai copy ... Invalid src_full_path` during polling**: expected and
  benign. `check_done_marker()` polls for `/root/DONE` via `vastai copy`
  before the pipeline has finished, so the remote file doesn't exist yet --
  the backend reports this as `Invalid src_full_path`. `orchestrate.py` runs
  this check quietly and just treats it as "not done yet". If you see this
  *after* `pipeline.log` shows the pipeline finished, something else is
  wrong -- SSH in (see above) and check `/root/` directly.

- **`vastai destroy instance <id>` hangs / prints "Aborted." after pressing
  Enter**: the bare CLI prompts `Are you sure you want to destroy instance
  ...? [y/N]` and a bare Enter defaults to **No**, so the instance is *not*
  destroyed (billing continues!). Always pass `-y` for non-interactive use:
  `vastai destroy instance <id> -y`. `orchestrate.py`'s `destroy_instance()`
  already does this. If you ever end up with a leaked instance from a manual
  `vastai destroy instance <id>` that printed "Aborted.", re-run it with `-y`
  or destroy it from the Vast.ai web console.

---

## Known gaps / TODO

- [x] **Resume/checkpoint-skip across runs** — `orchestrate.py --resume` now
  checks R2 for each stage's prior output (manifest/resampled audio, encoded
  codes, training `accel_state`/`training_state.json`) and downloads instead of
  recomputing; `train.py --resume` picks up training mid-run. See "Resuming a
  failed run".
- [ ] **Disk size is fixed at instance creation** — if `--disk 250` proves
  insufficient (see "Disk budget"), a failed run can't be resized; a fresh
  instance with a larger `--disk` is required. The cleanup step
  (`./data/raw/slr104/*/audio`, `./data/raw/hiacc` removal after resampling)
  reduces peak usage from ~160GB to ~144GB, but hasn't been verified against
  real data sizes.
- [x] **`encode_codes.py`/`train.py`** now use the real `qwen_tts` finetuning API
  (`Qwen3TTSTokenizer`, `Qwen3TTSModel` + `accelerate` + vendored
  `TTSDataset`/`collate_fn`) — see "Pipeline internals: data format, encoding,
  and training". Still needs a real-GPU smoke test (see "Before your first real
  run").
- [x] **HiACC zenodo archive structure** — `find_hiacc_pairs()` rewritten against
  the real downloaded `Corpus.zip` layout; see "HiACC corpus structure". Covered
  by `tests/test_build_manifest.py`.
- [ ] **OpenSLR-104 Kaldi data dir structure and `wav.scp` pipe-command handling**
  — `download_slr104.py`'s `find_kaldi_dir()` and `resolve_recording()` are based
  on the standard Kaldi recipe layout described in the MUCS 2021 baseline, but
  haven't been run against the actual extracted **train** tarballs (only the
  smaller test split was verified live). This is the **last remaining open item**
  — the small-pair run below surfaces mismatches here cheaply.
- [ ] **`FAILED_UPLOAD_MODEL` after successful training** is the worst-case
  failure — checkpoints exist only on the instance, which gets destroyed.
  Consider: don't destroy on this specific failure (leave instance running for
  manual recovery via `vastai copy`), or retry upload with backoff before giving
  up. Not currently differentiated from other failure modes.
- [ ] **`train_config.yaml` batch size** (`batch_size: 4`,
  `gradient_accumulation_steps: 8`) is the current best guess for full FT of
  1.7B on A100 80GB — unverified on real hardware, adjust per the OOM fallback
  if `FAILED_TRAIN` shows OOM.
