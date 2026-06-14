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

**Strategy: everything runs in a single Vast.ai A100 PCIe rental with a 200GB
container disk, orchestrated by `scripts/orchestrate.py`.** From your M4, one
command provisions the instance, clones this repo, and runs the entire pipeline
end-to-end on `/root/work` (the instance's container disk):

```
download HiACC + OpenSLR-104 (Hindi-English, Bengali-English)
  -> build unified manifest (resample to 24kHz mono, filter, train/eval split)
  -> backup raw manifests + resampled audio to R2
  -> free disk: remove raw sliced audio now that resampled/ exists
  -> encode audio -> discrete codes (Qwen3-TTS-Tokenizer-12Hz, on GPU)
  -> backup encoded codes to R2
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

A single `vastai create instance ... --disk 200` provisions a 200GB container
disk for the instance — no separate volume provisioning, no machine-pinning,
no two-phase search. All pipeline data (`/root/work/data/`,
`/root/work/models/`, `/root/work/checkpoints/`) lives on this disk. At the
end of the run, **only final and important intermediary artifacts are pushed
to R2** (raw manifests, resampled audio, encoded codes, all checkpoint
variants, logs) — `/root/work` disappears with the instance when destroyed, so
nothing left only there survives. Note (from Vast.ai docs): container disk
size is **fixed at instance creation** and can't be resized later — see "Disk
budget" below for why 200GB is sized with headroom.

### Disk budget (200GB container disk)

Steady-state and peak-transient usage, assuming the full ~141hr dataset
(HiACC 5.24hrs + SLR104 Hindi-English 89.86hrs + Bengali-English 46.11hrs):

| Component | Size |
|---|---|
| OS / pip / torch / CUDA baseline | ~15 GB |
| Base model download (Qwen3-TTS-12Hz-1.7B-Base) | ~6.8 GB |
| Raw dataset tarballs (during download, deleted after extract) | ~11.8 GB |
| Resampled audio (24kHz mono PCM16, ~141hrs) | ~24.4 GB |
| Encoded codes (jsonl) | ~2 GB |
| Checkpoints (3 recent + 1 best, ~17GB each) | ~68 GB |
| Final saves (fp32 + bf16 + fp16) | ~13.6 GB |
| llama.cpp clone (GGUF attempt) | ~2 GB |
| **Steady-state total** | **~144 GB** |
| Raw sliced audio (16kHz, before resample) — transient | +~16.3 GB |
| **Peak transient total** (during build_manifest, before cleanup) | **~160 GB** |

Against 200GB, that's **~40GB headroom at peak**. The onstart script removes
raw sliced audio (`./data/raw/slr104/*/audio`, `./data/raw/hiacc`) immediately
after `resampled/` is built and backed up to R2, bringing usage back down to
~144GB for the encode/train phases.

If `FAILED_BUILD_MANIFEST`, `FAILED_ENCODE_*`, or `FAILED_TRAIN` shows a disk
space error in `pipeline_failed.log`, increase `--disk` (e.g. to 250) — remember
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
  --disk 200 \
  --timeout-hours 30 \
  --slr104-pairs hindi-english bengali-english \
  --eval-frac 0.02
```

Flags:
- `--max-price` — max $/hr for A100 PCIe (searches `reliability>0.95` offers, picks cheapest)
- `--disk` — container disk size in GB (default 200 — see "Disk budget" above;
  steady-state ~144GB, peak transient ~160GB, ~40GB headroom). Fixed at
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
2. `create_instance()` launches it with `--disk 200` and an onstart script
   containing the full pipeline
3. The instance immediately starts running in the background (no SSH needed)
4. The orchestrator polls instance status + `/root/DONE` via `vastai copy` every
   `--poll-interval` seconds
5. On `SUCCESS`, `FAILED_*`, `TIMEOUT`, or a bad instance state (`exited`/`offline`),
   the instance is **destroyed automatically** — billing stops, and the
   container disk is destroyed with it (no separate cleanup step)

### `/root/DONE` status codes

- `SUCCESS` — all checkpoint variants + logs uploaded to R2 (see Output formats below)
- `FAILED_BOOTSTRAP` — dependency install or base model download failed
- `FAILED_DOWNLOAD_HIACC` — HiACC zenodo download/extract failed
- `FAILED_DOWNLOAD_SLR104` — OpenSLR-104 download/extract/parse failed (e.g. mirror
  unreachable, or Kaldi data dir structure inside the tarball didn't match
  `find_kaldi_dir()`'s expectations)
- `FAILED_BUILD_MANIFEST` — manifest building/resampling failed, or produced no
  eval split
- `FAILED_ENCODE_TRAIN` / `FAILED_ENCODE_EVAL` — audio-to-codes tokenization failed
- `FAILED_TRAIN` — training itself failed
- `FAILED_CONVERT_FP16` — fp32-to-fp16 conversion failed (training succeeded,
  fp32/bf16 checkpoints likely still present — check pipeline_failed.log)
- `FAILED_UPLOAD_MODEL` — upload to R2 failed after training succeeded (worst case
  to hit late — checkpoints may exist only on the now-destroyed instance)
- `ERROR_PROVISIONING` — failed before the pipeline even started, e.g.
  `create_instance()` itself failed (offer no longer available, account
  issue, etc.). No instance to destroy in this case.
- `INSTANCE_EXITED` / `INSTANCE_OFFLINE` — the instance itself reached a bad
  state (container crashed or host disconnected) before writing `/root/DONE`

On any `FAILED_*`, the orchestrator pulls `/root/pipeline.log` to
`./pipeline_failed.log` locally before destroying the instance, so you can debug
without re-renting.

**GGUF conversion failure does NOT set a FAILED status** — it's expected to likely
fail for Qwen3-TTS (see Output formats below) and the pipeline continues regardless.

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
  `pipeline.log` for "could not find wav.scp+segments+text" first.
- **`wav.scp` pipe-command entries**: the Bengali-English test split's `wav.scp`
  had 40/40 direct-path entries (no `sox ... |` pipes). Train tarballs may differ;
  `resolve_recording()`'s pipe-command branch (using `cat`/`sox` via shell) is
  exercised by the offline test but not against real pipe-command data.
- **HiACC archive structure**: `download_hiacc.py` queries the Zenodo API and
  extracts whatever it finds; `build_manifest.py`'s `find_hiacc_pairs()` assumes
  `.wav` files with sibling `.txt` transcripts. If HiACC's actual layout differs,
  this step may find 0 pairs — check `pipeline.log` for "found 0 .wav files"
  warnings early.
- **`encode_codes.py`'s `--device cuda` path and `train.py`'s audio-codes packing
  format** (`preprocess()` in `train.py`) are placeholders pending verification
  against the installed `qwen_tts` package's actual finetuning collator/tokenizer
  API — these will fail fast at `FAILED_ENCODE_TRAIN` or `FAILED_TRAIN` if
  mismatched.
- **`train_config.yaml`** assumes `per_device_train_batch_size: 4` fits in A100 80GB
  for a 1.7B full fine-tune with gradient checkpointing. If `FAILED_TRAIN` shows an
  OOM, drop to `per_device_train_batch_size: 2` and double
  `gradient_accumulation_steps` to keep the effective batch size constant.

To validate the remaining items cheaply before the full ~140hr run, run with a
single, smaller pair first:
```bash
python3 scripts/orchestrate.py --slr104-pairs bengali-english --timeout-hours 6
```
Bengali-English train alone is ~46hrs (3.9GB download) — a faster, cheaper
(~$5-10) end-to-end pass that exercises every pipeline stage and surfaces the
above issues before committing to the full Hindi-English + Bengali-English run.

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
(`manifest_raw.jsonl`, `manifest_eval_raw.jsonl`, `resampled/`,
`train_with_codes.jsonl`, `eval_with_codes.jsonl`) — a failed run after the
manifest/encode stage doesn't strictly require re-downloading or re-resampling on a
fresh rental, though `orchestrate.py` doesn't currently check for or reuse these on
a fresh run (see Known gaps).

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
| Everything (download, resample, encode, train, convert, upload) | Vast.ai A100 PCIe + 200GB container disk via `orchestrate.py` | ~$1.20/hr x actual runtime (disk is included in `dph_total` for most offers) -- container destroyed on completion, no separate cleanup |
| R2 storage (raw manifests, resampled audio, codes, 3-4 checkpoint variants) | Cloudflare R2 | ~$0.015/GB/month, negligible |

Total cost = (A100 hourly rate) x (download + resample + encode + train + convert
time). The combined dataset (~140hrs: HiACC 5.24 + SLR104 Hindi-English 89.86 +
SLR104 Bengali-English 46.11) is a known quantity, so total cost is more
predictable than the earlier IndicVoices-R-based plan. Run a small first pass
(`--slr104-pairs bengali-english --timeout-hours 6`, ~$5-10) to validate the
pipeline and get a real per-phase timing breakdown from `pipeline.log` before
committing to the full two-pair run.

---

## Known gaps / TODO

- [ ] **No resume/checkpoint-skip across runs** — if a run fails late (e.g.
  `FAILED_TRAIN` after encoding completed), rerunning `orchestrate.py` starts over
  from scratch on a fresh instance, though the manifest/codes backed up
  to R2 from the failed run could in principle be downloaded instead of
  regenerated. Not currently wired up.
- [ ] **Disk size is fixed at instance creation** — if `--disk 200` proves
  insufficient (see "Disk budget"), a failed run can't be resized; a fresh
  instance with a larger `--disk` is required. The cleanup step
  (`./data/raw/slr104/*/audio`, `./data/raw/hiacc` removal after resampling)
  reduces peak usage from ~160GB to ~144GB, but hasn't been verified against
  real data sizes.
- [ ] **`encode_codes.py`'s `--device cuda` path and `train.py`'s audio-codes
  packing format** (`preprocess()`) are placeholders — verify against the
  installed `qwen_tts` package, ideally via a small first run (see "Before your
  first real run").
- [ ] **HiACC zenodo archive structure** — `build_manifest.py`'s
  `find_hiacc_pairs()` assumes `.wav` + sibling `.txt`; verify against actual
  extracted layout.
- [ ] **OpenSLR-104 Kaldi data dir structure and `wav.scp` pipe-command handling**
  — `download_slr104.py`'s `find_kaldi_dir()` and `resolve_recording()` are based
  on the standard Kaldi recipe layout described in the MUCS 2021 baseline, but
  haven't been run against the actual extracted tarballs. A small first run (see
  "Before your first real run") surfaces mismatches here cheaply.
- [ ] **`FAILED_UPLOAD_MODEL` after successful training** is the worst-case
  failure — checkpoints exist only on the instance, which gets destroyed.
  Consider: don't destroy on this specific failure (leave instance running for
  manual recovery via `vastai copy`), or retry upload with backoff before giving
  up. Not currently differentiated from other failure modes.
- [ ] **`train_config.yaml` batch size** assumes full FT of 1.7B fits on A100 80GB
  with `per_device_train_batch_size: 4` + gradient checkpointing — unverified,
  adjust if `FAILED_TRAIN` shows OOM.
