# qwen3tts-hinglish

Full fine-tune of [`Qwen/Qwen3-TTS-12Hz-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
for Hinglish (code-switched Hindi-English / Bengali-English) speech, on a
combined manifest of:

- **HiACC** — Hinglish adult + children code-switched corpus (~5.24 hrs)
- **OpenSLR-104** (MUCS 2021 subtask2) — Hindi-English (89.86 hrs) and
  Bengali-English (46.11 hrs) code-switched speech, CC BY-SA 4.0

~140 hours total.

## Quick start

The entire pipeline (download datasets, build manifest, encode codes, train,
convert checkpoints, upload to R2) runs unattended in a single Vast.ai A100
rental, driven from your local machine by `scripts/orchestrate.py`:

```bash
pip install -r requirements.txt
pip install vastai && vastai set api-key <your-vast-api-key>

export R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_BUCKET=...
export REPO_GIT_URL=https://github.com/yourname/qwen3tts-hinglish.git

python3 scripts/orchestrate.py --dry-run --max-price 1.20   # check pricing first
python3 scripts/orchestrate.py --max-price 1.20 --disk 250 --timeout-hours 30
```

See **[docs/RUNBOOK.md](docs/RUNBOOK.md)** for the full setup, pipeline
stages, disk budget, output formats, troubleshooting, and known gaps.

## Repo layout

| Path | Purpose |
|---|---|
| `scripts/orchestrate.py` | Provisions the Vast.ai instance, runs the full pipeline via an onstart script, polls for completion, destroys the instance |
| `scripts/bootstrap_vastai.sh` | Instance setup: GPU/disk checks, Python deps, base model download |
| `scripts/download_hiacc.py`, `scripts/download_slr104.py` | Dataset downloaders |
| `scripts/build_manifest.py` | Resamples audio, builds unified train/eval manifest |
| `scripts/encode_codes.py` | Encodes audio to discrete codes via the Qwen3-TTS tokenizer |
| `scripts/train.py` | Full fine-tune, produces fp32/bf16/fp16 checkpoints |
| `scripts/convert_model.py` | Converts checkpoint dtypes / attempts GGUF export |
| `scripts/upload_to_r2.py` | Upload/download artifacts to/from Cloudflare R2 |
| `configs/train_config.yaml` | Training hyperparameters |
| `tests/test_download_slr104.py` | Local (M4/CPU) tests for the OpenSLR-104 downloader |
| `docs/RUNBOOK.md` | Full operational runbook |

## License

MIT — see [LICENSE](LICENSE).
