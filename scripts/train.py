#!/usr/bin/env python3
"""
Full fine-tune of Qwen3-TTS-12Hz-1.7B-Base on pre-encoded (audio_codes +
ref_audio) JSONL data.

This follows the official finetuning loop's forward pass and loss exactly
(https://github.com/QwenLM/Qwen3-TTS/blob/main/finetuning/sft_12hz.py): an
`accelerate`-driven loop over `Qwen3TTSModel`, using the vendored
`TTSDataset`/`collate_fn` from scripts/qwen_tts_dataset.py.

Unlike the official reference, this is a general MULTI-SPEAKER full
fine-tune: per-sample speaker embeddings (from each record's own ref_audio,
via model.speaker_encoder) condition training as in the reference, but we do
NOT bake a single global speaker embedding into codec_embedding[3000]
("custom_voice"/spk_id adaptation) at save time. Checkpoints stay loadable
like the Base model, with arbitrary --ref_audio at inference.

Usage:
    python3 train.py --config configs/train_config.yaml
    python3 train.py --config configs/train_config.yaml --max-steps 50   # dry run
    python3 train.py --config configs/train_config.yaml --resume         # resume
"""
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_tts_dataset import TTSDataset

R2_PREFIX = "finetune/qwen3tts-hinglish"


def _r2_client_and_bucket():
    """Return (client, bucket) if R2 env vars are set, else (None, None)."""
    import os
    if not all(os.environ.get(v) for v in
               ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")):
        return None, None
    from upload_to_r2 import ensure_bucket, get_client
    client = get_client()
    bucket = os.environ["R2_BUCKET"]
    ensure_bucket(client, bucket)
    return client, bucket


def upload_resume_state(accel_state_dir, state_path, best_dir=None):
    """Best-effort upload of resume state (and best checkpoint) to R2, so a
    fresh instance can resume via --resume even if this one is destroyed."""
    client, bucket = _r2_client_and_bucket()
    if client is None:
        return
    from upload_to_r2 import upload_dir, upload_file
    try:
        upload_dir(client, bucket, str(accel_state_dir), f"{R2_PREFIX}/accel_state")
        upload_file(client, bucket, str(state_path), f"{R2_PREFIX}/training_state.json")
        if best_dir is not None:
            upload_dir(client, bucket, str(best_dir), f"{R2_PREFIX}/best")
    except Exception as e:
        print(f"  WARNING: failed to upload resume state to R2: {e}")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _clean_config_for_portability(config_path: Path):
    """
    Strip training-time / CUDA-specific fields from a saved config.json so the
    checkpoint loads cleanly on MPS or CPU as well as CUDA.

    - Remove attn_implementation if it's "flash_attention_2" (CUDA-only).
      Downstream loaders should pick "sdpa" or "eager" based on the runtime device.
    - Remove any device_map / quantization_config that may have been serialized
      from the training environment.
    """
    if not config_path.exists():
        print(f"  WARNING: {config_path} not found, skipping config cleanup")
        return

    with open(config_path) as f:
        cfg = json.load(f)

    changed = False
    if cfg.get("attn_implementation") == "flash_attention_2":
        del cfg["attn_implementation"]
        changed = True

    for key in ("device_map", "quantization_config"):
        if key in cfg:
            del cfg[key]
            changed = True

    if changed:
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  cleaned {config_path} for cross-device portability")


def compute_loss(model, batch):
    """Forward pass + loss, following sft_12hz.py exactly."""
    input_ids = batch["input_ids"]
    codec_ids = batch["codec_ids"]
    ref_mels = batch["ref_mels"]
    text_embedding_mask = batch["text_embedding_mask"]
    codec_embedding_mask = batch["codec_embedding_mask"]
    attention_mask = batch["attention_mask"]
    codec_0_labels = batch["codec_0_labels"]
    codec_mask = batch["codec_mask"]

    speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()

    input_text_ids = input_ids[:, :, 0]
    input_codec_ids = input_ids[:, :, 1]

    input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
    input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
    input_codec_embedding[:, 6, :] = speaker_embedding

    input_embeddings = input_text_embedding + input_codec_embedding

    for i in range(1, 16):
        codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
        codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
        input_embeddings = input_embeddings + codec_i_embedding

    outputs = model.talker(
        inputs_embeds=input_embeddings[:, :-1, :],
        attention_mask=attention_mask[:, :-1],
        labels=codec_0_labels[:, 1:],
        output_hidden_states=True,
    )

    hidden_states = outputs.hidden_states[0][-1]
    talker_hidden_states = hidden_states[codec_mask[:, :-1]]
    talker_codec_ids = codec_ids[codec_mask]

    _, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)

    return outputs.loss + 0.3 * sub_talker_loss


def save_checkpoint(unwrapped_model, init_model_path, output_dir, dtype):
    """
    Write a full HF-loadable checkpoint to output_dir: config/processor/tokenizer
    files copied from init_model_path, weights from unwrapped_model's state dict
    (cast to dtype). Unlike the official reference, no weights are dropped and no
    codec_embedding/spk_id surgery is performed -- the result loads exactly like
    the Base model, with arbitrary --ref_audio at inference.
    """
    from safetensors.torch import save_file

    output_dir = Path(output_dir)
    shutil.copytree(init_model_path, output_dir, dirs_exist_ok=True)

    # Remove any pre-existing (sharded) weight files from init_model_path so
    # only the single model.safetensors written below remains.
    for pattern in ("model.safetensors", "model.safetensors.index.json", "model-*.safetensors"):
        for f in output_dir.glob(pattern):
            f.unlink()

    state_dict = {k: v.detach().to("cpu").to(dtype) for k, v in unwrapped_model.state_dict().items()}
    save_file(state_dict, str(output_dir / "model.safetensors"))
    _clean_config_for_portability(output_dir / "config.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="for dry-run timing -- stop after this many optimizer steps total")
    ap.add_argument("--resume", action="store_true",
                    help="resume from <output_model_path>/accel_state + training_state.json if present")
    args = ap.parse_args()

    cfg = load_config(args.config)
    m = cfg["model"]
    d = cfg["data"]
    t = cfg["training"]

    set_seed(t.get("seed", 42))

    output_dir = Path(m["output_model_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        mixed_precision=t.get("mixed_precision", "bf16"),
    )

    init_model_path = m["init_model_path"]

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    print(f"Loading Qwen3TTSModel from {init_model_path} ...")
    qwen3tts = Qwen3TTSModel.from_pretrained(
        init_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=m.get("attn_implementation", "sdpa"),
    )
    config = AutoConfig.from_pretrained(init_model_path)

    print("Loading datasets ...")
    train_data = load_jsonl(d["train_jsonl"])
    train_dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=t["batch_size"],
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=t.get("dataloader_num_workers", 0),
        pin_memory=t.get("dataloader_pin_memory", False),
    )

    eval_dataloader = None
    eval_jsonl = d.get("eval_jsonl")
    if eval_jsonl and Path(eval_jsonl).exists():
        eval_data = load_jsonl(eval_jsonl)
        if eval_data:
            eval_dataset = TTSDataset(eval_data, qwen3tts.processor, config)
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=t["batch_size"],
                shuffle=False,
                collate_fn=eval_dataset.collate_fn,
            )

    optimizer = AdamW(qwen3tts.model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 0.01))

    grad_accum = t["gradient_accumulation_steps"]
    num_epochs = t["num_epochs"]
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / grad_accum)
    total_update_steps = num_update_steps_per_epoch * num_epochs
    warmup_steps = int(t.get("warmup_ratio", 0.0) * total_update_steps)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_update_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader, lr_scheduler
    )
    if eval_dataloader is not None:
        eval_dataloader = accelerator.prepare_data_loader(eval_dataloader)

    max_grad_norm = t.get("max_grad_norm", 1.0)
    logging_steps = t.get("logging_steps", 10)

    accel_state_dir = output_dir / "accel_state"
    state_path = output_dir / "training_state.json"

    start_epoch = 0
    global_step = 0
    best_eval_loss = float("inf")
    best_epoch = -1

    if args.resume and accel_state_dir.exists() and state_path.exists():
        accelerator.print(f"Resuming from {accel_state_dir} ...")
        accelerator.load_state(str(accel_state_dir))
        with open(state_path) as f:
            saved = json.load(f)
        start_epoch = saved["epoch"] + 1
        global_step = saved.get("global_step", 0)
        best_eval_loss = saved.get("best_eval_loss", float("inf"))
        best_epoch = saved.get("best_epoch", -1)
        accelerator.print(f"  resuming at epoch {start_epoch}, global_step {global_step}")

    stop = False
    for epoch in range(start_epoch, num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                loss = compute_loss(model, batch)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                if accelerator.sync_gradients:
                    lr_scheduler.step()
                    global_step += 1
                optimizer.zero_grad()

            if step % logging_steps == 0:
                accelerator.print(
                    f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} "
                    f"| LR: {lr_scheduler.get_last_lr()[0]:.2e}"
                )

            if args.max_steps and global_step >= args.max_steps:
                stop = True
                break

        # --- eval ---
        eval_loss = None
        if eval_dataloader is not None:
            model.eval()
            losses = []
            with torch.no_grad():
                for batch in eval_dataloader:
                    losses.append(compute_loss(model, batch).item())
            if losses:
                eval_loss = sum(losses) / len(losses)
                accelerator.print(f"Epoch {epoch} | eval_loss: {eval_loss:.4f}")

        # --- checkpoint / resume state (main process only) ---
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)

            if eval_loss is not None and eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_epoch = epoch
                best_dir = output_dir / "best"
                accelerator.print(f"  new best eval_loss ({best_eval_loss:.4f}) -> saving {best_dir}")
                save_checkpoint(unwrapped_model, init_model_path, str(best_dir), dtype=torch.bfloat16)

            accelerator.print(f"  saving resume state -> {accel_state_dir}")
            accelerator.save_state(str(accel_state_dir))
            with open(state_path, "w") as f:
                json.dump({
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_eval_loss": best_eval_loss,
                    "best_epoch": best_epoch,
                }, f)

            accelerator.print("  uploading resume state to R2 (if configured) ...")
            upload_resume_state(
                accel_state_dir, state_path,
                best_dir=(output_dir / "best") if best_epoch == epoch else None,
            )

        accelerator.wait_for_everyone()

        if stop:
            break

    # --- final outputs ---
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)

        master_dir = output_dir / "final_fp32"
        print(f"Saving fp32 master to {master_dir} ...")
        save_checkpoint(unwrapped_model, init_model_path, str(master_dir), dtype=torch.float32)

        bf16_dir = output_dir / "final_bf16"
        print(f"Saving bf16 copy to {bf16_dir} ...")
        save_checkpoint(unwrapped_model, init_model_path, str(bf16_dir), dtype=torch.bfloat16)

        # Keep "final" as an alias for the fp32 master (back-compat with existing
        # upload/runbook references to checkpoints/final)
        final_dir = output_dir / "final"
        if final_dir.exists():
            shutil.rmtree(final_dir)
        shutil.copytree(master_dir, final_dir)

        print("Done. Outputs:")
        print(f"  fp32 master : {master_dir}  (source of truth -- use for all conversions)")
        print(f"  bf16 copy   : {bf16_dir}    (matches training dtype, ready for CUDA)")
        print(f"  final       : {final_dir}   (alias of fp32 master)")
        if best_epoch >= 0:
            print(f"  best        : {output_dir / 'best'}   (epoch {best_epoch}, eval_loss {best_eval_loss:.4f})")
        print("")
        print("Next: run scripts/convert_model.py against the fp32 master to produce")
        print("fp16 (MPS-friendly) and GGUF variants. See docs/RUNBOOK.md Phase 7.")
        print("")
        print("Upload all variants to R2, e.g.:")
        print(f"  python3 scripts/upload_to_r2.py --file {master_dir} "
              f"--bucket $R2_BUCKET --key finetune/qwen3tts-hinglish/final_fp32 --recursive")
        print(f"  python3 scripts/upload_to_r2.py --file {bf16_dir} "
              f"--bucket $R2_BUCKET --key finetune/qwen3tts-hinglish/final_bf16 --recursive")


if __name__ == "__main__":
    main()
