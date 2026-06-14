#!/usr/bin/env python3
"""
Full fine-tune Qwen3-TTS-12Hz-1.7B-Base on pre-encoded (audio_codes) JSONL data.

Usage:
    python3 train.py --config configs/train_config.yaml
    python3 train.py --config configs/train_config.yaml --max-steps 50   # dry run

NOTE: This wraps the official Qwen3-TTS fine-tuning entry point
(https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning). Verify the exact
CLI/import path against the installed `qwen-tts` version before the real run —
the official repo's finetuning script signature may differ slightly between
releases. This script assumes a HF Trainer-compatible interface; adjust the
`build_trainer()` function if the installed version exposes a different API.
"""
import argparse
import json
import os
import sys
import yaml
from pathlib import Path

import torch


def _clean_config_for_portability(config_path: Path):
    """
    Strip training-time / CUDA-specific fields from a saved config.json so the
    checkpoint loads cleanly on MPS or CPU as well as CUDA.

    Specifically:
    - Remove attn_implementation if it's set to "flash_attention_2" (CUDA-only).
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


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_trainer(cfg, max_steps=None):
    """
    Construct a Trainer for Qwen3-TTS full fine-tuning.

    This follows the pattern from the official finetuning README: load base model
    + tokenizer config, wrap pre-encoded audio_codes + text into the model's
    expected input format, and train with HF Trainer / Accelerate.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from datasets import load_dataset

    model_path = cfg["model"]["base_model_path"]
    print(f"Loading base model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if cfg["training"].get("bf16") else torch.float32,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if cfg["training"].get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()

    print("Loading datasets ...")
    train_ds = load_dataset("json", data_files=cfg["data"]["train_file"], split="train")
    eval_ds = load_dataset("json", data_files=cfg["data"]["eval_file"], split="train")

    def preprocess(example):
        # audio_codes: nested list [num_codebooks][T] -> flatten/format per model's
        # expected input_ids structure. The exact packing scheme (interleaving text
        # tokens + audio codebook tokens) depends on Qwen3-TTS's tokenizer convention —
        # consult the installed qwen_tts package's finetuning data_collator for the
        # authoritative format and adjust this function accordingly.
        text = example["text"]
        codes = example["audio_codes"]
        text_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        # Placeholder: flatten codebooks sequentially. REPLACE with the official
        # packing logic from qwen_tts.finetuning before running for real.
        flat_codes = [c for layer in codes for c in layer]
        input_ids = text_ids + flat_codes
        return {"input_ids": input_ids, "labels": input_ids}

    print("Preprocessing (this should be fast — codes are pre-encoded) ...")
    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(preprocess, remove_columns=eval_ds.column_names)

    from transformers import DataCollatorForLanguageModeling
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    t = cfg["training"]
    training_args = TrainingArguments(
        output_dir=cfg["model"]["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        max_grad_norm=t.get("max_grad_norm", 1.0),
        bf16=t.get("bf16", False),
        logging_steps=t["logging_steps"],
        eval_strategy=t.get("eval_strategy", "steps"),
        eval_steps=t["eval_steps"],
        save_steps=t["save_steps"],
        save_total_limit=t.get("save_total_limit"),
        # Best-model tracking: save the checkpoint with lowest eval loss, not
        # just the last one. Primary defense against overfitting with a small
        # dataset-to-model ratio (~0.1x tokens/params per epoch).
        load_best_model_at_end=t.get("load_best_model_at_end", False),
        metric_for_best_model=t.get("metric_for_best_model", "eval_loss"),
        greater_is_better=t.get("greater_is_better", False),
        # Data loading
        dataloader_num_workers=t.get("dataloader_num_workers", 0),
        dataloader_pin_memory=t.get("dataloader_pin_memory", False),
        # Reproducibility
        seed=t.get("seed", 42),
        max_steps=max_steps if max_steps else -1,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    return trainer, model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="for dry-run timing — overrides epoch-based training")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script is intended for the A100 instance.", file=sys.stderr)
        sys.exit(1)

    trainer, model, tokenizer = build_trainer(cfg, max_steps=args.max_steps)

    print("Starting training ...")
    trainer.train()

    output_dir = Path(cfg["model"]["output_dir"])

    # --- fp32 master copy (source of truth for downstream conversions) ---
    master_dir = output_dir / "final_fp32"
    print(f"Saving fp32 master to {master_dir} ...")
    model_fp32 = model.to(torch.float32)
    trainer.model = model_fp32
    trainer.save_model(str(master_dir))
    tokenizer.save_pretrained(str(master_dir))
    _clean_config_for_portability(master_dir / "config.json")

    # --- bf16 copy matching training dtype, for direct CUDA reuse without
    #     a separate downcast step ---
    bf16_dir = output_dir / "final_bf16"
    print(f"Saving bf16 copy to {bf16_dir} ...")
    model_bf16 = model_fp32.to(torch.bfloat16)
    trainer.model = model_bf16
    trainer.save_model(str(bf16_dir))
    tokenizer.save_pretrained(str(bf16_dir))
    _clean_config_for_portability(bf16_dir / "config.json")

    # Keep "final" as an alias for the fp32 master (back-compat with existing
    # upload/runbook references to checkpoints/final)
    final_dir = output_dir / "final"
    if final_dir.exists():
        import shutil
        shutil.rmtree(final_dir)
    import shutil
    shutil.copytree(master_dir, final_dir)

    print("Done. Outputs:")
    print(f"  fp32 master : {master_dir}  (source of truth — use for all conversions)")
    print(f"  bf16 copy   : {bf16_dir}    (matches training dtype, ready for CUDA)")
    print(f"  final       : {final_dir}   (alias of fp32 master)")
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
