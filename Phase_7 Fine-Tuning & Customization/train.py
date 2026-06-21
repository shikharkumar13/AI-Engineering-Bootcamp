"""
train.py — QLoRA Fine-tuning with Unsloth

DESIGNED TO RUN IN GOOGLE COLAB (free T4 GPU is sufficient).

This script is NOT meant to run on a CPU-only machine — Unsloth and
bitsandbytes both require a CUDA GPU. To use this:

  1. Open a new Google Colab notebook (colab.research.google.com)
  2. Runtime > Change runtime type > select "T4 GPU"
  3. In the first cell:
       !pip install unsloth datasets trl peft accelerate bitsandbytes
  4. Upload this file (and dataset_prep.py) or paste the contents into cells
  5. Run: !python train.py
     (or paste the contents of main() into a cell and run directly)

What this script does:
  1. Loads Llama-3-8B in 4-bit (QLoRA) via Unsloth
  2. Attaches LoRA adapters to attention + MLP projections
  3. Builds the domain dataset from dataset_prep.py
  4. Fine-tunes with SFTTrainer
  5. Saves the adapter, and optionally merges + pushes to Hugging Face Hub
"""

import os
import torch
from dotenv import load_dotenv

load_dotenv()


# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_NAME       = "unsloth/llama-3-8b-Instruct-bnb-4bit"  # Unsloth's pre-quantized model
MAX_SEQ_LENGTH   = 2048
OUTPUT_DIR       = "llama3_support_assistant_lora"
HF_REPO_NAME     = os.getenv("HF_REPO_NAME", "your-username/llama3-support-assistant")

LORA_R           = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0

NUM_EPOCHS       = 3
BATCH_SIZE       = 2
GRAD_ACCUM_STEPS = 4
LEARNING_RATE    = 2e-4


def check_gpu():
    """Verify a CUDA GPU is available before attempting to load a 4-bit model."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. This script requires a GPU.\n"
            "If you're in Google Colab: Runtime > Change runtime type > T4 GPU.\n"
            "Unsloth and bitsandbytes cannot run on CPU-only environments."
        )
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  ✓ GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    if vram_gb < 14:
        print(f"  ⚠ Warning: less than 14GB VRAM detected. Consider reducing "
              f"MAX_SEQ_LENGTH or BATCH_SIZE if you hit out-of-memory errors.")


def load_model_and_tokenizer():
    """Load the base model in 4-bit and attach LoRA adapters via Unsloth."""
    from unsloth import FastLanguageModel

    print(f"\n  Loading {MODEL_NAME} in 4-bit (QLoRA)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,          # auto-detect best dtype for the GPU
        load_in_4bit=True,
    )
    print(f"  ✓ Base model loaded (4-bit quantized)")

    print(f"\n  Attaching LoRA adapters (r={LORA_R}, alpha={LORA_ALPHA})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    model.print_trainable_parameters()
    return model, tokenizer


def prepare_training_data(tokenizer):
    """Build and format the domain dataset for training."""
    from dataset_prep import build_dataset, format_for_chat_template

    print(f"\n  Building domain dataset...")
    data = build_dataset(target_size=40)

    print(f"  Formatting with chat template...")
    train_dataset = data["train"].map(
        lambda ex: format_for_chat_template(ex, tokenizer)
    )
    eval_dataset = data["eval"].map(
        lambda ex: format_for_chat_template(ex, tokenizer)
    )

    print(f"  ✓ Train: {len(train_dataset)} examples, Eval: {len(eval_dataset)} examples")
    print(f"\n  Sample formatted training text:")
    print(f"  {train_dataset[0]['text'][:400]}...")

    return train_dataset, eval_dataset


def train(model, tokenizer, train_dataset):
    """Run the SFTTrainer training loop."""
    from trl import SFTTrainer
    from transformers import TrainingArguments

    print(f"\n  Starting training: {NUM_EPOCHS} epochs, "
          f"effective batch size {BATCH_SIZE * GRAD_ACCUM_STEPS}...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
            warmup_steps=5,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=OUTPUT_DIR,
            save_strategy="epoch",
            report_to="none",   # set to "wandb" if you want experiment tracking
        ),
    )

    stats = trainer.train()

    print(f"\n  ✓ Training complete.")
    print(f"  Final train loss: {stats.metrics.get('train_loss', 'N/A')}")
    print(f"  Training time:    {stats.metrics.get('train_runtime', 0):.1f}s")

    return trainer, stats


def save_and_export(model, tokenizer, push_to_hub: bool = False):
    """Save the adapter locally, and optionally merge + push to Hugging Face Hub."""

    print(f"\n  Saving LoRA adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  ✓ Adapter saved (small, ~50-200MB)")

    if push_to_hub:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("  ⚠ HF_TOKEN not set in .env — skipping Hub push.")
            print("  Get a token at https://huggingface.co/settings/tokens")
            return

        print(f"\n  Merging adapter into base model and pushing to {HF_REPO_NAME}...")
        model.push_to_hub_merged(
            HF_REPO_NAME,
            tokenizer,
            save_method="merged_16bit",
            token=hf_token,
        )
        print(f"  ✓ Pushed to https://huggingface.co/{HF_REPO_NAME}")
    else:
        print(f"\n  Skipping Hub push (push_to_hub=False).")
        print(f"  To merge locally for deployment, run:")
        print(f"    model.save_pretrained_merged('{OUTPUT_DIR}_merged', tokenizer, "
              f"save_method='merged_16bit')")


def main():
    print("=" * 62)
    print("  Phase 07 — QLoRA Fine-tuning with Unsloth")
    print("=" * 62)

    check_gpu()
    model, tokenizer = load_model_and_tokenizer()
    train_dataset, eval_dataset = prepare_training_data(tokenizer)
    trainer, stats = train(model, tokenizer, train_dataset)

    # Save eval dataset reference for evaluate.py to use
    eval_dataset.save_to_disk("eval_dataset_cache")

    save_and_export(model, tokenizer, push_to_hub=False)

    print("\n" + "=" * 62)
    print("  ✅ Fine-tuning complete.")
    print(f"  Adapter saved to: {OUTPUT_DIR}")
    print("  Next: run evaluate.py to compare against the base model.")
    print("=" * 62)


if __name__ == "__main__":
    main()
