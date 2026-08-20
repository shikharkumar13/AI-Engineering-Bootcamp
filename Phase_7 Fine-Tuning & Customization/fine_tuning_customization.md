# Phase 07 — Fine-tuning & Customization

> **Prerequisites:** Phases 01-06 complete, plus your existing knowledge of neural
> network training (backprop, gradient descent, loss functions).  
> **What you'll learn:** When to fine-tune vs prompt vs RAG; the Hugging Face ecosystem;
> the math and intuition behind LoRA/QLoRA; fast training with Unsloth; evaluating a
> fine-tuned model against its base.  
> **Project:** Fine-tune Llama 3 8B on a niche domain using QLoRA + Unsloth, evaluate
> against the base model, push to Hugging Face Hub, and serve via API.

> **Hardware note:** Unlike Phases 01-06, this phase requires a GPU. A free Google
> Colab T4 GPU (16GB VRAM) is sufficient for everything in this phase thanks to QLoRA's
> memory efficiency. The project code is written to run directly in Colab.

---

## Table of Contents

1. [The Big Picture — Fine-tune vs RAG vs Prompting](#1-the-big-picture--fine-tune-vs-rag-vs-prompting)
2. [The Hugging Face Ecosystem](#2-the-hugging-face-ecosystem)
3. [LoRA — Low-Rank Adaptation](#3-lora--low-rank-adaptation)
4. [QLoRA — Quantized LoRA](#4-qlora--quantized-lora)
5. [Fast Training with Unsloth](#5-fast-training-with-unsloth)
6. [Evaluation — Did Fine-tuning Actually Help?](#6-evaluation--did-fine-tuning-actually-help)
7. [Key Takeaways](#7-key-takeaways)
8. [Practice Exercises](#8-practice-exercises)

---

## 1. The Big Picture — Fine-tune vs RAG vs Prompting

### 1.1 Three ways to customize an LLM's behavior

By this point you have two tools for adapting an LLM to your needs: prompting (Phase 02)
and RAG (Phase 04). Fine-tuning is the third, and it works completely differently from
the other two.

```
PROMPTING        — Change what you ask. Model weights are untouched.
RAG               — Change what the model can see. Model weights are untouched.
FINE-TUNING       — Change the model's weights themselves via further training.
```

**From your ML background:** Fine-tuning is exactly what it sounds like to you already:
continuing the training process on a pretrained model, except now the "pretrained
model" is a full LLM and the new training data is your domain-specific examples. The
forward pass, loss computation, and backward pass are the same mechanisms you already
know. What's new in this phase is making that training computationally feasible on
consumer hardware, and knowing when this approach is actually the right choice.

---

### 1.2 The decision framework

This is the single most important judgment call in this phase. Fine-tuning is the most
powerful but also the most expensive, slowest-to-iterate, and easiest-to-misuse option.

| Need | Best solution | Why |
|---|---|---|
| Model needs facts not in its training data | RAG | Facts change; weights shouldn't need to change every time data updates |
| Model needs to follow a specific instruction format consistently | Prompting (few-shot) | Cheapest, fastest to iterate, no training needed |
| Model needs a personality/tone applied consistently | System prompt | Same reason — no training needed |
| Model needs to produce structured output reliably | Function calling / instructor | Phase 02 already solves this without training |
| Model needs to learn a new SKILL or STYLE deeply ingrained in outputs | Fine-tuning | The behavior must be "baked in," not retrieved or instructed each time |
| Model needs to be much smaller/faster while keeping task-specific quality | Fine-tuning (often combined with distillation) | A small fine-tuned model can beat a large general model on a narrow task |
| Model needs to understand a highly specialized vocabulary/jargon deeply | Fine-tuning (sometimes combined with RAG) | Domain vocabulary saturation is hard to achieve with few-shot alone |

**The most common mistake newcomers make:** reaching for fine-tuning to add knowledge
("teach the model about our company's products"). This is almost always wrong; that's
what RAG is for. Fine-tuning teaches a model **how to behave**, not **what facts to
know**. A fine-tuned model without RAG will still hallucinate facts; it will just
hallucinate them in a more consistent style.

```
Fine-tuning teaches HOW to respond (format, tone, reasoning style, task skill)
RAG provides WHAT to respond with (facts, current information, your documents)

Best production systems often combine both:
  fine-tuned model (knows how to format medical reports concisely)
  + RAG (retrieves the specific patient data to put in that report)
```

---

### 1.3 A concrete decision example

Imagine you want an AI assistant that answers legal questions about your company's
specific contracts.

- **Wrong approach:** Fine-tune a model on your contracts so it "knows" them.
  Contracts change. Fine-tuning teaches patterns, not facts: the model will produce
  plausible-sounding but potentially incorrect contract details (hallucination risk is
  not reduced by fine-tuning on facts).

- **Right approach:** Use RAG to retrieve the actual relevant contract clauses for each
  question (Phase 04). Optionally, fine-tune the model on examples of how a legal
  assistant should *phrase* answers (formal tone, always citing the specific clause
  number, always flagging ambiguity), because that is a *behavioral* pattern, not a
  fact, and few-shot prompting alone might not consistently enforce it across thousands
  of varied questions.

---

### 1.4 Full fine-tuning vs parameter-efficient fine-tuning (PEFT)

Once you've decided fine-tuning is the right tool, the next question is *how much* of
the model to update.

**Full fine-tuning:** Update every parameter in the model. For an 8B parameter model,
this means optimizing 8 billion weights, which requires storing gradients and optimizer
states for all of them: roughly 4x the model's base memory footprint (weights +
gradients + two Adam optimizer moments). An 8B model in fp16 needs ~16GB just for
weights; full fine-tuning needs ~60-80GB of VRAM. This requires multiple high-end GPUs.

**Parameter-Efficient Fine-Tuning (PEFT):** Freeze the original model weights entirely
and train only a small number of additional parameters. LoRA (Section 3) is the
dominant PEFT technique. It can reduce trainable parameters by 10,000x while retaining
most of full fine-tuning's quality benefit for many tasks.

```
Full fine-tuning (8B model):    ~8,000,000,000 trainable parameters
LoRA fine-tuning (typical):     ~10,000,000 trainable parameters (0.1-1% of total)
```

This is the difference between needing a multi-GPU server and being able to fine-tune
on a free Colab T4. PEFT is what makes this phase accessible without enterprise
infrastructure.

---

## 2. The Hugging Face Ecosystem

### 2.1 Why Hugging Face

Hugging Face is the central hub for open-source ML models, datasets, and training
tooling. Three libraries matter for this phase:

- **`transformers`:** load and run any open-source model with a unified API
- **`datasets`:** load, process, and stream training datasets efficiently
- **`Hub`:** host and share models/datasets (like GitHub, but for ML artifacts)

```bash
pip install transformers datasets huggingface_hub
```

```python
from huggingface_hub import login
login(token="hf_...")  # get a free token at huggingface.co/settings/tokens
```

---

### 2.2 Loading a model with `transformers`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# Note: Llama models require accepting a license on Hugging Face first —
# visit the model page and click "Agree" before you can download it.

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,   # half precision — half the memory, minimal quality loss
    device_map="auto",            # automatically place layers on available GPU(s)
)

# Tokenize input — same tokenization concepts from Phase 01
prompt = "Explain what fine-tuning means in one sentence."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

**`AutoModelForCausalLM` / `AutoTokenizer`:** The "Auto" classes inspect the model's
config and automatically load the right architecture class and tokenizer: you don't
need to know whether a model is LLaMA, Mistral, or Qwen architecture; the same loading
code works for all of them.

---

### 2.3 The `datasets` library

Training data for fine-tuning needs to be loaded, formatted, and tokenized efficiently,
often without fitting entirely in RAM. The `datasets` library handles this with
memory-mapped, streaming-capable dataset objects.

```python
from datasets import load_dataset, Dataset

# Load a public dataset from the Hub
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
print(dataset)
# Dataset({features: ['instruction', 'context', 'response', 'category'], num_rows: 15011})

print(dataset[0])
# {'instruction': 'When did Virgin Australia start operating?',
#  'context': 'Virgin Australia, the trading name of Virgin Australia Airlines...',
#  'response': 'Virgin Australia commenced services on 31 August 2000...',
#  'category': 'closed_qa'}

# Build a dataset from your own data (a list of dicts)
my_data = [
    {"instruction": "Summarize this contract clause", "response": "..."},
    {"instruction": "Extract key terms from this agreement", "response": "..."},
]
my_dataset = Dataset.from_list(my_data)

# Split into train/validation
split = my_dataset.train_test_split(test_size=0.1, seed=42)
train_data = split["train"]
eval_data  = split["test"]
```

---

### 2.4 Formatting data into a chat template

Modern instruction-tuned models expect input in a specific chat format with special
tokens marking turns. Using the tokenizer's built-in chat template ensures your training
data matches the format the model was originally trained on.

```python
# The tokenizer knows the model's expected chat format
messages = [
    {"role": "system", "content": "You are a helpful legal assistant."},
    {"role": "user", "content": "What does 'force majeure' mean?"},
    {"role": "assistant", "content": "Force majeure refers to unforeseeable "
                                       "circumstances that prevent a party from "
                                       "fulfilling a contract..."},
]

formatted = tokenizer.apply_chat_template(messages, tokenize=False)
print(formatted)
# → "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a
#    helpful legal assistant...<|eot_id|><|start_header_id|>user<|end_header_id|>..."

# For training, you typically format every example this way before tokenizing
def format_example(example: dict) -> dict:
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
    return example

formatted_dataset = my_dataset.map(format_example)
```

**Why this matters:** Each model family has a different chat template (LLaMA 3 uses
`<|start_header_id|>`, Mistral uses `[INST]`, etc.). If your training data doesn't match
the format the model expects, you either confuse the model during training or it learns
to ignore the special tokens it was pretrained to respect, degrading instruction-
following ability you'd otherwise inherit for free from the base model.

---

### 2.5 Pushing to the Hugging Face Hub

```python
# After fine-tuning, push your model so others (or your own production system) can load it
model.push_to_hub("your-username/llama3-legal-assistant")
tokenizer.push_to_hub("your-username/llama3-legal-assistant")

# Anyone (with appropriate permissions) can then load it exactly like a base model:
# model = AutoModelForCausalLM.from_pretrained("your-username/llama3-legal-assistant")

# Push a dataset too, if you want to share or version it
my_dataset.push_to_hub("your-username/legal-qa-dataset")
```

---

## 3. LoRA — Low-Rank Adaptation

### 3.1 The core idea

LoRA (Hu et al., 2021) is based on a key empirical observation: when you fine-tune a
large model, the *change* in weights (the difference between the fine-tuned weights and
the original weights) tends to have low "intrinsic rank," meaning it can be well
approximated by a much smaller matrix decomposition, even though the original weight
matrix is huge.

**From your linear algebra background:** Any matrix `ΔW` of shape `(d, k)` can be
approximated as the product of two smaller matrices: `ΔW ≈ B × A`, where `B` has shape
`(d, r)` and `A` has shape `(r, k)`, with `r` (the rank) much smaller than `d` or `k`.
If `d = k = 4096` (a typical hidden dimension) and you choose `r = 8`, you go from
`4096 × 4096 = 16.7M` parameters to `(4096×8) + (8×4096) = 65,536` parameters: a 256x
reduction for that one weight matrix.

```
Original weight update (full fine-tuning):
    W_new = W_original + ΔW                    ΔW is (4096 x 4096) = 16.7M params

LoRA weight update:
    W_new = W_original + B·A                    B is (4096 x 8), A is (8 x 4096)
                                                  = 65,536 params total (256x fewer)
```

---

### 3.2 How LoRA works during training and inference

```
                    ┌─────────────────┐
   Input x ────────►│  Frozen W       │────┐
                     │  (original      │    │
                     │   weights,      │    ▼
                     │   NOT trained)  │   ( + )───► Output
                     └─────────────────┘    ▲
                                             │
                     ┌────┐    ┌────┐       │
   Input x ─────────►│  A │───►│  B │───────┘
                     └────┘    └────┘
                     (trained)  (trained)
                     rank r       rank r
```

The original weight matrix `W` is completely frozen: no gradients flow through it, no
gradient computation needed, no optimizer state needed for it. Only the small matrices
`A` and `B` are trained. During the forward pass, the output is the original frozen
computation PLUS the low-rank adaptation:

```
output = W·x + (B·A)·x = (W + B·A)·x
```

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """
    A simplified illustration of LoRA's mechanism — this is conceptually
    what the `peft` library implements for you under the hood.
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: int = 16):
        super().__init__()
        # The original frozen layer (in practice, this already exists in the model)
        self.frozen_weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)

        # The LoRA matrices — these are what actually get trained
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        # B is initialized to zero so that at the START of training,
        # B·A = 0 and the model behaves EXACTLY like the original pretrained model.
        # This is crucial — training starts from the known-good base behavior.

        self.scaling = alpha / rank   # controls how much the LoRA update affects output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frozen_output = x @ self.frozen_weight.T
        lora_output   = x @ self.lora_A.T @ self.lora_B.T
        return frozen_output + self.scaling * lora_output


# Demonstration: parameter count comparison
in_features, out_features, rank = 4096, 4096, 8

full_params = in_features * out_features
lora_params = (rank * in_features) + (out_features * rank)

print(f"Full fine-tuning params for this layer: {full_params:,}")
print(f"LoRA params for this layer (r={rank}):   {lora_params:,}")
print(f"Reduction: {full_params / lora_params:.0f}x fewer trainable parameters")
```

---

### 3.3 Key hyperparameters: rank (r) and alpha

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,                    # rank — higher = more expressive, more parameters, more memory
    lora_alpha=32,           # scaling factor — typically set to 2x the rank as a starting point
    lora_dropout=0.05,       # dropout on the LoRA layers, helps prevent overfitting
    target_modules=[         # which weight matrices in the model get LoRA adapters
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention projections
        "gate_proj", "up_proj", "down_proj",       # MLP projections
    ],
    bias="none",             # don't train bias terms
    task_type="CAUSAL_LM",   # this is a causal language model (decoder-only, like GPT)
)
```

**Choosing `r` (rank):**

| Rank | Trainable params (typical 8B model) | Use case |
|---|---|---|
| 4-8 | ~5-10M | Simple style/tone adaptation, small datasets |
| 16-32 | ~20-40M | Standard instruction fine-tuning (most common choice) |
| 64-128 | ~80-160M | Complex new skills, larger datasets, harder tasks |

Higher rank gives the adapter more expressive capacity but increases memory use, risk
of overfitting on small datasets, and training time. Start with `r=16` and adjust based
on validation loss behavior.

**Choosing `lora_alpha`:** The `scaling = alpha / r` factor controls the magnitude of
the LoRA update relative to the frozen weights. The community convention of setting
`alpha = 2 × r` is a reasonable default; some recipes use `alpha = r` for more
conservative updates.

**Choosing `target_modules`:** Originally, LoRA papers only adapted the attention
query/value projections. Modern practice (especially with Unsloth, see Section 5) often
adapts all linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
`down_proj`) for the best quality, since the marginal memory cost is still small relative
to full fine-tuning.

---

### 3.4 Why B is initialized to zero

This deserves emphasis because it's an elegant and important design choice. At the start
of training:

```
B = 0  (all zeros)
→ B·A = 0  (regardless of what A contains)
→ output = W·x + 0 = W·x   (exactly the original model's behavior)
```

This means training starts from a model that behaves identically to the pretrained
base model. As gradients update `A` and `B`, the adaptation gradually shifts behavior
away from the base model's defaults; there's no "cold start" instability where the
model briefly behaves erratically before learning useful patterns, unlike if both
matrices were randomly initialized.

---

### 3.5 Merging LoRA weights for deployment

After training, you can keep the LoRA adapter separate (load base model + adapter at
inference time) or merge the adapter into the base weights for a single, self-contained
model with no inference-time overhead.

```python
from peft import PeftModel

# Load base model + trained LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
model_with_adapter = PeftModel.from_pretrained(base_model, "path/to/lora/adapter")

# Merge: bakes the LoRA update directly into the original weights
# W_merged = W_original + scaling * (B @ A)
merged_model = model_with_adapter.merge_and_unload()

# merged_model is now a standard model with no LoRA overhead at inference time
merged_model.save_pretrained("path/to/merged/model")
```

**When to merge vs keep separate:**

| Keep adapter separate | Merge into base |
|---|---|
| Want to swap between multiple task-specific adapters on one base model | Deploying a single fixed fine-tuned model |
| Storage efficiency matters (adapters are tiny, ~10-100MB vs full model GBs) | Want zero LoRA computational overhead at inference |
| Experimenting with different adapter combinations | Production deployment of one final model |

---

## 4. QLoRA — Quantized LoRA

### 4.1 The memory problem QLoRA solves

LoRA dramatically reduces *trainable* parameters, but the frozen base model still needs
to be loaded into memory in full precision to compute the forward and backward pass. An
8B model in bfloat16 (2 bytes per parameter) needs `8 × 10^9 × 2 bytes = 16GB` just to
hold the weights, before accounting for activations, gradients of the LoRA parameters,
and optimizer states.

QLoRA (Dettmers et al., 2023) solves this by **quantizing** the frozen base model to
4-bit precision, while keeping the small trainable LoRA parameters in higher precision
(typically bfloat16). This is the critical innovation that makes fine-tuning an 8B
model feasible on a single consumer GPU.

```
Standard LoRA:  Base model in bf16 (16GB for 8B model) + LoRA adapters in bf16
QLoRA:          Base model in 4-bit (~5GB for 8B model) + LoRA adapters in bf16
```

---

### 4.2 What 4-bit quantization actually means

Quantization reduces the number of bits used to represent each weight value, trading
some numerical precision for a massive reduction in memory footprint.

```
fp32 (32-bit float):  full precision, 4 bytes/param  — rarely used for inference
fp16/bf16 (16-bit):   half precision, 2 bytes/param  — standard for modern LLMs
int8 (8-bit):         1 byte/param  — common quantization level
NF4 (4-bit, QLoRA):   0.5 bytes/param — QLoRA's innovation
```

QLoRA specifically uses **NF4 (4-bit NormalFloat)**, a quantization scheme designed
around the empirical observation that pretrained neural network weights tend to follow
a roughly normal (Gaussian) distribution. NF4 allocates its limited bit budget to
represent values more precisely where the weight distribution has more mass (near
zero), rather than spacing quantization levels uniformly. This preserves more
information than naive uniform 4-bit quantization for the same bit budget.

```python
from transformers import BitsAndBytesConfig
import torch

# This config tells transformers to load the model in 4-bit quantized form
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",                  # NF4 quantization (QLoRA's scheme)
    bnb_4bit_compute_dtype=torch.bfloat16,       # computation happens in bf16, weights stored in 4-bit
    bnb_4bit_use_double_quant=True,              # quantize the quantization constants too — saves a bit more memory
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# Check actual memory footprint
print(f"Memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
# An 8B model that would need ~16GB in bf16 now needs ~5-6GB in 4-bit
```

---

### 4.3 Why compute happens in bf16 despite 4-bit storage

A critical detail: QLoRA stores weights in 4-bit but **de-quantizes them to bf16 on the
fly** for the actual matrix multiplication during the forward/backward pass. The 4-bit
representation is purely for storage; compute happens at higher precision to avoid
numerical instability during training. This is why the `bnb_4bit_compute_dtype` parameter
exists: it controls the precision used during the actual arithmetic, separate from the
storage precision.

```
Storage:    weight stored as 4-bit NF4 value (saves memory)
              ↓ dequantize on the fly
Compute:    weight cast to bf16 for the matmul (preserves numerical stability)
              ↓
Result:     output computed, gradients flow only to LoRA A/B matrices (bf16)
              the base weights are never updated, so their 4-bit storage is fine
```

This is what makes QLoRA's quality nearly match full-precision LoRA despite the
aggressive 4-bit compression: the model never actually computes anything in 4-bit; it
only **stores** the frozen weights that way.

---

### 4.4 Combining QLoRA with PEFT

```python
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# Step 1: load the quantized base model (Section 4.2)
model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto"
)

# Step 2: prepare the quantized model for training
# This handles details like enabling gradient checkpointing and ensuring
# certain layers (like layer norms) stay in higher precision for stability
model = prepare_model_for_kbit_training(model)

# Step 3: attach LoRA adapters (Section 3.3)
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Inspect trainable vs frozen parameter counts
model.print_trainable_parameters()
# → "trainable params: 41,943,040 || all params: 8,072,204,288 || trainable%: 0.5197"
```

This combination (4-bit quantized frozen base + bf16 trainable LoRA adapters) is what
allows an 8B (or even 13B-70B with more aggressive settings) model to be fine-tuned on a
single consumer GPU with 16-24GB of VRAM, hardware that would be utterly insufficient
for full fine-tuning of the same model.

---

## 5. Fast Training with Unsloth

### 5.1 What Unsloth optimizes

Unsloth is a library that reimplements the core QLoRA training operations (attention,
the LoRA forward/backward pass, gradient checkpointing) using custom, hand-optimized
Triton/CUDA kernels instead of relying purely on PyTorch's default operators. The
practical result: 2x faster training and roughly 50-70% less VRAM usage compared to
standard `transformers` + `peft` + `bitsandbytes`, with mathematically identical
training results (no accuracy tradeoff).

```bash
pip install unsloth
```

---

### 5.2 Loading a model with Unsloth

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048   # context length for training examples

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",  # Unsloth's pre-quantized version
    max_seq_length=max_seq_length,
    dtype=None,            # auto-detect: bf16 on Ampere+ GPUs, fp16 otherwise
    load_in_4bit=True,     # QLoRA-style 4-bit loading
)
```

**Why `unsloth/llama-3-8b-Instruct-bnb-4bit` instead of the original Meta repo?**
Unsloth maintains pre-quantized versions of popular models on the Hub. These download
faster (already quantized, smaller file size) and are verified to work correctly with
Unsloth's optimized kernels.

---

### 5.3 Attaching LoRA with Unsloth's API

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,         # Unsloth's optimized kernels require dropout=0 for full speed
    bias="none",
    use_gradient_checkpointing="unsloth",   # Unsloth's custom checkpointing — even more memory savings
    random_state=42,
)
```

`use_gradient_checkpointing="unsloth"` activates Unsloth's custom gradient checkpointing
implementation, which trades a small amount of recomputation for significant additional
memory savings, useful for fitting longer sequences or larger batch sizes on limited
VRAM.

---

### 5.4 Preparing the dataset for training

```python
def formatting_func(examples: dict) -> dict:
    """
    Format raw examples into the model's chat template.
    Must return a dict with a 'text' key containing the fully formatted string,
    including the EOS token so the model learns where a response ends.
    """
    texts = []
    for instruction, response in zip(examples["instruction"], examples["response"]):
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_func, batched=True)
```

**Why the EOS token matters:** Without an explicit end-of-sequence token after each
training example, the model doesn't learn when a response should stop. At inference
time, this can manifest as the model continuing to generate text past where it should
have stopped (e.g., starting to hallucinate a follow-up question and answering it too).

---

### 5.5 The training loop with `SFTTrainer`

`SFTTrainer` (Supervised Fine-Tuning Trainer), from the `trl` library, wraps Hugging
Face's standard `Trainer` with conveniences specific to instruction fine-tuning.

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",     # which column holds the formatted training text
    max_seq_length=max_seq_length,
    dataset_num_proc=2,             # parallel preprocessing workers

    args=TrainingArguments(
        per_device_train_batch_size=2,   # examples per GPU per step
        gradient_accumulation_steps=4,   # effective batch size = 2 * 4 = 8
        warmup_steps=10,                  # gradually ramp up learning rate at the start
        num_train_epochs=3,               # how many passes over the full dataset
        learning_rate=2e-4,                # typical LoRA learning rate (higher than full fine-tuning)
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,                  # print loss every N steps
        optim="adamw_8bit",                # 8-bit Adam optimizer — saves more memory
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="outputs",
        save_strategy="epoch",
    ),
)

trainer_stats = trainer.train()
```

**Why the LoRA learning rate (2e-4) is higher than typical full fine-tuning rates
(e.g., 2e-5):** Since LoRA's `B` matrix starts at zero and only a small number of
parameters are being trained, a larger learning rate is needed to make meaningful
progress within a reasonable number of steps, without destabilizing the (much larger,
frozen) rest of the network.

**Understanding gradient accumulation:** `per_device_train_batch_size=2` with
`gradient_accumulation_steps=4` means the model processes 2 examples, computes
gradients, processes another 2, accumulates those gradients, and so on for 4 micro-
batches before actually updating the weights, simulating a batch size of 8 without
needing the memory to hold 8 examples' activations simultaneously. This is essential
when VRAM is the binding constraint, which it usually is for LLM fine-tuning.

---

### 5.6 Monitoring training loss

```python
# trainer_stats contains training metrics
print(trainer_stats.metrics)
# {'train_runtime': 612.4, 'train_samples_per_second': 1.96, 
#  'train_loss': 0.847, ...}

# A healthy fine-tuning run shows:
# - Loss decreasing steadily over the first epoch
# - Loss continuing to decrease, but more slowly, in later epochs
# - NOT loss increasing or oscillating wildly (signals learning rate too high)
# - NOT loss plateauing immediately at a high value (signals data formatting issue)
```

---

### 5.7 Saving the fine-tuned model

```python
# Save just the LoRA adapter (small, ~50-200MB)
model.save_pretrained("llama3_legal_assistant_lora")
tokenizer.save_pretrained("llama3_legal_assistant_lora")

# Save merged model (base + adapter combined, full model size)
model.save_pretrained_merged(
    "llama3_legal_assistant_merged",
    tokenizer,
    save_method="merged_16bit",   # or "merged_4bit" for a smaller, quantized merged model
)

# Push to Hugging Face Hub
model.push_to_hub_merged(
    "your-username/llama3-legal-assistant",
    tokenizer,
    save_method="merged_16bit",
    token="hf_...",
)

# Export to GGUF format for use with llama.cpp / Ollama (popular for local deployment)
model.save_pretrained_gguf(
    "llama3_legal_assistant_gguf",
    tokenizer,
    quantization_method="q4_k_m",   # a good quality/size tradeoff for local inference
)
```

---

## 6. Evaluation — Did Fine-tuning Actually Help?

### 6.1 Why this step is often skipped (and shouldn't be)

It's tempting to declare success once training loss goes down and the model "feels"
better on a few manual tests. This is insufficient. Fine-tuning can:
- **Overfit** to the training examples, hurting generalization to new inputs
- **Catastrophically forget** general capabilities the base model had (e.g., degraded
  reasoning on topics outside the fine-tuning domain)
- **Improve on the metric you tracked** while silently regressing on something you
  didn't measure (e.g., better domain jargon usage but worse instruction-following)

A rigorous before/after comparison on held-out data is mandatory before considering a
fine-tune "done."

---

### 6.2 Setting up a held-out evaluation set

This must be data the model never saw during training, ideally collected or split
*before* training begins.

```python
# When preparing data, ALWAYS split before formatting/training
from datasets import load_dataset

full_dataset = load_dataset("your-username/legal-qa-dataset", split="train")
split = full_dataset.train_test_split(test_size=0.15, seed=42)

train_dataset = split["train"]   # used for fine-tuning
eval_dataset  = split["test"]    # NEVER shown to the model during training — held out for evaluation
```

---

### 6.3 Quantitative comparison: base model vs fine-tuned model

```python
from unsloth import FastLanguageModel
import torch

def generate_response(model, tokenizer, instruction: str, max_new_tokens: int = 200) -> str:
    FastLanguageModel.for_inference(model)   # enables Unsloth's faster inference mode
    messages = [{"role": "user", "content": instruction}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs, max_new_tokens=max_new_tokens, temperature=0.3, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response


def compare_models(base_model, finetuned_model, tokenizer, eval_examples: list[dict]) -> dict:
    """Generate responses from both models on the same held-out questions."""
    comparisons = []
    for example in eval_examples:
        instruction = example["instruction"]
        base_response = generate_response(base_model, tokenizer, instruction)
        ft_response   = generate_response(finetuned_model, tokenizer, instruction)
        comparisons.append({
            "instruction": instruction,
            "base_response": base_response,
            "finetuned_response": ft_response,
            "reference": example.get("response", None),
        })
    return comparisons
```

---

### 6.4 Using an LLM as judge

For open-ended generation quality (not just exact-match accuracy), use a stronger LLM
(e.g., GPT-4o) to score outputs: a well-established evaluation pattern called
"LLM-as-judge."

```python
from openai import OpenAI
from pydantic import BaseModel, Field
import instructor

client = instructor.from_openai(OpenAI())

class ComparisonJudgment(BaseModel):
    winner: str = Field(description="'base', 'finetuned', or 'tie'")
    reasoning: str = Field(description="Brief explanation of the judgment")
    finetuned_score: int = Field(ge=1, le=10, description="Quality score for the fine-tuned response")
    base_score: int = Field(ge=1, le=10, description="Quality score for the base model response")

def judge_comparison(instruction: str, base_response: str, ft_response: str, domain: str) -> ComparisonJudgment:
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""You are evaluating two AI responses for a {domain} use case.

Instruction: {instruction}

Response A (base model): {base_response}

Response B (fine-tuned model): {ft_response}

Judge which response is better for this {domain} context, considering accuracy,
appropriate tone/format for the domain, and helpfulness. Score each 1-10."""
        }],
        response_model=ComparisonJudgment,
        temperature=0,
    )


# Run the comparison across the eval set
results = []
for comparison in comparisons:
    judgment = judge_comparison(
        comparison["instruction"],
        comparison["base_response"],
        comparison["finetuned_response"],
        domain="legal assistance",
    )
    results.append(judgment)

# Aggregate
finetuned_wins = sum(1 for j in results if j.winner == "finetuned")
base_wins      = sum(1 for j in results if j.winner == "base")
ties           = sum(1 for j in results if j.winner == "tie")
avg_ft_score   = sum(j.finetuned_score for j in results) / len(results)
avg_base_score = sum(j.base_score for j in results) / len(results)

print(f"Fine-tuned wins: {finetuned_wins}/{len(results)}")
print(f"Base wins:       {base_wins}/{len(results)}")
print(f"Ties:            {ties}/{len(results)}")
print(f"Avg fine-tuned score: {avg_ft_score:.2f}/10")
print(f"Avg base score:       {avg_base_score:.2f}/10")
```

**Important caveat about LLM-as-judge:** The judge model has its own biases (e.g.,
favoring longer responses, favoring responses stylistically similar to its own outputs).
For high-stakes evaluation, combine LLM-as-judge with a sample of human review, and
consider randomizing which response is labeled "A" vs "B" to control for position bias.

---

### 6.5 Checking for catastrophic forgetting

Test the fine-tuned model on general capabilities **outside** the fine-tuning domain to
verify it hasn't lost broad competence.

```python
general_capability_tests = [
    "Write a Python function to reverse a string.",
    "What is the capital of Australia?",
    "Explain the difference between a list and a tuple in Python.",
    "Solve: if x + 5 = 12, what is x?",
]

print("Checking for catastrophic forgetting (general capabilities):\n")
for question in general_capability_tests:
    base_resp = generate_response(base_model, tokenizer, question, max_new_tokens=100)
    ft_resp   = generate_response(finetuned_model, tokenizer, question, max_new_tokens=100)
    print(f"Q: {question}")
    print(f"  Base:       {base_resp[:150]}")
    print(f"  Fine-tuned: {ft_resp[:150]}")
    print()

# If the fine-tuned model's answers to these GENERAL questions are noticeably
# worse than the base model's, you may have overfit too aggressively —
# consider: fewer epochs, lower learning rate, or a lower LoRA rank.
```

---

### 6.6 A practical evaluation checklist

Before declaring a fine-tune successful, verify:

1. ✅ **Domain task quality improved:** fine-tuned model wins or ties on domain-specific
   held-out examples (Section 6.4)
2. ✅ **No catastrophic forgetting:** general capability tests show comparable
   performance to the base model (Section 6.5)
3. ✅ **Training loss converged sensibly:** decreased steadily without erratic spikes
   (Section 5.6)
4. ✅ **No obvious overfitting signs:** model doesn't just regurgitate training examples
   verbatim when given slightly different phrasings of the same question
5. ✅ **Output format consistency:** if the fine-tuning goal was format/style
   consistency, verify this holds across a range of held-out prompts, not just the
   ones you eyeballed during development

---

## 7. Key Takeaways

1. **Fine-tuning teaches HOW, not WHAT.** Use RAG for facts and current information.
   Use fine-tuning for behavior, format, tone, and deeply ingrained task skills. Most
   "I need to fine-tune on our data" instincts are actually RAG use cases.

2. **PEFT (LoRA) makes fine-tuning accessible by training ~0.1-1% of parameters.**
   The frozen base weights need no gradients or optimizer state; only the small
   low-rank `A` and `B` matrices are trained. `B` starts at zero so training begins
   from the exact base model behavior.

3. **QLoRA adds 4-bit quantization of the frozen base model**, cutting memory
   requirements roughly 3-4x further. Computation still happens in bf16 on the fly;
   only storage is 4-bit, preserving training stability.

4. **Unsloth provides the same QLoRA math with custom optimized kernels**, roughly
   2x faster and using less VRAM, with no accuracy tradeoff. It's the practical tool
   of choice for fine-tuning on consumer/free-tier GPUs (e.g., Colab T4).

5. **Match your training data format to the model's chat template.** Use
   `tokenizer.apply_chat_template()` rather than hand-rolling a format: mismatched
   formats degrade the instruction-following ability the base model already has.

6. **The LoRA learning rate (~2e-4) is higher than full fine-tuning rates (~2e-5)**
   because you're updating far fewer parameters and need a larger step size to make
   meaningful progress in a reasonable number of steps.

7. **Always evaluate on held-out data, against the base model, on both domain tasks
   AND general capabilities.** Training loss going down is necessary but nowhere near
   sufficient evidence that fine-tuning succeeded. Use LLM-as-judge for open-ended
   quality comparison, and explicitly test for catastrophic forgetting.

---

## 8. Practice Exercises

### Exercise 1 — Rank Ablation (Easy-Medium)
Fine-tune the same small model (e.g., a 1-3B parameter model for faster iteration) on
the same dataset three times, with `r=4`, `r=16`, and `r=64`. Compare final training
loss, evaluation quality (using the LLM-as-judge pattern), and wall-clock training time
for each. Plot rank vs evaluation quality.

### Exercise 2 — Catastrophic Forgetting Detector (Medium)
Build a standardized "general capability" test suite of 15 questions spanning coding,
math, general knowledge, and reasoning. Write a function that runs this suite against
any fine-tuned model and the base model, scores both with an LLM judge, and flags any
question where the fine-tuned model's score dropped by more than 2 points, a signal
of forgetting on that specific capability.

### Exercise 3 — Dataset Size Curve (Medium-Hard)
Using the same hyperparameters, fine-tune on subsets of your dataset: 50, 200, 500, and
all available examples. Evaluate each resulting model on the same held-out set. Plot
dataset size vs evaluation win-rate against the base model. Identify the point of
diminishing returns: where adding more data stops meaningfully improving quality.

### Exercise 4 — Merge vs Adapter Inference Benchmark (Hard)
Benchmark inference latency and memory usage for: (a) the base model + LoRA adapter
loaded separately (using `PeftModel`), and (b) the merged model (using
`merge_and_unload()`). Run 50 generations of similar length with each setup and report
average latency, peak memory, and confirm the outputs are functionally equivalent.

---

*Next: Phase 08, Production & Deployment*  
*You will containerize your AI services, build FastAPI backends with streaming,
add automated evaluation with RAGAS, and set up observability with Langfuse,
closing the gap between a working prototype and a production-ready product.*
