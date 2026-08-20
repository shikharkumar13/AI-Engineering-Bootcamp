# Phase 07 — Domain Expert Fine-tune

## What this project does
Fine-tunes Llama-3-8B with QLoRA (via Unsloth) to **always respond in a strict
structured format** (`Issue / Resolution / Next Steps`) for customer support
tickets: a behavioral pattern that's a textbook fine-tuning use case (not a
RAG use case; see Phase 07 Section 1's decision framework).

Pipeline: prepare dataset → fine-tune with QLoRA → evaluate vs base model
(LLM-as-judge + catastrophic forgetting check) → serve via FastAPI.

## ⚠️ GPU requirement
`train.py` and `evaluate.py` **require a CUDA GPU**. Unsloth and bitsandbytes
cannot run on CPU. Use a **free Google Colab T4 GPU**: that's all this project
needs thanks to QLoRA's memory efficiency.

`inference_api.py` also needs a GPU to serve the model (run it on the same
Colab instance, or any cloud GPU). `demo_client.py` does NOT need a GPU: it's
just an HTTP client that talks to the API.

## Project structure
```
Phase_7 Fine-Tuning & Customization/
├── dataset_prep.py     ← Domain dataset (customer support, structured format)
├── train.py             ← QLoRA fine-tuning script (Unsloth) — GPU required
├── evaluate.py            ← Base vs fine-tuned comparison — GPU required
├── inference_api.py        ← FastAPI server for the fine-tuned model — GPU required
├── demo_client.py            ← Test client for the API — NO GPU needed
├── requirements.txt
└── .env.example         ← copy to .env locally, or set as Colab env vars (see step 4)
```

## Quick start (Google Colab)

### 1. Open a new Colab notebook
Runtime > Change runtime type > Hardware accelerator > **T4 GPU**

### 2. Install dependencies (Colab cell)
```python
!pip install unsloth
!pip install --no-deps trl peft accelerate bitsandbytes
!pip install instructor openai python-dotenv fastapi uvicorn
```

### 3. Upload project files
Upload `dataset_prep.py`, `train.py`, `evaluate.py` to the Colab file browser,
or paste their contents into cells.

### 4. Set environment variables (Colab cell)
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."     # for LLM-as-judge evaluation
os.environ["HF_TOKEN"] = "hf_..."           # optional, for pushing to Hub
```

### 5. Train
```python
!python train.py
```
Takes roughly 5-15 minutes on a T4 for the demo dataset (40 examples, 3 epochs).

### 6. Evaluate
```python
!python evaluate.py
```
Compares the fine-tuned model against the base model on held-out examples,
plus a catastrophic-forgetting check on general capability questions.

### 7. Serve
```python
# In Colab, use a tunnel like ngrok or Colab's built-in port forwarding
!uvicorn inference_api:app --host 0.0.0.0 --port 8000
```

### 8. Test the API (from anywhere — no GPU needed)
```bash
python demo_client.py
```

## API reference (once inference_api.py is running)

```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "I was charged twice this month, please help."}'
```

```json
{
  "response": "Issue: Customer billed twice for the same subscription period...",
  "issue": "Customer billed twice for the same subscription period.",
  "resolution": "I've identified the duplicate charge and processed a refund...",
  "next_steps": "No action needed — refund will appear within 5-7 business days.",
  "latency_s": 1.84,
  "model": "llama3-support-assistant-qlora"
}
```

## Concepts demonstrated

| File | Concept |
|------|---------|
| `dataset_prep.py` | Chat template formatting, train/eval split, format-as-behavior dataset design |
| `train.py` | 4-bit QLoRA loading, LoRA adapter config, SFTTrainer, gradient accumulation |
| `evaluate.py` | Held-out evaluation, LLM-as-judge, catastrophic forgetting detection |
| `inference_api.py` | Loading a fine-tuned model for serving, FastAPI lifespan model loading |
| `demo_client.py` | Treating your fine-tuned model exactly like any other LLM API (Phase 01 callback) |

## Adapting this to your own domain

Replace `SAMPLE_TICKETS` in `dataset_prep.py` with your own examples. The key
design principle from Phase 07: fine-tuning works best for **behavior/format**
patterns (tone, structure, task-specific skill), not for teaching new facts;
use RAG (Phase 04) for facts that need to stay current.
