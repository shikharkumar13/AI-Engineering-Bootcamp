"""
inference_api.py — Serve the Fine-tuned Model via FastAPI

Runs on a GPU machine (the same Colab instance, a cloud GPU, or your own
hardware). Wraps the merged fine-tuned model in a simple REST API matching
the same request/response shape you've used throughout this course (Phase 01
style), so it's a drop-in replacement for an OpenAI/Claude call in your own
applications.

Run with:
    uvicorn inference_api:app --host 0.0.0.0 --port 8000

Then call it like:
    curl -X POST http://localhost:8000/chat \\
         -H "Content-Type: application/json" \\
         -d '{"message": "My invoice is wrong, can you help?"}'
"""

import os
import time
import torch
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


MODEL_PATH    = os.getenv("FINETUNED_MODEL_PATH", "llama3_support_assistant_lora")
MAX_SEQ_LENGTH = 2048
SYSTEM_PROMPT = (
    "You are a customer support assistant. Always respond in this exact format:\n"
    "Issue: <restate the problem>\nResolution: <the fix>\n"
    "Next Steps: <what happens next, or 'None needed'>"
)

# Globals populated at startup
model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup, not on every request."""
    global model, tokenizer

    if not torch.cuda.is_available():
        print("⚠ WARNING: No GPU detected. The model will fail to load.")
        print("  This API must run on a CUDA-capable machine.")

    print(f"Loading fine-tuned model from: {MODEL_PATH}")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print("✓ Model loaded and ready for inference.")

    yield   # app runs here

    print("Shutting down — releasing model resources.")
    del model, tokenizer
    torch.cuda.empty_cache()


app = FastAPI(
    title="Fine-tuned Support Assistant API",
    description="Serves the QLoRA fine-tuned Llama-3-8B support assistant",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request/response schemas ───────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(description="The customer's message or question")
    max_new_tokens: int = Field(default=250, ge=10, le=1000)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)

class ChatResponse(BaseModel):
    response: str
    issue: str | None = None
    resolution: str | None = None
    next_steps: str | None = None
    latency_s: float
    model: str = "llama3-support-assistant-qlora"

class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    model_loaded: bool


# ── Endpoints ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check API and model status."""
    return HealthResponse(
        status="ok" if model is not None else "model not loaded",
        gpu_available=torch.cuda.is_available(),
        model_loaded=model is not None,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Generate a structured support response for a customer message."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    t0 = time.time()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": request.message},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=request.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    raw_response = tokenizer.decode(
        outputs[0][inputs.shape[1]:], skip_special_tokens=True
    ).strip()

    # Parse the structured format back into fields (best-effort)
    issue, resolution, next_steps = _parse_structured_response(raw_response)

    return ChatResponse(
        response=raw_response,
        issue=issue,
        resolution=resolution,
        next_steps=next_steps,
        latency_s=round(time.time() - t0, 2),
    )


def _parse_structured_response(text: str) -> tuple[str | None, str | None, str | None]:
    """Best-effort parsing of the Issue/Resolution/Next Steps format into fields."""
    issue = resolution = next_steps = None

    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("Issue:"):
            issue = line.replace("Issue:", "").strip()
        elif line.startswith("Resolution:"):
            resolution = line.replace("Resolution:", "").strip()
        elif line.startswith("Next Steps:"):
            next_steps = line.replace("Next Steps:", "").strip()

    return issue, resolution, next_steps


@app.get("/")
def root():
    return {
        "service": "Fine-tuned Support Assistant API",
        "endpoints": {
            "POST /chat": "Send a support message, get a structured response",
            "GET /health": "Check API and model status",
        },
    }
