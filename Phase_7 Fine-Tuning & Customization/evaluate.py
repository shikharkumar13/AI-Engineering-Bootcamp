"""
evaluate.py — Base vs Fine-tuned Model Evaluation

ALSO DESIGNED FOR GPU (Colab). Run after train.py completes.

Implements the full evaluation checklist from Phase 07 Section 6:
  1. Generate responses from both base and fine-tuned models on held-out data
  2. Score with LLM-as-judge (GPT-4o-mini) on domain task quality
  3. Check for catastrophic forgetting on general capability questions
  4. Report a pass/fail summary against the checklist
"""

import os
import json
import torch
from dataclasses import dataclass, field
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


MODEL_NAME    = "unsloth/llama-3-8b-Instruct-bnb-4bit"
ADAPTER_PATH  = "llama3_support_assistant_lora"
MAX_SEQ_LENGTH = 2048

SYSTEM_PROMPT = (
    "You are a customer support assistant. Always respond in this exact format:\n"
    "Issue: <restate the problem>\nResolution: <the fix>\n"
    "Next Steps: <what happens next, or 'None needed'>"
)

# General capability tests — outside the fine-tuning domain entirely.
# If the fine-tuned model does noticeably worse here, that's catastrophic forgetting.
GENERAL_CAPABILITY_TESTS = [
    "Write a Python function that checks if a number is prime.",
    "What is the capital of Australia?",
    "Explain the difference between a list and a tuple in Python.",
    "If x + 5 = 12, what is x?",
    "Translate 'good morning' to French.",
]


# ── Structured judgment schema ─────────────────────────────────────────────────

class ComparisonJudgment(BaseModel):
    winner: str = Field(description="'base', 'finetuned', or 'tie'")
    reasoning: str = Field(description="Brief explanation, 1-2 sentences")
    finetuned_score: int = Field(ge=1, le=10, description="Quality score 1-10 for fine-tuned response")
    base_score: int = Field(ge=1, le=10, description="Quality score 1-10 for base response")
    finetuned_followed_format: bool = Field(
        description="Did the fine-tuned response follow the required Issue/Resolution/Next Steps format?"
    )


@dataclass
class EvalReport:
    domain_results:  list[dict] = field(default_factory=list)
    general_results: list[dict] = field(default_factory=list)

    def summarize(self):
        n = len(self.domain_results)
        ft_wins = sum(1 for r in self.domain_results if r["judgment"].winner == "finetuned")
        base_wins = sum(1 for r in self.domain_results if r["judgment"].winner == "base")
        ties = n - ft_wins - base_wins
        avg_ft_score = sum(r["judgment"].finetuned_score for r in self.domain_results) / n
        avg_base_score = sum(r["judgment"].base_score for r in self.domain_results) / n
        format_compliance = sum(
            1 for r in self.domain_results if r["judgment"].finetuned_followed_format
        ) / n

        print(f"\n{'='*62}")
        print(f"  DOMAIN TASK EVALUATION ({n} held-out examples)")
        print(f"{'='*62}")
        print(f"  Fine-tuned wins: {ft_wins}/{n}")
        print(f"  Base wins:       {base_wins}/{n}")
        print(f"  Ties:            {ties}/{n}")
        print(f"  Avg fine-tuned score: {avg_ft_score:.2f}/10")
        print(f"  Avg base score:       {avg_base_score:.2f}/10")
        print(f"  Format compliance (fine-tuned): {format_compliance:.0%}")

        if self.general_results:
            print(f"\n{'='*62}")
            print(f"  CATASTROPHIC FORGETTING CHECK ({len(self.general_results)} general questions)")
            print(f"{'='*62}")
            forgetting_flags = 0
            for r in self.general_results:
                j = r["judgment"]
                delta = j.finetuned_score - j.base_score
                flag = "⚠ POSSIBLE FORGETTING" if delta <= -2 else "✓ OK"
                if delta <= -2:
                    forgetting_flags += 1
                print(f"  [{flag}] '{r['question'][:50]}...' "
                      f"(base={j.base_score}, ft={j.finetuned_score}, Δ={delta:+d})")
            print(f"\n  Forgetting flags: {forgetting_flags}/{len(self.general_results)}")

        print(f"\n{'='*62}")
        print(f"  CHECKLIST SUMMARY")
        print(f"{'='*62}")
        checks = [
            ("Domain task quality improved (ft wins >= base wins)", ft_wins >= base_wins),
            ("Average fine-tuned score >= 6/10", avg_ft_score >= 6),
            ("Format compliance >= 90%", format_compliance >= 0.9),
            ("No catastrophic forgetting (0 flags)",
             not self.general_results or forgetting_flags == 0),
        ]
        for check_name, passed in checks:
            print(f"  [{'✓' if passed else '✗'}] {check_name}")

        all_passed = all(passed for _, passed in checks)
        print(f"\n  {'✅ ALL CHECKS PASSED' if all_passed else '⚠ SOME CHECKS FAILED — review above'}")
        return all_passed


# ── Model loading and generation ───────────────────────────────────────────────

def load_models():
    """Load both the base model and the fine-tuned (adapter-applied) model."""
    from unsloth import FastLanguageModel

    print("  Loading base model...")
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(base_model)

    print("  Loading fine-tuned model (base + LoRA adapter)...")
    ft_model, _ = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH,   # loading from the saved adapter directory
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(ft_model)

    return base_model, ft_model, tokenizer


def generate(model, tokenizer, instruction: str, use_system_prompt: bool = True,
             max_new_tokens: int = 250) -> str:
    """Generate a response from a model given an instruction."""
    messages = []
    if use_system_prompt:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": instruction})

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response.strip()


# ── LLM-as-judge ────────────────────────────────────────────────────────────────

def judge(instruction: str, base_response: str, ft_response: str,
          context: str = "customer support") -> ComparisonJudgment:
    """Use GPT-4o-mini to judge which response is better."""
    import instructor
    from openai import OpenAI

    client = instructor.from_openai(OpenAI())

    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""You are evaluating two AI responses for a {context} use case.

The fine-tuned model was trained to ALWAYS respond in this format:
"Issue: ...\\nResolution: ...\\nNext Steps: ..."

Instruction: {instruction}

Response A (base model): {base_response}

Response B (fine-tuned model): {ft_response}

Judge which response is better considering: accuracy, helpfulness, and (for
Response B specifically) whether it followed the required Issue/Resolution/
Next Steps format exactly. Score each 1-10."""
        }],
        response_model=ComparisonJudgment,
        temperature=0,
    )


# ── Main evaluation flow ───────────────────────────────────────────────────────

def run_domain_evaluation(base_model, ft_model, tokenizer) -> list[dict]:
    """Evaluate on held-out domain examples."""
    from datasets import load_from_disk

    try:
        eval_dataset = load_from_disk("eval_dataset_cache")
    except Exception:
        print("  ⚠ eval_dataset_cache not found — run train.py first, or using "
              "dataset_prep.py directly to build an eval set.")
        from dataset_prep import build_dataset
        eval_dataset = build_dataset(target_size=40)["eval"]

    print(f"\n  Evaluating on {len(eval_dataset)} held-out domain examples...")
    results = []

    for i, example in enumerate(eval_dataset):
        instruction = example["instruction"]
        print(f"  [{i+1}/{len(eval_dataset)}] {instruction[:60]}...")

        base_resp = generate(base_model, tokenizer, instruction, use_system_prompt=False)
        ft_resp   = generate(ft_model, tokenizer, instruction, use_system_prompt=True)

        judgment = judge(instruction, base_resp, ft_resp)

        results.append({
            "instruction": instruction,
            "base_response": base_resp,
            "finetuned_response": ft_resp,
            "judgment": judgment,
        })

    return results


def run_forgetting_check(base_model, ft_model, tokenizer) -> list[dict]:
    """Evaluate on general capability questions to check for catastrophic forgetting."""
    print(f"\n  Checking {len(GENERAL_CAPABILITY_TESTS)} general capability questions...")
    results = []

    for question in GENERAL_CAPABILITY_TESTS:
        print(f"  Testing: {question[:60]}...")
        base_resp = generate(base_model, tokenizer, question, use_system_prompt=False)
        ft_resp   = generate(ft_model, tokenizer, question, use_system_prompt=False)

        judgment = judge(question, base_resp, ft_resp, context="general assistance")

        results.append({
            "question": question,
            "base_response": base_resp,
            "finetuned_response": ft_resp,
            "judgment": judgment,
        })

    return results


def save_report(report: EvalReport, path: str = "eval_report.json"):
    """Save full evaluation results to a JSON file for later review."""
    serializable = {
        "domain_results": [
            {**{k: v for k, v in r.items() if k != "judgment"},
             "judgment": r["judgment"].model_dump()}
            for r in report.domain_results
        ],
        "general_results": [
            {**{k: v for k, v in r.items() if k != "judgment"},
             "judgment": r["judgment"].model_dump()}
            for r in report.general_results
        ],
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  ✓ Full report saved to {path}")


def main():
    print("=" * 62)
    print("  Phase 07 — Evaluation: Base vs Fine-tuned Model")
    print("=" * 62)

    base_model, ft_model, tokenizer = load_models()

    report = EvalReport()
    report.domain_results  = run_domain_evaluation(base_model, ft_model, tokenizer)
    report.general_results = run_forgetting_check(base_model, ft_model, tokenizer)

    report.summarize()
    save_report(report)


if __name__ == "__main__":
    main()
