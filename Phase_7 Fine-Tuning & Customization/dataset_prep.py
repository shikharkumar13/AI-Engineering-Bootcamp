"""
dataset_prep.py — Domain Dataset Preparation

Domain chosen for this project: a customer support assistant that must always
respond in a strict structured format:

    Issue: <one-line restatement of the problem>
    Resolution: <the fix or answer>
    Next Steps: <what the customer should do next, or "None needed">

This is a deliberately good fine-tuning use case (per Phase 07's decision
framework): it's a FORMAT/BEHAVIOR pattern, not new facts, so fine-tuning is
the right tool — few-shot prompting could approximate this, but baking it in
via fine-tuning makes the format unbreakable across thousands of varied
real-world tickets, including in agent pipelines where you can't easily
re-inject a long few-shot prompt every call.

Swap SAMPLE_TICKETS below with your own domain data to adapt this project to
any niche use case.
"""

import random
from pathlib import Path
from datasets import Dataset


# ── Sample domain dataset ──────────────────────────────────────────────────────
# In a real project, this would be hundreds-to-thousands of real examples.
# We include 40 illustrative examples here; train.py works with any size,
# but more data (and more diverse phrasing) produces a more robust fine-tune.

SAMPLE_TICKETS = [
    {
        "instruction": "My invoice shows $49 but I'm on the $29 plan. Why was I overcharged?",
        "response": (
            "Issue: Customer was billed $49 instead of the expected $29 plan rate.\n"
            "Resolution: This is usually caused by a mid-cycle plan upgrade being "
            "prorated into the current invoice. I've checked your account and can "
            "confirm the $20 difference matches a prorated upgrade charge from the 14th.\n"
            "Next Steps: No action needed — this is expected billing behavior. If you "
            "did not intend to upgrade, reply here and I'll process a refund for the "
            "difference within 1 business day."
        ),
    },
    {
        "instruction": "The app keeps crashing every time I try to export a report to PDF.",
        "response": (
            "Issue: App crashes specifically during PDF export of reports.\n"
            "Resolution: This matches a known issue affecting reports with more than "
            "500 rows, caused by a memory limit in the PDF rendering engine. A fix is "
            "scheduled for next week's release (v4.2.1).\n"
            "Next Steps: As a temporary workaround, try exporting to CSV instead, or "
            "filter the report to under 500 rows before exporting to PDF. I'll notify "
            "you personally when v4.2.1 ships."
        ),
    },
    {
        "instruction": "I forgot my password and the reset email never arrived.",
        "response": (
            "Issue: Password reset email not received.\n"
            "Resolution: I've manually triggered a new reset email to the address on "
            "file. This sometimes happens if the original email landed in spam or was "
            "blocked by a corporate mail filter.\n"
            "Next Steps: Please check your spam/junk folder for an email from "
            "noreply@ourapp.com in the next 5 minutes. If it still doesn't arrive, "
            "reply here and I'll reset your password manually from our end."
        ),
    },
    {
        "instruction": "Can I get a refund for my annual subscription? I only used it for a month.",
        "response": (
            "Issue: Customer requesting a partial refund on an annual subscription "
            "after one month of use.\n"
            "Resolution: Per our refund policy, annual subscriptions are eligible for "
            "a prorated refund within the first 30 days. Since you're within that "
            "window, I've processed a refund of $231.00 (11 months remaining, prorated).\n"
            "Next Steps: The refund will appear on your original payment method within "
            "5-7 business days. Your account will remain active until the end of the "
            "current billing period, then automatically downgrade to the free tier."
        ),
    },
    {
        "instruction": "How do I add a teammate to my workspace?",
        "response": (
            "Issue: Customer needs guidance on adding a team member to their workspace.\n"
            "Resolution: You can invite teammates from Settings > Team > Invite Member. "
            "Enter their email address and select a role (Admin, Editor, or Viewer). "
            "They'll receive an invitation email with a link to join.\n"
            "Next Steps: No further action needed from our side. If your teammate "
            "doesn't receive the invitation within 10 minutes, ask them to check spam, "
            "or you can resend the invite from the same Team settings page."
        ),
    },
    {
        "instruction": "Your API is returning 500 errors on the /users endpoint since this morning.",
        "response": (
            "Issue: Customer reporting consistent 500 errors on the /users API endpoint "
            "since this morning.\n"
            "Resolution: I've checked our status dashboard and confirmed there was a "
            "20-minute partial outage affecting the /users endpoint starting at 9:14 AM "
            "UTC, caused by a database connection pool issue. This was resolved at 9:34 AM UTC.\n"
            "Next Steps: The endpoint should be functioning normally now. Please retry "
            "your requests. If you're still seeing 500 errors, reply with a sample "
            "request ID and I'll investigate your specific case further."
        ),
    },
    {
        "instruction": "I want to downgrade from Pro to the Free plan.",
        "response": (
            "Issue: Customer requesting to downgrade from Pro to the Free plan.\n"
            "Resolution: I've scheduled your downgrade to take effect at the end of "
            "your current billing cycle (March 14th), so you'll retain Pro features "
            "until then with no service interruption.\n"
            "Next Steps: No action needed. You'll receive a confirmation email when "
            "the downgrade completes. Note that the Free plan has a 3-project limit, "
            "so if you have more than 3 active projects, you'll need to archive some "
            "before the downgrade date."
        ),
    },
    {
        "instruction": "Is there a dark mode option?",
        "response": (
            "Issue: Customer asking whether dark mode is available.\n"
            "Resolution: Yes — dark mode is available under Settings > Appearance > "
            "Theme. You can choose Light, Dark, or System (which follows your "
            "device's setting automatically).\n"
            "Next Steps: No action needed. Let us know if you run into any display "
            "issues after switching themes."
        ),
    },
]


def expand_dataset_with_variations(base_examples: list[dict], target_size: int = 40) -> list[dict]:
    """
    For demo purposes, lightly paraphrase the seed examples to reach a larger
    dataset size. In a real project, replace this with your actual collected
    examples — synthetic paraphrasing is a reasonable bootstrap, not a
    substitute for real data at scale.
    """
    if len(base_examples) >= target_size:
        return base_examples[:target_size]

    expanded = list(base_examples)
    prefixes = ["", "Hi, ", "Hello, ", "Hey there, ", "Quick question - ", "Hi support team, "]

    while len(expanded) < target_size:
        original = random.choice(base_examples)
        prefix = random.choice(prefixes)
        variant = {
            "instruction": prefix + original["instruction"],
            "response": original["response"],
        }
        expanded.append(variant)

    return expanded


def build_dataset(target_size: int = 40, eval_fraction: float = 0.15, seed: int = 42) -> dict:
    """
    Build the train/eval split for fine-tuning.

    Splits the *base* tickets first, then expands each side independently into
    paraphrased variants. Expanding first and splitting afterward (the naive
    order) lets near-duplicate paraphrases of the same base ticket land in both
    train and eval — since expand_dataset_with_variations only varies the
    instruction's greeting prefix and keeps the response text identical, that
    would mean the eval set isn't actually held out: the model could have seen
    the exact same response, word for word, during training.

    Returns:
        dict with 'train' and 'eval' Dataset objects
    """
    random.seed(seed)
    base_examples = list(SAMPLE_TICKETS)
    random.shuffle(base_examples)

    n_eval_base = max(1, round(len(base_examples) * eval_fraction))
    eval_base, train_base = base_examples[:n_eval_base], base_examples[n_eval_base:]

    eval_size = max(1, round(target_size * eval_fraction))
    train_size = target_size - eval_size

    train_examples = expand_dataset_with_variations(train_base, target_size=train_size)
    eval_examples  = expand_dataset_with_variations(eval_base, target_size=eval_size)
    random.shuffle(train_examples)
    random.shuffle(eval_examples)

    return {
        "train": Dataset.from_list(train_examples),
        "eval":  Dataset.from_list(eval_examples),
    }


def format_for_chat_template(example: dict, tokenizer) -> dict:
    """
    Format a single example using the model's chat template.
    Call this via dataset.map() with a tokenizer bound via functools.partial
    or a lambda, as shown in train.py.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a customer support assistant. Always respond in this "
                "exact format:\nIssue: <restate the problem>\nResolution: <the fix>\n"
                "Next Steps: <what happens next, or 'None needed'>"
            ),
        },
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False) + tokenizer.eos_token
    return {"text": text}


def save_dataset_to_disk(dataset_dict: dict, output_dir: str = "support_dataset"):
    """Save the train/eval datasets to disk for reuse without rebuilding."""
    Path(output_dir).mkdir(exist_ok=True)
    dataset_dict["train"].save_to_disk(f"{output_dir}/train")
    dataset_dict["eval"].save_to_disk(f"{output_dir}/eval")
    print(f"  ✓ Saved {len(dataset_dict['train'])} train / "
          f"{len(dataset_dict['eval'])} eval examples to {output_dir}/")


if __name__ == "__main__":
    print("Building domain dataset (customer support, structured format)...")
    data = build_dataset(target_size=40)

    print(f"\n  Train examples: {len(data['train'])}")
    print(f"  Eval examples:  {len(data['eval'])}")

    print(f"\n  Sample training example:")
    sample = data["train"][0]
    print(f"  Instruction: {sample['instruction']}")
    print(f"  Response:\n  {sample['response']}")

    save_dataset_to_disk(data)
