# Phase 02 — Prompt Engineering

> **Prerequisites:** Phase 01 complete — you can call OpenAI, Claude, and Gemini APIs.  
> **What you'll learn:** Zero-shot, few-shot, and Chain-of-Thought prompting; structured
> outputs; function calling; system prompts and Jinja2 templates.  
> **Project:** A pipeline that reads unstructured text and returns validated JSON.

---

## Table of Contents

1. [The Big Picture — Why Prompt Engineering?](#1-the-big-picture--why-prompt-engineering)
2. [Zero-shot, One-shot & Few-shot Prompting](#2-zero-shot-one-shot--few-shot-prompting)
3. [Chain-of-Thought & Self-Consistency](#3-chain-of-thought--self-consistency)
4. [Structured Outputs & JSON Mode](#4-structured-outputs--json-mode)
5. [Function Calling & Tool Use](#5-function-calling--tool-use)
6. [The `instructor` Library](#6-the-instructor-library)
7. [System Prompts & Prompt Templates](#7-system-prompts--prompt-templates)
8. [Key Takeaways](#8-key-takeaways)
9. [Practice Exercises](#9-practice-exercises)

---

## 1. The Big Picture — Why Prompt Engineering?

Every AI engineer hits the same realization early on: **the model is fixed, but your
prompt is not.** The same underlying model can answer brilliantly or poorly depending
entirely on how you phrase the request.

Prompt engineering is the practice of designing inputs to LLMs to reliably get the
outputs you want. It has the highest return on investment of any AI skill — a better
prompt can improve accuracy dramatically with zero extra compute, zero extra cost, and
zero extra infrastructure.

**From your ML background:** Think of prompting as selecting the right "region of
behavior" from a model that has been trained on a vast distribution of text. The model
has learned to respond to patterns. Your prompt is the pattern that activates the
behavior you want. Few-shot examples are not random — they shift the probability
distribution of the model's output toward the pattern you demonstrate.

**What this phase covers in production contexts:**
- Reliably extracting structured data from messy text (emails, documents, articles)
- Building reusable prompt templates that non-engineers can modify
- Getting JSON back that your code can actually use, not markdown-wrapped strings
- Reasoning through multi-step problems accurately

---

## 2. Zero-shot, One-shot & Few-shot Prompting

### 2.1 Zero-shot prompting

Zero-shot means asking the model to do a task with no examples. Just describe what you
want.

```python
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def zero_shot(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # deterministic output for classification tasks
    )
    return response.choices[0].message.content

# Zero-shot sentiment classification — just ask
result = zero_shot(
    "Classify the sentiment of this review as positive, negative, or neutral.\n"
    "Review: 'The product is decent but grossly overpriced for what it offers.'\n"
    "Sentiment:"
)
print(result)  # → "negative"
```

**When to use zero-shot:** Start here. GPT-4o-mini and Claude Haiku are capable enough
to handle most classification, summarization, and extraction tasks without examples.
Only add examples if the output is inconsistent or wrong.

---

### 2.2 Few-shot prompting

Few-shot means including examples directly in the prompt. The model sees input-output
pairs and infers the pattern you want.

```python
def few_shot_sentiment(review: str) -> str:
    """Classify sentiment using in-context examples."""
    
    prompt = f"""Classify the sentiment of each review as positive, negative, or neutral.
Only output one word: positive, negative, or neutral.

Review: "Absolutely love it! Works perfectly and looks great."
Sentiment: positive

Review: "Broke after two days. Complete waste of money."
Sentiment: negative

Review: "It does the job. Nothing remarkable, but no complaints."
Sentiment: neutral

Review: "Shipping was fast, but the quality is below average."
Sentiment: negative

Review: "{review}"
Sentiment:"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5,  # we only need one word
    )
    return response.choices[0].message.content.strip().lower()


# Test it
reviews = [
    "The product is decent but grossly overpriced.",
    "Best purchase I've made all year!",
    "Arrived on time, works as described.",
]
for review in reviews:
    print(f"'{review[:50]}' → {few_shot_sentiment(review)}")
```

**Why few-shot works:** The model is not "learning" from the examples in the traditional
ML sense — its weights are not being updated. Instead, the examples are part of the
context the model attends to. They establish a clear pattern: given this type of input,
this is the expected format of output. The model's in-context learning ability lets it
generalize this pattern to new inputs.

---

### 2.3 One-shot prompting

One-shot is exactly what it sounds like — one example. It is less about improving
accuracy and more about showing the model the **exact format** of output you want.

```python
def one_shot_extraction(text: str) -> str:
    """
    Use a single example purely to define the output format.
    The example does not need to be about the same topic — just the same structure.
    """
    
    prompt = f"""Extract the key information from the text in this format:

Text: "John Smith called to discuss the Q3 budget meeting on Friday at 2pm.
He needs the financial report by Thursday."

Extraction:
Who: John Smith
What: Budget meeting discussion
When: Friday at 2pm
Deadline: Financial report by Thursday

---

Text: "{text}"

Extraction:"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content
```

---

### 2.4 How many examples to use

| Scenario | Recommended approach |
|---|---|
| Task is simple and well-described | Zero-shot — just ask |
| Output format is unusual or strict | One-shot — show the format |
| Model gives inconsistent results | Few-shot (3-5 examples) |
| Task is niche or domain-specific | Few-shot (5-10 examples) |
| Model still struggles | Fine-tuning (Phase 07) |

**A practical guideline:** If you can describe what you want clearly in words, try
zero-shot first. Only add examples if the output is wrong or inconsistent. Each example
costs tokens (and therefore money), so do not add them without reason.

---

### 2.5 Designing good few-shot examples

Poor few-shot examples undermine the technique. Good examples:

```python
# BAD: examples all have the same label — model will predict that label for everything
bad_examples = [
    ("Great product!", "positive"),
    ("Love it!", "positive"),
    ("Perfect quality.", "positive"),
]

# BAD: examples are not representative of the variation in real inputs
bad_examples_2 = [
    ("This is bad", "negative"),
    ("This sucks", "negative"),
    ("Terrible product", "negative"),
]

# GOOD: balanced, diverse, representative of the actual distribution
good_examples = [
    ("Absolutely love it! Works perfectly.", "positive"),
    ("Broke after two days. Complete waste of money.", "negative"),
    ("It does the job. Nothing remarkable.", "neutral"),
    ("Fast shipping but poor quality materials.", "negative"),
    ("Exactly as described, very happy.", "positive"),
]
```

**Rule:** Your examples should cover the full range of variation in real inputs. If your
actual data is 60% negative, 30% positive, 10% neutral, try to reflect that roughly in
your examples. Imbalanced examples bias the model toward certain outputs.

---

## 3. Chain-of-Thought & Self-Consistency

### 3.1 The problem CoT solves

Without Chain-of-Thought, the model generates an answer directly. For simple questions
this is fine, but for multi-step reasoning it often fails.

```python
# Without CoT — model jumps to an answer and gets it wrong
prompt = """If a train travels at 60 mph and needs to cover 150 miles, 
but makes a 30-minute stop halfway, how long is the total journey?"""

# Model might say: "2.5 hours" (forgetting the stop)
# Or: "3 hours" (correct but for wrong reasons)
# Or: "2 hours 30 minutes" (same as first, expressed differently)
```

The problem is that the model is generating the answer token by token. Without
intermediate reasoning steps in the context, it has no "scratch space" to work through
the problem.

---

### 3.2 Zero-shot CoT — "Think step by step"

The simplest CoT technique — just ask the model to reason before answering.

```python
def zero_shot_cot(problem: str) -> str:
    """
    Adding 'Let's think step by step' dramatically improves multi-step reasoning.
    This was discovered empirically by Kojima et al. (2022) — a single phrase
    that unlocks much better performance on reasoning tasks.
    """
    
    prompt = f"""{problem}

Let's think step by step."""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


# Example: math problem
problem = """If a train travels at 60 mph and needs to cover 150 miles,
but makes a 30-minute stop halfway, how long is the total journey?"""

print(zero_shot_cot(problem))
# Model reasons:
# "Step 1: Distance is 150 miles at 60 mph = 2.5 hours of travel time.
#  Step 2: There's a 30-minute (0.5 hour) stop.
#  Step 3: Total = 2.5 + 0.5 = 3 hours."
```

**Why this works from an ML perspective:** Each generated token becomes part of the
context for generating subsequent tokens. By generating reasoning steps first, the model
creates intermediate results in its context window. These intermediate tokens allow it to
attend to the correct information when generating the final answer. The model is
effectively using its generation space as working memory.

---

### 3.3 Few-shot CoT — providing reasoning examples

Even more powerful: show the model examples where reasoning is explicitly written out.

```python
def few_shot_cot_classifier(text: str) -> str:
    """
    Few-shot CoT for a classification task.
    We show reasoning chains, not just labels.
    The model learns to reason before classifying.
    """
    
    prompt = f"""Determine if a customer email should be escalated to a human agent.
Reason through it before deciding.

---
Email: "Hi, I ordered item #1234 two weeks ago and it still hasn't arrived. 
The tracking shows it hasn't moved in 10 days."
Reasoning: The customer has been waiting two weeks with no movement in tracking.
This suggests a lost package — a situation that requires manual investigation
and likely compensation. The customer is probably frustrated.
Decision: ESCALATE

---
Email: "Hi, can you tell me what your return policy is?"
Reasoning: This is a simple information request. The return policy is documented
and can be answered automatically without human intervention.
Decision: DO NOT ESCALATE

---
Email: "I've been a customer for 10 years and this is the worst experience I've ever had.
Your support team has ignored my last three messages about my $800 refund."
Reasoning: Long-standing customer, high-value refund ($800), multiple ignored contacts.
This is a high-frustration, high-stakes situation. Automated response risks losing
a loyal customer entirely.
Decision: ESCALATE

---
Email: "{text}"
Reasoning:"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300,
    )
    return response.choices[0].message.content
```

---

### 3.4 Separating reasoning from the final answer

In production, you often want to capture the reasoning separately from the final answer.
The cleanest way is a two-step pattern:

```python
def cot_with_extracted_answer(problem: str) -> dict:
    """
    Step 1: Generate reasoning chain
    Step 2: Extract just the final answer from the reasoning
    
    This gives you both the reasoning (for logging/debugging) and
    the clean answer (for downstream use).
    """
    
    # Step 1: Reason through the problem
    reasoning_prompt = f"""{problem}

Think through this carefully, step by step. Work out each part before concluding."""
    
    reasoning_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": reasoning_prompt}],
        temperature=0,
    )
    reasoning = reasoning_response.choices[0].message.content
    
    # Step 2: Extract a clean final answer from the reasoning
    extraction_prompt = f"""Given this reasoning:

{reasoning}

What is the final answer? Give only the answer, no explanation."""
    
    answer_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": extraction_prompt}],
        temperature=0,
        max_tokens=50,
    )
    answer = answer_response.choices[0].message.content.strip()
    
    return {"reasoning": reasoning, "answer": answer}
```

---

### 3.5 Self-consistency — multiple paths, majority vote

Self-consistency generates multiple independent reasoning chains (using higher
temperature so they differ) and takes a majority vote on the final answer. It is one of
the most reliable accuracy improvements for reasoning tasks.

```python
from collections import Counter

def self_consistent_answer(problem: str, n_samples: int = 5) -> dict:
    """
    Generate n_samples independent reasoning chains at temperature > 0,
    extract the final answer from each, and take a majority vote.
    
    More reliable than a single CoT pass, especially for math and logic.
    High confidence = all samples agree.
    Low confidence = samples are split (might need a better prompt or model).
    """
    
    prompt = f"""{problem}

Think step by step and give your final answer at the end on a new line 
starting with "Answer:"."""
    
    answers = []
    reasonings = []
    
    for i in range(n_samples):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # non-zero so each sample takes a different path
        )
        
        full_response = response.choices[0].message.content
        reasonings.append(full_response)
        
        # Extract the answer line
        for line in full_response.split('\n'):
            if line.strip().startswith("Answer:"):
                answer = line.replace("Answer:", "").strip()
                answers.append(answer)
                break
    
    # Count how many times each answer appears
    vote_counts = Counter(answers)
    majority_answer, majority_count = vote_counts.most_common(1)[0]
    confidence = majority_count / n_samples
    
    return {
        "answer": majority_answer,
        "confidence": confidence,
        "vote_breakdown": dict(vote_counts),
        "all_reasonings": reasonings,
    }


# Usage
result = self_consistent_answer(
    "A store offers a 20% discount on a $150 jacket. "
    "Then there's an additional 10% off the discounted price. "
    "What is the final price?",
    n_samples=5
)
print(f"Answer:     {result['answer']}")
print(f"Confidence: {result['confidence']:.0%}")
print(f"Votes:      {result['vote_breakdown']}")
# High confidence → most or all samples agree → trustworthy answer
# Low confidence → samples disagree → prompt needs improvement or use stronger model
```

**When to use self-consistency:**
- Mathematical calculations with clear right/wrong answers
- Logic puzzles
- Code generation where correctness matters
- Any case where you need higher reliability than a single sample

**Cost consideration:** `n_samples=5` means 5× the token cost. Use it for high-stakes
decisions, not bulk processing.

---

## 4. Structured Outputs & JSON Mode

### 4.1 The reliability problem

In production code, you need to parse the model's output programmatically. The problem:

```python
# What you ask for:
prompt = "Extract the name and age from: 'John Smith is 32 years old.'"
         "Return as JSON."

# What you might get (any of these):
# Option 1 (ideal):
# {"name": "John Smith", "age": 32}

# Option 2 (markdown-wrapped):
# ```json
# {"name": "John Smith", "age": 32}
# ```

# Option 3 (different key names):
# {"full_name": "John Smith", "age_in_years": "32"}

# Option 4 (with commentary):
# Here is the extracted information:
# {"name": "John Smith", "age": 32}
# Note: The age is given as an integer.

# Option 5 (wrong type):
# {"name": "John Smith", "age": "32"}  ← age is a string, not int
```

Any of these breaks `json.loads()` or your downstream code. You need a reliable way to
get exactly the structure and types you want.

---

### 4.2 OpenAI JSON mode

The simplest fix: tell OpenAI to return valid JSON, no markdown, no prose.

```python
import json

def extract_with_json_mode(text: str) -> dict:
    """
    JSON mode guarantees valid JSON output.
    Limitation: you cannot specify the schema — you get valid JSON,
    but the keys and types might not match what you want.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a data extraction engine. Always respond with valid JSON."
            },
            {
                "role": "user",
                "content": f"Extract the person's name and age from this text: '{text}'\n"
                           f"Return JSON with keys: name (string), age (integer)."
            }
        ],
        response_format={"type": "json_object"},  # ← enables JSON mode
        temperature=0,
    )
    
    raw_json = response.choices[0].message.content
    return json.loads(raw_json)  # guaranteed to not throw JSONDecodeError


# Usage
result = extract_with_json_mode("Dr. Sarah Chen is 45 years old and lives in Boston.")
print(result)  # → {"name": "Dr. Sarah Chen", "age": 45}
print(type(result["age"]))  # → <class 'int'>  (if model got the type right)
```

**Limitation of JSON mode:** It guarantees syntactically valid JSON but not a specific
schema. The model might use different key names, wrong types, or omit fields. You need
validation on top.

---

### 4.3 OpenAI Structured Outputs (schema enforcement)

A more recent feature: pass a JSON schema derived from a Pydantic model, and OpenAI
guarantees the output matches the schema exactly.

```python
from pydantic import BaseModel
from typing import Optional

class PersonInfo(BaseModel):
    name: str
    age: int
    city: Optional[str] = None
    occupation: Optional[str] = None

def extract_with_schema(text: str) -> PersonInfo:
    """
    Uses OpenAI Structured Outputs to enforce the exact schema.
    The response is guaranteed to match PersonInfo — no validation needed.
    """
    
    from openai.lib._parsing import type_to_response_format_param
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract person information from the text."},
            {"role": "user", "content": text}
        ],
        response_format=PersonInfo,  # ← pass the Pydantic class directly
    )
    
    # response.choices[0].message.parsed is already a PersonInfo instance
    return response.choices[0].message.parsed


# Usage
info = extract_with_schema(
    "Dr. Sarah Chen is 45 years old, works as a neuroscientist, and lives in Boston."
)
print(info.name)       # → "Dr. Sarah Chen"
print(info.age)        # → 45  (guaranteed int, not "45")
print(info.city)       # → "Boston"
print(info.occupation) # → "neuroscientist"
```

---

### 4.4 Pydantic for validation

Even without structured outputs, Pydantic lets you validate and coerce model outputs.
It is the backbone of reliable data extraction.

```python
from pydantic import BaseModel, Field, validator, field_validator
from typing import Optional, List
from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class ProductReview(BaseModel):
    product_name: str = Field(description="Name or description of the product")
    rating: int = Field(ge=1, le=5, description="Rating from 1 to 5")
    sentiment: Sentiment
    pros: List[str] = Field(default_factory=list, description="Positive aspects mentioned")
    cons: List[str] = Field(default_factory=list, description="Negative aspects mentioned")
    summary: str = Field(max_length=200, description="One-sentence summary")
    
    @field_validator('rating', mode='before')
    @classmethod
    def coerce_rating(cls, v):
        # Handle if model returns "4/5" or "4 out of 5" or "four"
        if isinstance(v, str):
            v = v.split('/')[0].split(' ')[0]
            try:
                return int(v)
            except ValueError:
                word_to_num = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}
                return word_to_num.get(v.lower(), 3)
        return v


# Pydantic validates on instantiation
try:
    review = ProductReview(
        product_name="Wireless Headphones",
        rating=6,       # invalid! must be 1-5
        sentiment="positive",
        pros=["great sound"],
        cons=[],
        summary="Good product overall"
    )
except Exception as e:
    print(f"Validation error: {e}")
    # → "rating: Input should be less than or equal to 5"
```

**Pydantic `Field` parameters you'll use constantly:**

| Parameter | What it does | Example |
|---|---|---|
| `description` | Hints for the model about what this field means | `Field(description="Email subject line")` |
| `default` | Default value if field is absent | `Field(default=None)` |
| `default_factory` | Default for mutable types like lists | `Field(default_factory=list)` |
| `ge`, `le` | Greater/less than or equal (for numbers) | `Field(ge=0, le=100)` |
| `min_length`, `max_length` | Length bounds for strings | `Field(max_length=500)` |
| `pattern` | Regex pattern the string must match | `Field(pattern=r'^\d{4}-\d{2}-\d{2}$')` |

---

## 5. Function Calling & Tool Use

### 5.1 What function calling actually is

Function calling is widely misunderstood. The model does **not** call your function. The
model generates a structured JSON object that describes which function to call and with
what arguments. Your code then decides whether to actually call the function.

The flow:

```
1. You define a function schema (name, description, parameters as JSON Schema)
2. You send a prompt + the schema to the API
3. The model decides it wants to "call" the function and generates arguments
4. The API returns those arguments as JSON — the model stops here
5. Your code receives the JSON, validates it, and calls the actual function
6. (Optional) You send the result back to the model for a follow-up response
```

In Phase 02, we stop at step 5 — we use function calling purely for **structured
extraction**. In Phase 05 (Agents), you will complete the loop by returning results to
the model.

---

### 5.2 Raw function calling with OpenAI

```python
import json

# Step 1: Define the function schema using JSON Schema format
# This tells the model what arguments the function expects
extract_email_tool = {
    "type": "function",
    "function": {
        "name": "extract_email_data",
        "description": "Extract structured information from an email",
        "parameters": {
            "type": "object",
            "properties": {
                "sender_name": {
                    "type": "string",
                    "description": "Full name of the email sender"
                },
                "intent": {
                    "type": "string",
                    "enum": ["request", "complaint", "update", "question", "approval"],
                    "description": "Primary intent of the email"
                },
                "action_items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of action items or tasks mentioned"
                },
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"],
                    "description": "Overall emotional tone"
                },
                "urgency": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "How urgent is this email"
                }
            },
            "required": ["intent", "action_items", "sentiment", "urgency"]
        }
    }
}

def extract_email_raw(email_text: str) -> dict:
    """Extract structured data from an email using raw function calling."""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an email analysis assistant. Extract structured information."
            },
            {
                "role": "user",
                "content": f"Analyze this email:\n\n{email_text}"
            }
        ],
        tools=[extract_email_tool],
        tool_choice={"type": "function", "function": {"name": "extract_email_data"}},
        # tool_choice="auto" lets the model decide whether to call a function
        # tool_choice={"type": "function", ...} forces this specific function
    )
    
    message = response.choices[0].message
    
    # Check if the model called a function
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        
        # The model generated a JSON string of arguments
        arguments_str = tool_call.function.arguments
        arguments = json.loads(arguments_str)
        
        return arguments
    
    # If no function was called, model responded with text instead
    return {"error": "Model did not call the function", "text": message.content}


# Test it
email = """
Hi team,

Just wanted to follow up on the Q3 budget report we discussed last week. 
Can you please have the draft ready by Thursday EOD? Also, we need to schedule
a review meeting before the board presentation on Monday.

This is pretty urgent as the board is expecting the numbers.

Thanks,
Michael
"""

result = extract_email_raw(email)
print(result)
# → {
#     "sender_name": "Michael",
#     "intent": "request",
#     "action_items": ["Prepare Q3 budget report draft by Thursday EOD",
#                      "Schedule review meeting before Monday"],
#     "sentiment": "neutral",
#     "urgency": "high"
#   }
```

---

### 5.3 Claude tool use (equivalent)

Anthropic calls the same concept "tool use" with slightly different syntax:

```python
from anthropic import Anthropic

anthropic_client = Anthropic()

# Claude's tool definition format
extract_tool = {
    "name": "extract_email_data",
    "description": "Extract structured information from an email",
    "input_schema": {  # ← Claude calls it input_schema, not parameters
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": ["request", "complaint", "update", "question", "approval"],
            },
            "action_items": {
                "type": "array",
                "items": {"type": "string"},
            },
            "urgency": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            }
        },
        "required": ["intent", "action_items", "urgency"]
    }
}

def extract_email_claude(email_text: str) -> dict:
    response = anthropic_client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1024,
        tools=[extract_tool],
        tool_choice={"type": "tool", "name": "extract_email_data"},
        messages=[
            {"role": "user", "content": f"Analyze this email:\n\n{email_text}"}
        ]
    )
    
    # Find the tool_use block in the content
    for block in response.content:
        if block.type == "tool_use":
            return block.input  # already a dict, not a JSON string
    
    return {}
```

---

### 5.4 The JSON Schema format

Function calling uses JSON Schema to define the structure of arguments. These are the
types and keywords you'll use most:

```python
# Type: string
{"type": "string", "description": "A text field"}

# Type: integer
{"type": "integer", "minimum": 0, "maximum": 100}

# Type: number (float)
{"type": "number"}

# Type: boolean
{"type": "boolean"}

# Type: array of strings
{"type": "array", "items": {"type": "string"}}

# Type: array of objects
{
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "score": {"type": "number"}
        },
        "required": ["name"]
    }
}

# Enum (restricted set of values)
{"type": "string", "enum": ["positive", "negative", "neutral"]}

# Optional field (not in "required")
# Just don't include the field name in the "required" array
```

---

## 6. The `instructor` Library

### 6.1 Why instructor exists

Raw function calling works but involves boilerplate: write JSON schema, parse JSON
response, validate types. The `instructor` library removes all of this. You define a
Pydantic model, and instructor handles everything else.

```bash
pip install instructor
```

**What instructor does internally:**
1. Takes your Pydantic model and converts it to JSON Schema
2. Sends it to the API as a function/tool definition
3. Parses the JSON response back into your Pydantic model
4. Validates all fields against Pydantic constraints
5. If validation fails, automatically retries with an error message

---

### 6.2 instructor basics

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

# Patch the OpenAI client — this adds instructor's magic
client = instructor.from_openai(OpenAI())

# Define what you want to extract
class EmailData(BaseModel):
    sender_name: Optional[str] = Field(default=None, description="Full name of sender")
    intent: str = Field(description="Primary intent: request, complaint, update, question, approval")
    action_items: List[str] = Field(default_factory=list, description="Tasks or follow-ups needed")
    sentiment: str = Field(description="positive, negative, or neutral")
    urgency: str = Field(description="high, medium, or low")
    summary: str = Field(description="One sentence summary of the email")

# Use it — pass response_model to get a Pydantic instance back
def extract_email(email_text: str) -> EmailData:
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Extract structured information from the email."
            },
            {
                "role": "user",
                "content": email_text
            }
        ],
        response_model=EmailData,   # ← instructor's key parameter
        temperature=0,
    )

# Usage
email = """
Hi,
I still haven't received my refund from 3 weeks ago (Order #45231, $89.99).
I've contacted support twice with no response. This is unacceptable.
Please resolve this immediately or I'll dispute the charge.
- Angry Customer
"""

result = extract_email(email)

# result is a validated EmailData instance
print(result.sender_name)    # → None (no name given)
print(result.intent)         # → "complaint"
print(result.urgency)        # → "high"
print(result.action_items)   # → ["Process refund for order #45231 ($89.99)"]
print(result.summary)        # → "Customer demanding refund for 3-week-old order..."
print(result.model_dump())   # → dict representation
print(result.model_dump_json(indent=2))  # → formatted JSON string
```

---

### 6.3 Nested Pydantic models

instructor handles nested models seamlessly — any complexity that Pydantic supports,
instructor can extract:

```python
class ActionItem(BaseModel):
    task: str = Field(description="The specific action to take")
    assignee: Optional[str] = Field(default=None, description="Who should do this")
    deadline: Optional[str] = Field(default=None, description="Due date if mentioned")
    priority: str = Field(description="high, medium, or low")

class MeetingNotes(BaseModel):
    meeting_title: str
    date: Optional[str] = None
    attendees: List[str] = Field(default_factory=list)
    key_decisions: List[str] = Field(description="Decisions made in the meeting")
    action_items: List[ActionItem] = Field(description="Tasks assigned during the meeting")
    next_meeting: Optional[str] = Field(default=None, description="When is the next meeting")

def extract_meeting_notes(notes_text: str) -> MeetingNotes:
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Extract meeting information:\n\n{notes_text}"}],
        response_model=MeetingNotes,
        temperature=0,
    )

# Test
notes = """
Q3 Planning Meeting — October 14, 2024
Attendees: Alice (PM), Bob (Engineering), Carol (Design)

We decided to move the product launch from November to December to allow
more time for testing. Bob will have the backend complete by Oct 28.
Carol needs to finalize the UI mockups by Oct 21. Alice will update the
stakeholder presentation and share it by Friday.

Next meeting: October 21 at 2pm.
"""

result = extract_meeting_notes(notes)
print(result.meeting_title)     # → "Q3 Planning Meeting"
print(result.attendees)         # → ["Alice", "Bob", "Carol"]
print(result.key_decisions)     # → ["Move product launch to December"]
for item in result.action_items:
    print(f"  [{item.priority}] {item.task} ({item.assignee}, by {item.deadline})")
```

---

### 6.4 instructor with Claude

instructor works with Anthropic too — just patch a different client:

```python
import instructor
from anthropic import Anthropic

# Patch the Anthropic client
anthropic_client = instructor.from_anthropic(Anthropic())

def extract_with_claude(text: str, output_model) -> BaseModel:
    return anthropic_client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": text}],
        response_model=output_model,
    )
```

---

### 6.5 Validation and automatic retry

One of instructor's most powerful features: if the model returns data that fails Pydantic
validation, it automatically retries with the error message.

```python
from pydantic import field_validator

class StrictRating(BaseModel):
    product: str
    rating: int = Field(ge=1, le=5)
    review_text: str = Field(min_length=10)
    
    @field_validator('review_text')
    @classmethod
    def must_be_specific(cls, v):
        generic_phrases = ["good product", "nice item", "okay"]
        if any(phrase in v.lower() for phrase in generic_phrases):
            raise ValueError("Review must be specific, not generic")
        return v

# If the model returns rating=0 or a generic review,
# instructor automatically retries with the validation error as feedback.
# Default: 3 retries. Configure with max_retries parameter.
result = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Rate this product: wireless headphones"}],
    response_model=StrictRating,
    max_retries=3,  # retry up to 3 times on validation failure
)
```

---

## 7. System Prompts & Prompt Templates

### 7.1 The anatomy of an effective system prompt

The system prompt sets the model's behavior for the entire conversation. A weak system
prompt produces inconsistent outputs. A strong one produces reliable, production-ready
behavior.

**Structure of a strong system prompt:**

```
1. Role definition — who/what the model is
2. Context — what kind of inputs to expect
3. Instructions — exactly what to do
4. Constraints — what not to do
5. Output format — exactly how to format the response
```

```python
# WEAK system prompt — vague, no constraints, no format
weak_system = "You are a helpful assistant that extracts information."

# STRONG system prompt — specific role, clear instructions, explicit constraints
strong_system = """You are a precise data extraction engine specialized in analyzing
customer support emails.

Your task: Extract structured information from each email you receive.

Instructions:
- Extract only information explicitly stated in the email
- Do not infer or assume information that is not present
- If a field cannot be determined from the email, use null
- Classify urgency based on: explicit time pressure, financial stakes, and customer frustration level

Constraints:
- Never add your own commentary or opinions
- Never ask clarifying questions
- Always extract action items as separate, actionable tasks (not vague notes)

Output: Respond only with the requested structured format. No preamble."""
```

---

### 7.2 System prompt patterns

**The Persona Pattern** — Give the model a specific identity:

```python
persona_system = """You are a senior software engineer at a Fortune 500 company
with 15 years of experience in Python and distributed systems.
You write concise, production-quality code with proper error handling.
You prefer simplicity over cleverness.
You always consider edge cases and mention potential issues."""
```

**The Output Format Pattern** — Specify format exactly:

```python
format_system = """You analyze technical documents and produce summaries.
Format your response exactly as follows:

SUMMARY (2-3 sentences):
[summary here]

KEY TECHNICAL CONCEPTS (bullet points):
- [concept 1]
- [concept 2]

COMPLEXITY LEVEL: [Beginner / Intermediate / Advanced]

RECOMMENDED BACKGROUND: [What the reader should know first]"""
```

**The Constraint Pattern** — Specify what NOT to do:

```python
constraint_system = """You classify customer support tickets into categories.

Categories: billing, technical, account, shipping, general

Rules:
- Output ONLY the category name, nothing else
- If multiple categories apply, choose the primary one
- Do not explain your reasoning
- Do not say "I think" or "The ticket seems to be about"
- Do not add punctuation after the category name"""
```

---

### 7.3 Jinja2 for dynamic prompt templates

As your prompts get more complex — with conditional sections, loops, and variable
substitution — f-strings become unmaintainable. Jinja2 is the standard templating
engine for this.

```bash
pip install jinja2
```

**Why Jinja2 over f-strings:**

| Situation | f-string | Jinja2 |
|---|---|---|
| Simple variable substitution | `f"Analyze {text}"` | `{{ text }}` |
| Conditional sections | Requires Python logic in the string | `{% if user_type == 'vip' %}...{% endif %}` |
| Loops (e.g., few-shot examples) | Clunky concatenation | `{% for ex in examples %}...{% endfor %}` |
| Template reuse | Copy-paste | Template inheritance |
| Non-developer editing | Must touch Python code | Separate .j2 files |

---

### 7.4 Jinja2 basics

```python
from jinja2 import Template, Environment, FileSystemLoader

# Simple variable substitution
template = Template("Classify this text as {{ categories|join(', ') }}: '{{ text }}'")
rendered = template.render(
    categories=["positive", "negative", "neutral"],
    text="The product is decent but overpriced."
)
print(rendered)
# → "Classify this text as positive, negative, neutral: 'The product is decent...'"
```

**Jinja2 in prompt templates:**

```python
from jinja2 import Template

# Conditional sections
email_template = Template("""Analyze this email and extract structured information.

{% if strict_mode %}
IMPORTANT: Only extract information explicitly stated. Never infer.
{% endif %}

{% if examples %}
Examples of good extractions:
{% for ex in examples %}
---
Email: "{{ ex.input }}"
Extraction: {{ ex.output }}
{% endfor %}
---
{% endif %}

Now extract from this email:
"{{ email_text }}"

{% if output_format == 'json' %}
Respond with JSON only, no prose.
{% elif output_format == 'markdown' %}
Respond with a markdown formatted summary.
{% endif %}""")

rendered = email_template.render(
    strict_mode=True,
    examples=[
        {
            "input": "Can you tell me your return policy?",
            "output": '{"intent": "question", "urgency": "low"}'
        }
    ],
    email_text="I've been waiting 3 weeks for my refund on order #1234!",
    output_format="json"
)
print(rendered)
```

---

### 7.5 Reusable prompt template system

In production, you store templates in files and load them dynamically. Here is the
pattern used in the project:

```python
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

class PromptLibrary:
    """
    Manages a collection of Jinja2 prompt templates.
    Templates are stored as .j2 files in a templates/ directory.
    """
    
    def __init__(self, template_dir: str = "templates"):
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,     # removes newline after block tags
            lstrip_blocks=True,   # removes leading whitespace before block tags
        )
        self._inline_templates = {}
    
    def register(self, name: str, template_str: str):
        """Register an inline template (not from file)."""
        self._inline_templates[name] = Template(template_str)
    
    def render(self, template_name: str, **kwargs) -> str:
        """Render a template by name with given variables."""
        if template_name in self._inline_templates:
            return self._inline_templates[template_name].render(**kwargs)
        
        template = self.env.get_template(f"{template_name}.j2")
        return template.render(**kwargs)


# Usage
library = PromptLibrary()

library.register("email_extraction", """
You are an email analysis assistant.
{% if persona %}Act as: {{ persona }}{% endif %}

Extract structured data from this email:
{{ email_text }}

{% if fields %}Focus on these fields: {{ fields|join(', ') }}{% endif %}
""")

prompt = library.render(
    "email_extraction",
    email_text="Please send the report by Friday.",
    persona="a senior executive assistant",
    fields=["intent", "deadline", "action_items"]
)
```

---

### 7.6 Few-shot examples in templates

This is a common production pattern — store your few-shot examples in a list and render
them into the template:

```python
from jinja2 import Template

few_shot_template = Template("""
Classify customer support tickets into one of these categories:
billing, technical, account, shipping, general

{% for example in examples %}
Ticket: "{{ example.text }}"
Category: {{ example.label }}

{% endfor %}
Ticket: "{{ ticket }}"
Category:""")

examples = [
    {"text": "My invoice shows the wrong amount",  "label": "billing"},
    {"text": "The app keeps crashing on startup",  "label": "technical"},
    {"text": "How do I change my password?",        "label": "account"},
    {"text": "My order hasn't arrived in 2 weeks", "label": "shipping"},
]

prompt = few_shot_template.render(
    examples=examples,
    ticket="I was charged twice for the same subscription"
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
    max_tokens=10,
)
print(response.choices[0].message.content.strip())  # → "billing"
```

---

### 7.7 Prompt template best practices

**Be specific with field descriptions:**

```python
# BAD — ambiguous
class Extract(BaseModel):
    date: str

# GOOD — the model knows exactly what format you want
class Extract(BaseModel):
    date: str = Field(description="Date in ISO 8601 format (YYYY-MM-DD)")
```

**Use Enums to constrain choices:**

```python
from enum import Enum

class Category(str, Enum):
    BILLING   = "billing"
    TECHNICAL = "technical"
    ACCOUNT   = "account"
    SHIPPING  = "shipping"
    GENERAL   = "general"

class Ticket(BaseModel):
    category: Category  # model can only return one of these 5 values
```

**Separate instructions from data with clear markers:**

```python
# Using XML-style tags to separate context from instructions
prompt = """<instructions>
Extract the invoice data from the document below.
Only extract fields that are explicitly present.
</instructions>

<document>
{document_text}
</document>

Extract:"""
```

---

## 8. Key Takeaways

After completing this phase, you understand:

1. **Start with zero-shot.** For GPT-4 class models, just asking clearly often works.
   Only add examples when output is inconsistent or wrong.

2. **Few-shot calibrates format, not just accuracy.** Examples show the model exactly
   what output structure you want, not just what the right answer is.

3. **CoT is "Let's think step by step" — but understand why it works.** Generated
   reasoning tokens become context for subsequent tokens. The model attends over its
   own reasoning when producing the final answer.

4. **JSON mode prevents parse errors; Pydantic prevents type errors.** Use both.
   JSON mode ensures the output is parseable; Pydantic validates that it is correct.

5. **Function calling is structured extraction.** The model generates JSON arguments
   for a function. Your code receives those arguments. Nothing is "called" automatically.

6. **instructor is function calling minus the boilerplate.** Define a Pydantic model,
   get back a validated instance. Use it for all structured extraction tasks.

7. **System prompts set behavior; Jinja2 templates scale prompts.** Use the constraint
   pattern to restrict output format. Use Jinja2 when your prompts have conditional
   sections or need to embed variable numbers of examples.

8. **Field descriptions are instructions.** The `description=` in `Field()` is sent to
   the model as part of the schema. Write it as if you are instructing the model what
   to put there.

---

## 9. Practice Exercises

### Exercise 1 — Few-shot Format Calibration (Easy)
Take a zero-shot extraction prompt that returns inconsistent key names (sometimes
"full_name", sometimes "name", sometimes "person_name"). Add three few-shot examples
that demonstrate exactly the key names and types you want. Measure how often the output
matches your expected format before and after.

### Exercise 2 — CoT for Document Routing (Medium)
Build a ticket routing system. Given a customer support message, use Chain-of-Thought
to first reason about: the severity of the issue, what department can resolve it, and
whether it needs immediate attention. Then output a structured routing decision. Compare
accuracy with and without CoT reasoning.

### Exercise 3 — Multi-type Extractor (Medium)
Extend the `DataExtractor` class to auto-detect document type (email, job posting,
news article, meeting notes) from the text, then apply the appropriate Pydantic model.
The output should always include the detected `document_type` plus the extracted fields
for that type.

```python
# Target interface
extractor = DataExtractor()
result = extractor.auto_extract(some_text)
# result.document_type → "email"
# result.data → EmailData instance
```

### Exercise 4 — Prompt Template Library (Hard)
Build a `PromptLibrary` class that:
- Loads templates from a `templates/` directory as `.j2` files
- Supports template inheritance (a base template that other templates extend)
- Has a `render_few_shot(name, examples, query)` convenience method
- Caches rendered templates to avoid re-rendering identical prompts
- Has a `validate(name)` method that checks all required variables are present

---

*Next: Phase 03 — LangChain & Orchestration*  
*You will build reusable LLM pipelines with chains, add conversation memory,
process documents at scale, and trace every call in LangSmith.*
