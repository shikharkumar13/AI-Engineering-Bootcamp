# Phase 01 — LLM API Foundations

> **Who this is for:** ML/DL engineers who understand neural networks and want to start
> building products with large language models.  
> **What you'll learn:** How to call OpenAI, Claude, and Gemini APIs; stream responses;
> count tokens and track costs; retry reliably; and run calls in parallel.

---

## Table of Contents

1. [The Big Picture — Why LLM APIs?](#1-the-big-picture--why-llm-apis)
2. [Setup — Environment & API Keys](#2-setup--environment--api-keys)
3. [OpenAI, Claude & Gemini SDKs](#3-openai-claude--gemini-sdks)
4. [Chat Completions & Streaming](#4-chat-completions--streaming)
5. [Token Management & Pricing](#5-token-management--pricing)
6. [Error Handling & Exponential Backoff](#6-error-handling--exponential-backoff)
7. [Async & Parallel API Calls](#7-async--parallel-api-calls)
8. [Key Takeaways](#8-key-takeaways)
9. [Practice Exercises](#9-practice-exercises)

---

## 1. The Big Picture — Why LLM APIs?

As someone who has trained neural networks, you understand the relationship between model
size and capability. But consider the scale gap:

| Model | Approximate Parameters | VRAM to Run Inference |
|---|---|---|
| BERT-base | 110M | ~1 GB |
| GPT-2 (large) | 774M | ~3 GB |
| LLaMA 3 (8B) | 8B | ~16 GB |
| LLaMA 3 (70B) | 70B | ~140 GB |
| GPT-4o | ~200B (estimated) | ~400 GB |
| Claude Sonnet 5 | Undisclosed | Anthropic's infrastructure |

The models that actually perform well in production (GPT-4o, Claude Sonnet 5, Gemini
2.5 Pro) are either too large to run locally or are entirely proprietary. You cannot
download them. The only way to use them is through their APIs.

**What an LLM API actually is:**  
It is the same neural network you already understand: transformer with attention,
embedding layers, softmax output, except it is already trained, already deployed on
clusters of H100s, and sitting behind an HTTP endpoint. Your code sends text in, text
comes back. You pay per token. No GPU required on your side.

**Why this matters for AI engineering:**  
The skill of an AI engineer is not retraining these models from scratch. It is knowing
how to call them effectively, combine them with tools and data, and ship products around
them. Everything in this roadmap (RAG, agents, fine-tuning) sits on top of the
foundation you are building right now.

---

## 2. Setup — Environment & API Keys

### 2.1 What is an API key?

An API key is a secret string that identifies you to the API provider. When you make a
request, the server looks up your key, checks your account, and bills you for the tokens
you use. It is exactly like a password.

**The most common mistake new AI engineers make:** hardcoding the API key directly in
their Python file and accidentally pushing it to GitHub. Someone scrapes it within
minutes and runs up thousands of dollars in API charges on your account.

**The correct approach:** Store secrets in a `.env` file that never gets committed.

### 2.2 Project setup

```bash
# Install all libraries for Phase 01
pip install openai anthropic google-generativeai python-dotenv httpx tenacity tiktoken
```

Copy the provided `.env.example` to `.env` and fill in your real keys. Never commit
`.env` itself:

```bash
cp .env.example .env
```

```bash
# .env — NEVER commit this file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
```

Create a `.gitignore` so Git ignores it:

```bash
# .gitignore
.env
__pycache__/
*.pyc
.DS_Store
```

### 2.3 Why python-dotenv?

`python-dotenv` reads your `.env` file and loads its contents into `os.environ`, the
dictionary Python uses to store environment variables. After calling `load_dotenv()`,
you can access your keys via `os.getenv("OPENAI_API_KEY")`. The SDK clients do this
automatically, so you rarely need to pass the key explicitly.

```python
from dotenv import load_dotenv
import os

load_dotenv()  # reads .env and populates os.environ

key = os.getenv("OPENAI_API_KEY")  # now available everywhere
```

### 2.4 Project file structure

A clean structure for Phase 01 work:

```
Phase_1 LLM APIs/
├── .env                  ← API keys (never committed)
├── .env.example          ← template you copy to .env
├── .gitignore            ← includes .env
├── requirements.txt      ← pip dependencies
├── llm_client.py         ← your universal client (the project)
└── demo.py               ← usage examples
```

---

## 3. OpenAI, Claude & Gemini SDKs

### 3.1 What is an SDK and why use one?

SDK stands for Software Development Kit. In this context, it is a Python library that
wraps raw HTTP calls to the API. Without an SDK, calling GPT-4o looks like this:

```python
# Without an SDK — verbose and error-prone
import requests, json, os

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    },
    json={
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hello"}],
    }
)
data = response.json()
text = data["choices"][0]["message"]["content"]
```

With the SDK:

```python
# With the SDK — clean and handles auth, errors, retries automatically
from openai import OpenAI
client = OpenAI()  # reads OPENAI_API_KEY from environment automatically
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)
text = response.choices[0].message.content
```

The SDK handles authentication, JSON serialization, response parsing, and raises proper
Python exceptions on errors instead of raw HTTP status codes.

---

### 3.2 The Messages Format — The Most Important Concept in This Phase

Every major LLM API uses a "chat messages" format. Understanding this is fundamental to
everything else you will build.

```python
messages = [
    {
        "role": "system",
        "content": "You are a concise Python tutor. Explain things clearly."
    },
    {
        "role": "user",
        "content": "What is a list comprehension?"
    },
    {
        "role": "assistant",
        "content": "A list comprehension is a compact way to create a list..."
    },
    {
        "role": "user",
        "content": "Can you show me an example?"
    }
]
```

There are three roles:

| Role | Who it represents | Purpose |
|---|---|---|
| `system` | You, the developer | Sets the model's persona, behavior, and constraints. Not shown to the end user. |
| `user` | The person using your app | The actual questions or instructions being sent |
| `assistant` | The model's previous replies | Previous turns in the conversation |

**Critical insight: models are stateless.**  
The model has no memory between API calls. It does not remember your previous message.
Every time you call the API, you send the *entire* conversation history. The model reads
all messages from the top, then generates the next assistant reply.

This means if you want a multi-turn conversation, you are responsible for appending each
new user message and assistant reply to the history list before the next call. This
pattern looks like:

```python
history = [
    {"role": "system", "content": "You are a helpful assistant."}
]

# Turn 1
history.append({"role": "user", "content": "What is backpropagation?"})
response = call_api(history)
history.append({"role": "assistant", "content": response})

# Turn 2
history.append({"role": "user", "content": "And what is gradient descent?"})
response = call_api(history)
history.append({"role": "assistant", "content": response})
# ... and so on
```

From your ML background: the model attends over all tokens in the messages list. The
context window limit (discussed in Section 5) is the maximum number of tokens the model
can process in a single forward pass, which includes all messages you send.

---

### 3.3 OpenAI SDK

```python
# file: 01_sdks.py (OpenAI section)
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# The client automatically picks up OPENAI_API_KEY from the environment
client = OpenAI()

def call_openai(prompt: str, system: str = "You are a helpful assistant.") -> str:
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",    # use this model while learning — it is cheap
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=300,         # maximum tokens the model can generate in its reply
        temperature=0.7,        # 0.0 = deterministic, 1.0 = creative, 2.0 = chaotic
    )
    
    # Understanding the response object
    # ─────────────────────────────────
    # response.choices[0].message.content   → the actual text reply
    # response.choices[0].finish_reason     → "stop" = ended normally
    #                                          "length" = hit max_tokens limit
    # response.usage.prompt_tokens          → tokens you sent (input)
    # response.usage.completion_tokens      → tokens in the reply (output)
    # response.usage.total_tokens           → sum of both
    # response.model                        → exact model that was used
    
    print(f"Finish reason : {response.choices[0].finish_reason}")
    print(f"Tokens used   : {response.usage.total_tokens} "
          f"({response.usage.prompt_tokens} in + {response.usage.completion_tokens} out)")
    
    return response.choices[0].message.content
```

**About `temperature`:**  
Temperature controls how the model samples from its output probability distribution.
At temperature 0, it always picks the highest-probability token (greedy decoding). At
temperature 1, it samples according to the raw probabilities. Higher values make the
distribution flatter, producing more surprising outputs. For factual tasks use 0-0.3;
for creative tasks use 0.7-1.0.

**About `max_tokens`:**  
This caps how many tokens the model generates. If the model reaches this limit, it stops
mid-sentence and `finish_reason` becomes `"length"` instead of `"stop"`. Set it high
enough for your expected response, but not so high that a runaway generation burns
through your budget.

---

### 3.4 Claude (Anthropic) SDK

```python
from anthropic import Anthropic

# Reads ANTHROPIC_API_KEY from environment
client_anthropic = Anthropic()

def call_claude(prompt: str, system: str = "You are a helpful assistant.") -> str:
    
    # KEY DIFFERENCE FROM OPENAI:
    # Claude separates the system prompt from the messages list.
    # It is a top-level parameter, not a message with role="system".
    
    response = client_anthropic.messages.create(
        model="claude-haiku-4-5-20251001",  # cheapest current Claude model
        max_tokens=300,                     # REQUIRED for Claude — not optional
        system=system,                      # ← separate parameter, not in messages
        messages=[
            {"role": "user", "content": prompt}
            # Note: no system message inside this list
        ]
    )
    
    # Response structure is different from OpenAI:
    # response.content[0].text        → the actual text reply
    # response.usage.input_tokens     → tokens you sent
    # response.usage.output_tokens    → tokens in the reply
    # response.stop_reason            → "end_turn" = normal, "max_tokens" = capped
    
    print(f"Stop reason : {response.stop_reason}")
    print(f"Tokens used : {response.usage.input_tokens} in + "
          f"{response.usage.output_tokens} out")
    
    return response.content[0].text
```

**Why `max_tokens` is required for Claude:**  
Anthropic made it a required parameter to force you to think about output length upfront,
which helps control costs. OpenAI makes it optional (it defaults to the model's maximum).

---

### 3.5 Gemini (Google) SDK

```python
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def call_gemini(prompt: str, system: str = "You are a helpful assistant.") -> str:
    
    # Gemini requires creating a model object first
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",   # fast and cheap; generous free tier
        system_instruction=system         # Gemini calls it system_instruction
    )
    
    response = model.generate_content(prompt)
    
    # Response structure:
    # response.text                              → the text reply
    # response.usage_metadata.total_token_count  → total tokens used
    # response.usage_metadata.prompt_token_count → input tokens
    
    print(f"Tokens used : {response.usage_metadata.total_token_count}")
    
    return response.text
```

---

### 3.6 Side-by-Side Comparison

The three SDKs have different naming conventions but the same underlying concept:

| Concept | OpenAI | Claude | Gemini |
|---|---|---|---|
| Client init | `OpenAI()` | `Anthropic()` | `genai.GenerativeModel()` |
| System prompt | Inside `messages[]` as `role: "system"` | Separate `system=` param | `system_instruction=` param |
| API call | `client.chat.completions.create()` | `client.messages.create()` | `model.generate_content()` |
| Get text | `response.choices[0].message.content` | `response.content[0].text` | `response.text` |
| Input tokens | `response.usage.prompt_tokens` | `response.usage.input_tokens` | `response.usage_metadata.prompt_token_count` |
| Max output | `max_tokens` (optional) | `max_tokens` (required) | `max_output_tokens` in config |

Your job as an AI engineer is to write an abstraction layer (like the `LLMClient` class
in the project files) so that the rest of your application never has to think about these
differences.

---

## 4. Chat Completions & Streaming

### 4.1 What is streaming and why does it matter?

Without streaming, the sequence is:

```
You send request → Model generates all tokens → You receive complete response
[0s]               [~3-10 seconds]              [~3-10 seconds]
```

The user sees nothing for several seconds, then the entire response appears at once.

With streaming, the model sends each token to you as it generates it:

```
You send request → Token 1 → Token 2 → Token 3 → ... → Last token
[0s]               [~0.3s]    [~0.35s]   [~0.4s]         [~3-10s]
```

The user sees text appearing almost immediately, which makes the application feel
dramatically more responsive even though the total time is the same.

**From your ML background:** Autoregressive generation already produces tokens one at a
time: the model samples one token, appends it to the context, runs another forward pass,
then samples the next token. Streaming simply forwards each token to the client as it is
generated, instead of buffering them all and sending at the end.

---

### 4.2 OpenAI Streaming

```python
import time
from openai import OpenAI

client = OpenAI()

def openai_stream(prompt: str) -> str:
    """Stream a response, printing each token as it arrives."""
    
    # stream=True is literally the only difference from a normal call
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    
    full_text = ""
    
    for chunk in stream:
        delta = chunk.choices[0].delta
        
        # delta.content is the new text in this chunk.
        # It can be None for metadata-only chunks (start/end of stream),
        # so always check before using it.
        if delta.content is not None:
            print(delta.content, end="", flush=True)
            # end=""   → don't add a newline after each chunk
            # flush=True → force the output buffer to flush immediately
            #              (without this, output might appear in batches)
            full_text += delta.content
    
    print()  # newline after the stream ends
    return full_text
```

**Understanding `chunk.choices[0].finish_reason`:**  
For most chunks during streaming, `finish_reason` is `None`. It becomes `"stop"` in the
final chunk when the model finishes naturally, or `"length"` if it hit `max_tokens`.
You can use this to detect the end of the stream if needed.

---

### 4.3 Claude Streaming

```python
from anthropic import Anthropic

client_anthropic = Anthropic()

def claude_stream(prompt: str) -> str:
    """Claude uses a context manager (with block) for streaming."""
    
    full_text = ""
    
    # The 'with' block ensures the connection is properly closed when done
    with client_anthropic.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        # stream.text_stream is a generator that yields strings (not chunk objects)
        # This is cleaner than OpenAI's approach — no need to extract .delta.content
        for text_piece in stream.text_stream:
            print(text_piece, end="", flush=True)
            full_text += text_piece
    
    print()
    return full_text
```

---

### 4.4 The Stream-and-Capture Pattern

In real applications, you need to do two things at once: display the stream to the user
and also capture the full text so you can use it later (save to a database, pass to the
next step in a pipeline, etc.).

```python
def stream_and_capture(prompt: str) -> str:
    """
    Stream for immediate user feedback while also building the full response string.
    This is the pattern you will use in almost every real application.
    """
    
    chunks = []
    
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content is not None:
            chunks.append(delta.content)
            print(delta.content, end="", flush=True)  # display live
    
    print()
    full_response = "".join(chunks)  # assemble for downstream use
    return full_response
```

---

### 4.5 Streaming vs Non-Streaming: Latency Comparison

```python
def compare_perceived_latency(prompt: str):
    """
    Demonstrates that streaming makes apps FEEL faster even though
    total generation time is the same.
    """
    
    # Non-streaming: time to FIRST character = total generation time
    t0 = time.time()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    total_time = time.time() - t0
    print(f"Non-streaming: waited {total_time:.2f}s before seeing anything")
    print(response.choices[0].message.content)
    
    # Streaming: time to FIRST character is much shorter
    t0 = time.time()
    first_token_time = None
    
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.time() - t0
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    print(f"\nStreaming: first character appeared in {first_token_time:.2f}s")
    # first_token_time is typically 0.2-0.5s, regardless of response length
```

---

## 5. Token Management & Pricing

### 5.1 What is a token?

A token is not a word. It is a sub-word unit that the model's vocabulary (the embedding
matrix you know from transformers) knows how to represent. The tokenization splits text
into these units before encoding.

Some examples with GPT-4o tokenization:

```
"Hello"             →  1 token   ["Hello"]
"hello"             →  1 token   ["hello"]
"extraordinary"     →  2 tokens  ["extra", "ordinary"]
"backpropagation"   →  3 tokens  ["back", "prop", "agation"]
"ChatGPT"           →  2 tokens  ["Chat", "GPT"]
" the"              →  1 token   [" the"]  ← space is part of the token
"100,000"           →  3 tokens  ["100", ",", "000"]
```
(counts above are from `tiktoken.encoding_for_model("gpt-4o-mini")`. Run it yourself and
you'll see splits like this vary by model family, so always measure rather than guess.)

**Rule of thumb:** approximately 4 characters = 1 token, or about ¾ of an English word
per token. So:

- 100 tokens ≈ 75 words
- 1,000 tokens ≈ 750 words ≈ 1.5 pages
- 1,000,000 tokens ≈ 750,000 words ≈ 1,500 pages

---

### 5.2 Why tokens matter

**Pricing:** API providers charge per token, with different prices for input (what you
send) and output (what the model generates). Output tokens typically cost more because
generation is more compute-intensive than encoding.

**Context window:** This is the maximum number of tokens the model can process in a
single call. It is the model's "working memory": the maximum size of the sequence it
can attend over.

| Model | Context Window | Approximate Word Limit |
|---|---|---|
| gpt-4o-mini | 128,000 tokens | ~96,000 words |
| gpt-4o | 128,000 tokens | ~96,000 words |
| claude-haiku-4-5 | 200,000 tokens | ~150,000 words |
| claude-sonnet-5 | 200,000 tokens | ~150,000 words |
| gemini-2.5-flash | 1,000,000 tokens | ~750,000 words |
| gemini-2.5-pro | 1,000,000 tokens | ~750,000 words |

The context window limit applies to the **total** of your input tokens plus the model's
output tokens. If you have a 128k context window and you send 100k tokens of input, the
model can only generate up to 28k tokens of output.

---

### 5.3 Pricing (approximate; check each provider's pricing page before budgeting)

Providers revise pricing and retire model IDs on their own schedule, often faster than a
document like this one gets updated. Treat the numbers below as illustrative of *relative*
cost (mini/flash/haiku tiers are roughly 10-20x cheaper than their flagship siblings), not
as numbers to hardcode into a production budget.

| Model | Input per 1M tokens | Output per 1M tokens |
|---|---|---|
| gpt-4o | $5.00 | $15.00 |
| **gpt-4o-mini** | **$0.15** | **$0.60** |
| claude-opus-5 | $5.00 | $25.00 |
| claude-sonnet-5 | $2.00 | $10.00 |
| **claude-haiku-4-5** | **$1.00** | **$5.00** |
| gemini-2.5-pro | $1.25 | $10.00 |
| **gemini-2.5-flash** | **$0.30** | **$2.50** |

**Cost example:** If you send a 500-token prompt and receive a 200-token response with
gpt-4o-mini:

```
Input cost:  500  / 1,000,000 × $0.15 = $0.000075
Output cost: 200  / 1,000,000 × $0.60 = $0.000120
Total cost:                              $0.000195  (~0.02 cents)
```

For learning and prototyping, stick to gpt-4o-mini, claude-haiku-4-5, and
gemini-2.5-flash. They are cheap enough that you can make thousands of calls without
worrying about cost.

---

### 5.4 Counting tokens with tiktoken

`tiktoken` is OpenAI's tokenization library. It gives you exact token counts for GPT
models without making an API call. For Claude and Gemini, it is approximate: the
tokenizers are similar but not identical.

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count how many tokens a string contains."""
    
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    
    print(f"Text      : '{text[:60]}'")
    print(f"Characters: {len(text)}")
    print(f"Tokens    : {len(tokens)}")
    print(f"Ratio     : {len(text)/len(tokens):.1f} chars per token")
    
    return len(tokens)

# Examples
count_tokens("The transformer architecture uses multi-head self-attention.")
# → 10 tokens

count_tokens("Hello world")
# → 2 tokens

count_tokens("supercalifragilisticexpialidocious")
# → 10 tokens (rare/complex words get split into many small tokens)
```

**Counting tokens in a messages array:**

When you send a messages list to the API, each message has hidden formatting tokens
added by the API for the role delimiters. A more accurate count includes these:

```python
def count_messages_tokens(messages: list, model: str = "gpt-4o-mini") -> int:
    """
    Count tokens for a full messages array, including formatting overhead.
    This is what the API actually charges you for.
    """
    enc = tiktoken.encoding_for_model(model)
    total = 0
    
    for message in messages:
        total += 4  # ~4 overhead tokens per message for role formatting
        for key, value in message.items():
            total += len(enc.encode(str(value)))
    
    total += 2  # 2 tokens to prime the assistant's reply
    return total
```

---

### 5.5 Calculating cost

```python
PRICING = {
    "gpt-4o":                        {"input": 5.00,   "output": 15.00},
    "gpt-4o-mini":                   {"input": 0.15,   "output": 0.60},
    "claude-sonnet-5":               {"input": 2.00,   "output": 10.00},
    "claude-haiku-4-5-20251001":     {"input": 1.00,   "output": 5.00},
    "gemini-2.5-flash":              {"input": 0.30,   "output": 2.50},
}

def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> dict:
    """Calculate the exact cost for a completed API call."""
    
    p = PRICING[model]
    input_cost  = (input_tokens  / 1_000_000) * p["input"]
    output_cost = (output_tokens / 1_000_000) * p["output"]
    total       = input_cost + output_cost
    
    return {
        "input_tokens"  : input_tokens,
        "output_tokens" : output_tokens,
        "input_cost"    : f"${input_cost:.6f}",
        "output_cost"   : f"${output_cost:.6f}",
        "total_cost"    : f"${total:.6f}",
    }
```

---

### 5.6 Full example: estimate, call, report

```python
def call_with_full_tracking(prompt: str, model: str = "gpt-4o-mini") -> dict:
    """
    Best practice pattern:
    1. Estimate token count before calling
    2. Make the call
    3. Use actual counts from response (more accurate than estimates)
    4. Calculate and report cost
    """
    from openai import OpenAI
    client = OpenAI()
    
    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user",   "content": prompt},
    ]
    
    # Pre-call estimate
    estimated_input = count_messages_tokens(messages)
    print(f"Estimated input tokens: ~{estimated_input}")
    
    # Make the call
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=500,
    )
    
    # Use actual counts from the response
    actual_input  = response.usage.prompt_tokens
    actual_output = response.usage.completion_tokens
    
    cost = calculate_cost(actual_input, actual_output, model)
    
    print(f"Actual input tokens : {actual_input}")
    print(f"Actual output tokens: {actual_output}")
    print(f"Total cost          : {cost['total_cost']}")
    
    return {
        "text": response.choices[0].message.content,
        "cost": cost,
    }
```

---

### 5.7 Context window guard

For long conversations, you need to ensure the total tokens do not exceed the model's
context window. Here is a utility that checks this before each call:

```python
CONTEXT_WINDOWS = {
    "gpt-4o-mini":               128_000,
    "claude-haiku-4-5-20251001": 200_000,
    "gemini-2.5-flash":        1_000_000,
}

def fits_in_context(messages: list, model: str, reserve_for_output: int = 1000) -> bool:
    """
    Returns True if the messages fit within the model's context window,
    leaving room for the expected output.
    
    Call this before any API call on a long conversation.
    """
    tokens_used = count_messages_tokens(messages)
    limit       = CONTEXT_WINDOWS.get(model, 128_000)
    headroom    = limit - tokens_used - reserve_for_output
    
    if headroom < 0:
        print(f"Context overflow! Used {tokens_used}, limit {limit}. "
              f"Trim {-headroom} tokens from conversation history.")
        return False
    
    print(f"Context OK: {tokens_used}/{limit} tokens used, {headroom} remaining.")
    return True
```

---

## 6. Error Handling & Exponential Backoff

### 6.1 Why API calls fail

Unlike calling a local function, network-based API calls can fail for many reasons:

| Error | HTTP Code | Cause | Should You Retry? |
|---|---|---|---|
| `RateLimitError` | 429 | Too many requests per minute / too many tokens per minute | Yes — after waiting |
| Server error | 500, 502, 503 | Provider infrastructure issue | Yes — after waiting |
| `APIConnectionError` | — | Your network connection dropped | Yes — after waiting |
| Bad request | 400 | Malformed input (wrong params, empty messages) | No — fix your code |
| Unauthorized | 401 | Invalid API key | No — check your key |
| Timeout | — | Request took too long | Depends — retry with caution |

Understanding which errors are retryable versus which require code changes is essential.
The 4xx errors (except 429) mean your request is fundamentally wrong, and retrying will
produce the same error. The 5xx errors and 429 are transient, so waiting and retrying will
usually succeed.

---

### 6.2 What is exponential backoff?

Naively retrying immediately when you hit a rate limit creates a thundering herd problem:
if 100 clients all hit a rate limit and all retry immediately, they all hit it again at
the exact same moment.

Exponential backoff solves this by waiting longer each time:

```
Attempt 1 fails  →  wait  1 second
Attempt 2 fails  →  wait  2 seconds
Attempt 3 fails  →  wait  4 seconds
Attempt 4 fails  →  wait  8 seconds
Attempt 5 fails  →  wait 16 seconds
```

Plus a small random **jitter** (e.g., +0 to +1 second) so that if many clients fail at
the same time, they don't all retry at exactly the same intervals.

---

### 6.3 Manual exponential backoff

Writing it manually is educational: it shows exactly what is happening.

```python
import time, random
from openai import OpenAI, RateLimitError, APIStatusError, APIConnectionError

client = OpenAI()

def call_with_manual_backoff(
    prompt: str,
    max_retries: int = 5,
    base_delay: float = 1.0,
) -> str:
    """
    Make an API call with manual exponential backoff.
    Good to understand before using a library.
    """
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                timeout=30.0,   # raise an error if no response within 30 seconds
            )
            return response.choices[0].message.content
        
        except RateLimitError:
            if attempt == max_retries - 1:
                raise  # last attempt — give up and propagate the error
            
            # Exponential wait + random jitter
            wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
            print(f"Rate limit hit. Waiting {wait_time:.1f}s "
                  f"(attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)
        
        except APIStatusError as e:
            if e.status_code == 400:
                # Bad request — your code is wrong, don't retry
                raise ValueError(f"Bad API request: {e.message}") from e
            elif e.status_code == 401:
                # Wrong API key — don't retry
                raise PermissionError("Invalid API key. Check your .env file.") from e
            elif e.status_code >= 500:
                # Server error — wait and retry
                if attempt == max_retries - 1:
                    raise
                wait_time = base_delay * (2 ** attempt)
                print(f"Server error {e.status_code}. Retrying in {wait_time:.0f}s...")
                time.sleep(wait_time)
            else:
                raise
        
        except APIConnectionError:
            if attempt == max_retries - 1:
                raise
            wait_time = base_delay * (2 ** attempt)
            print(f"Connection error. Retrying in {wait_time:.0f}s...")
            time.sleep(wait_time)
    
    raise RuntimeError("Exhausted all retry attempts")
```

---

### 6.4 Tenacity — the production approach

`tenacity` is a Python library that wraps your retry logic into a decorator, removing the
boilerplate loop. Under the hood it does exactly what `call_with_manual_backoff` does,
but with a cleaner API.

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

logger = logging.getLogger(__name__)

@retry(
    # Which exceptions trigger a retry?
    # Only retry on rate limits and network errors.
    # Do NOT retry on 400/401 — those are your bugs.
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    
    # Wait strategy: start at 1s, multiply by 2, cap at 60s
    # e.g. 1s → 2s → 4s → 8s → 16s → 32s → 60s → 60s → ...
    wait=wait_exponential(multiplier=1, min=1, max=60),
    
    # Give up after 5 total attempts
    stop=stop_after_attempt(5),
    
    # Automatically log each retry with the wait time and error
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def reliable_api_call(prompt: str) -> str:
    """
    This function automatically retries with exponential backoff
    if a RateLimitError or APIConnectionError occurs.
    The decorator handles all the retry logic transparently.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    return response.choices[0].message.content

# Usage — looks exactly like a normal call; retries are invisible
result = reliable_api_call("What is the chain rule in calculus?")
```

**Key tenacity parameters:**

| Parameter | What it does |
|---|---|
| `retry_if_exception_type(...)` | Only retry if the exception is one of these types |
| `wait_exponential(min, max)` | Wait at least `min` seconds, at most `max` seconds, doubling each time |
| `stop_after_attempt(n)` | Give up after `n` total attempts (first call + n-1 retries) |
| `before_sleep_log(logger, level)` | Log each retry automatically |

---

### 6.5 Multi-provider fallback

The most robust production pattern is to fall back to a different provider entirely if
your primary one keeps failing:

```python
from anthropic import Anthropic

client_openai    = OpenAI()
client_anthropic = Anthropic()

def call_with_fallback(prompt: str) -> dict:
    """
    Try OpenAI first. If it fails (after retries), switch to Claude.
    This achieves high availability: your app keeps working even if
    one provider has an outage.
    """
    
    providers = [
        {
            "name": "OpenAI GPT-4o-mini",
            "call": lambda p: client_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": p}],
                max_tokens=300,
            ).choices[0].message.content,
        },
        {
            "name": "Claude Haiku",
            "call": lambda p: client_anthropic.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                messages=[{"role": "user", "content": p}],
            ).content[0].text,
        },
    ]
    
    for provider in providers:
        try:
            text = provider["call"](prompt)
            return {"provider": provider["name"], "text": text, "success": True}
        except Exception as e:
            print(f"[{provider['name']}] failed: {type(e).__name__} — trying next")
            continue
    
    return {"success": False, "error": "All providers failed"}
```

---

### 6.6 What errors look like in practice

```python
def demonstrate_error_types():
    """Intentionally trigger errors to see what they look like."""
    
    # 1. Invalid model name → APIStatusError with status_code=404
    try:
        client.chat.completions.create(
            model="gpt-99-ultra-pro-max",   # does not exist
            messages=[{"role": "user", "content": "hi"}]
        )
    except APIStatusError as e:
        print(f"Type    : {type(e).__name__}")
        print(f"Code    : {e.status_code}")
        print(f"Message : {e.message[:80]}")
    
    # 2. Empty messages array → APIStatusError with status_code=400
    try:
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[]
        )
    except APIStatusError as e:
        print(f"Empty messages error: {e.status_code} — {e.message[:80]}")
    
    # RateLimitError looks like:
    # openai.RateLimitError: Error code: 429 — Rate limit reached for requests...
    # It happens when you exceed your RPM (requests per minute) or
    # TPM (tokens per minute) quota for your tier.
```

---

## 7. Async & Parallel API Calls

### 7.1 Why async matters for LLM applications

API calls are **I/O-bound** operations. Your CPU does almost nothing during an API call:
it sends the request, then waits for bytes to come back over the network. This is
fundamentally different from CPU-bound work like training a neural network.

**Synchronous execution** (the default in Python):

```
Start call 1 → [wait 2 seconds] → Get result 1
Start call 2 → [wait 2 seconds] → Get result 2
Start call 3 → [wait 2 seconds] → Get result 3
Total: 6 seconds
```

**Asynchronous execution:**

```
Start call 1 ─────────────────── Get result 1
Start call 2 ──────────────── Get result 2
Start call 3 ─────────────────────── Get result 3
Total: ~2 seconds (the longest single call)
```

All three calls are waiting for network responses simultaneously. Your Python process is
not doing three things at once in parallel (that would be multithreading). It is
interleaving them on a single thread using an event loop. When call 1 is waiting for the
network, the event loop starts call 2. When call 2 is waiting, the event loop starts
call 3. They are concurrent, not parallel, but for I/O-bound work this achieves the
same speedup.

**Analogy from ML:** Think of batched inference. Instead of running samples through your
network one at a time, you batch them. Async API calls are the I/O equivalent: instead
of waiting for one call to complete before starting the next, you run multiple calls
concurrently. The speedup is proportional to the number of concurrent calls, bounded by
rate limits.

---

### 7.2 Python async basics

Python's async system uses `asyncio` and two keywords:

```python
import asyncio

# 'async def' defines a coroutine — a function that can be paused and resumed
async def my_coroutine():
    # 'await' pauses this coroutine and lets other coroutines run
    # while waiting for the result
    result = await some_async_operation()
    return result

# asyncio.run() starts the event loop and runs a top-level coroutine
asyncio.run(my_coroutine())
```

The `await` keyword is the key. When execution hits an `await`, Python says:
"This operation will take a while (network I/O). I will pause here and let
other coroutines run in the meantime. When the result arrives, I will resume."

---

### 7.3 Async API clients

The SDK providers offer async versions of their clients. You must use these: calling
the regular synchronous client inside an `async def` function blocks the entire event
loop and defeats the purpose of async.

```python
from openai import OpenAI, AsyncOpenAI       # regular vs async
from anthropic import Anthropic, AsyncAnthropic

# Synchronous clients (for regular Python code)
sync_openai    = OpenAI()
sync_anthropic = Anthropic()

# Async clients (for async def functions)
async_openai    = AsyncOpenAI()
async_anthropic = AsyncAnthropic()
```

---

### 7.4 Single async call

```python
async def async_call(prompt: str, label: str = "") -> tuple[str, str]:
    """
    A single async API call.
    The 'await' keyword pauses this function while waiting for the response,
    allowing other async functions to run in the meantime.
    """
    response = await async_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
    )
    return label, response.choices[0].message.content
```

---

### 7.5 Parallel calls with asyncio.gather

`asyncio.gather()` is the core function for running multiple coroutines concurrently.
It launches all the tasks, then waits until all of them complete.

```python
async def parallel_calls():
    """Run multiple API calls simultaneously."""
    
    questions = [
        ("q1", "What is gradient descent in one sentence?"),
        ("q2", "What is a loss function in one sentence?"),
        ("q3", "What is regularization in one sentence?"),
        ("q4", "What is batch normalization in one sentence?"),
        ("q5", "What is dropout in one sentence?"),
    ]
    
    # Sequential benchmark
    t0 = time.time()
    seq_results = []
    for label, q in questions:
        result = await async_call(q, label)   # waits for each one before starting next
        seq_results.append(result)
    print(f"Sequential: {time.time()-t0:.1f}s")
    
    # Parallel: create all tasks, then gather all results
    t0 = time.time()
    tasks = [async_call(q, label) for label, q in questions]  # creates coroutines
    par_results = await asyncio.gather(*tasks)                  # runs all concurrently
    print(f"Parallel  : {time.time()-t0:.1f}s")
    
    # Typical output:
    # Sequential: 10.3s
    # Parallel  :  2.1s  (5x speedup with 5 concurrent calls)
    
    return par_results
```

---

### 7.6 Rate-limited parallel calls with Semaphore

Sending 100 requests simultaneously will hit your rate limit. A `Semaphore` acts as a
concurrency throttle: at most N tasks can be "inside" the semaphore block at any time.
When one task leaves (finishes), another can enter.

```python
async def rate_limited_batch(prompts: list[str], max_concurrent: int = 5) -> list[str]:
    """
    Process many prompts concurrently but cap the number of simultaneous calls.
    
    asyncio.Semaphore(N) = a counter initialized to N.
    'async with semaphore' decrements the counter (or waits if it's 0).
    When the 'with' block exits, the counter increments again.
    
    Effect: at most max_concurrent calls run simultaneously.
    """
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def limited_call(prompt: str) -> str:
        async with semaphore:   # acquire a slot (waits if all slots are taken)
            response = await async_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
            )
            return response.choices[0].message.content
        # slot is automatically released when exiting the 'with' block
    
    tasks = [limited_call(p) for p in prompts]
    return await asyncio.gather(*tasks)


# Usage: process 20 prompts, max 5 at a time
async def main():
    prompts = [f"Give me a Python tip about topic #{i}" for i in range(1, 21)]
    results = await rate_limited_batch(prompts, max_concurrent=5)
    for i, r in enumerate(results, 1):
        print(f"#{i}: {r[:80]}")
```

---

### 7.7 Async streaming

You can combine async and streaming. The pattern is the same as sync streaming but uses
`await` and `async for`:

```python
async def async_streaming_call(prompt: str) -> str:
    """Async streaming — standard pattern for web backends."""
    
    stream = await async_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    
    chunks = []
    async for chunk in stream:   # 'async for' instead of regular 'for'
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
            chunks.append(delta.content)
    
    print()
    return "".join(chunks)
```

---

### 7.8 When to use async vs sync

| Situation | Use |
|---|---|
| Single API call, simple script | Synchronous |
| Need multiple API calls and speed matters | Async |
| Building a web API backend (FastAPI, etc.) | Async (FastAPI is async by default) |
| Building a CLI tool | Either; sync is simpler |
| Batch processing (many prompts) | Async with rate-limited semaphore |
| Streaming to a web client | Async streaming |

---

## 8. Key Takeaways

After completing this phase, you understand:

1. **LLM APIs are neural networks behind HTTP endpoints.** You call them via SDK
   libraries that abstract away the HTTP layer.

2. **The messages format is everything.** Every multi-turn conversation is just a list
   of `{role, content}` dicts sent on every API call. The model is stateless, so you
   maintain history.

3. **Streaming is `stream=True`.** It dramatically improves perceived responsiveness by
   sending tokens as they are generated.

4. **Tokens ≠ words.** Count them with tiktoken. Track them in every response.
   Calculate costs before they surprise you.

5. **Only retry transient errors.** Use exponential backoff for 429 and 5xx. Never
   retry 400 and 401; those are bugs in your code.

6. **Async unlocks parallelism for I/O.** `asyncio.gather()` for concurrent calls,
   `Semaphore` to avoid rate limits. 5 parallel calls ≈ 5x speedup.

7. **Abstract your providers.** Build a wrapper class so the rest of your code never
   calls OpenAI or Claude directly. This lets you switch providers, add fallbacks, and
   track usage from one place.

---

## 9. Practice Exercises

Work through these before moving to Phase 02. They progressively build on each other.

### Exercise 1 — Cross-Provider Quality Comparison (Easy)
Call all three providers (OpenAI, Claude, Gemini) with the same technical ML question.
Print each response with its token count and cost. Compare the quality, style, and cost
of the three answers.

### Exercise 2 — Multi-Turn Conversation (Medium)
Build a simple conversation loop. Maintain a `history` list. After each model response,
append both the user message and the assistant response to the list. Before each call,
check if the total tokens exceed 80% of the context window and print a warning.

```python
# Expected behavior:
history = [{"role": "system", "content": "You are a helpful ML tutor."}]

user_input = "What is a perceptron?"
# → append to history, call API, append response, print

user_input = "How does it relate to modern neural networks?"
# → model refers back to perceptron because history is included
```

### Exercise 3 — Budget Guard (Medium-Hard)
Extend the `LLMClient` class to accept a `max_budget_usd` parameter at initialization.
Before each `chat()` call, check if the accumulated `stats.total_cost_usd` plus the
estimated cost of the current call would exceed the budget. If it would, raise a custom
`BudgetExceededError` with a clear message showing how much has been spent and what the
limit is.

### Exercise 4 — Parallel Prompt Batch (Hard)
Build an `async def batch_summarize(texts: list[str]) -> list[str]` function that takes
a list of long strings and returns a summary for each, using parallel async calls with
a semaphore capped at 3 concurrent calls. Track and print the total time saved vs
sequential execution.

---

*Next: Phase 02, Prompt Engineering*  
*You will learn zero-shot, few-shot, Chain-of-Thought prompting, structured outputs,
function calling, and how to build a data extraction pipeline.*
