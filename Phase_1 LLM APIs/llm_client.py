"""
Phase 01 Project — Universal LLM Client
========================================
A single class that wraps OpenAI, Claude, and Gemini with:
  - Unified interface across all providers
  - Streaming support
  - Token counting and cost tracking
  - Automatic retry with exponential backoff (OpenAI and Claude calls; the
    google-generativeai SDK exposes a different exception hierarchy and
    isn't wired into this retry path)
  - Async parallel calls
  - Multi-provider fallback

Usage:
  from llm_client import LLMClient, Provider

  client = LLMClient()
  response = client.chat("What is backpropagation?")
  response = client.chat("...", provider=Provider.CLAUDE, stream=True)
  results  = await client.parallel(["q1", "q2", "q3"])
"""

import os, asyncio, time, logging
from enum import Enum
from dataclasses import dataclass, field
from dotenv import load_dotenv

from openai import OpenAI, AsyncOpenAI, RateLimitError, APIConnectionError
from anthropic import (
    Anthropic,
    AsyncAnthropic,
    RateLimitError as AnthropicRateLimitError,
    APIConnectionError as AnthropicAPIConnectionError,
)
import google.generativeai as genai
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────

class Provider(Enum):
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"


# Default model per provider — change these to your preference
DEFAULT_MODELS = {
    Provider.OPENAI: "gpt-4o-mini",
    Provider.CLAUDE: "claude-haiku-4-5-20251001",
    Provider.GEMINI: "gemini-2.5-flash",
}

# Pricing: USD per 1 million tokens — verify current numbers on each
# provider's pricing page before relying on these for real budgeting;
# providers revise pricing and retire model IDs without much notice.
PRICING = {
    "gpt-4o":                        {"input": 5.00,   "output": 15.00},
    "gpt-4o-mini":                   {"input": 0.15,   "output": 0.60},
    "claude-opus-5":                 {"input": 5.00,   "output": 25.00},
    "claude-sonnet-5":               {"input": 2.00,   "output": 10.00},
    "claude-haiku-4-5-20251001":     {"input": 1.00,   "output": 5.00},
    "gemini-2.5-flash":              {"input": 0.30,   "output": 2.50},
    "gemini-2.5-pro":                {"input": 1.25,   "output": 10.00},
}


@dataclass
class LLMResponse:
    """Structured response from any provider"""
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float

    def __str__(self):
        return (
            f"[{self.provider} / {self.model}]\n"
            f"{self.text}\n\n"
            f"tokens: {self.input_tokens} in + {self.output_tokens} out | "
            f"cost: ${self.cost_usd:.6f} | {self.latency_s:.2f}s"
        )


@dataclass
class UsageStats:
    """Tracks cumulative usage across all calls in a session"""
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    calls_by_provider: dict = field(default_factory=dict)

    def record(self, response: LLMResponse):
        self.total_calls += 1
        self.total_input_tokens  += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.total_cost_usd      += response.cost_usd
        self.calls_by_provider[response.provider] = \
            self.calls_by_provider.get(response.provider, 0) + 1

    def report(self):
        print("\n" + "=" * 50)
        print("SESSION USAGE REPORT")
        print("=" * 50)
        print(f"  Total calls:          {self.total_calls}")
        print(f"  Total input tokens:   {self.total_input_tokens:,}")
        print(f"  Total output tokens:  {self.total_output_tokens:,}")
        print(f"  Total cost:           ${self.total_cost_usd:.6f}")
        print(f"  By provider:          {self.calls_by_provider}")
        print("=" * 50)


# ──────────────────────────────────────────────────────────────
# Token utilities
# ──────────────────────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    """Count tokens using OpenAI's tiktoken (works for GPT models; approximate for others)"""
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    return len(enc.encode(text))


def _calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for a given model and token counts"""
    p = PRICING.get(model)
    if not p:
        return 0.0
    return (input_tokens / 1_000_000) * p["input"] + \
           (output_tokens / 1_000_000) * p["output"]


# ──────────────────────────────────────────────────────────────
# Main client class
# ──────────────────────────────────────────────────────────────

class LLMClient:
    """
    Universal wrapper around OpenAI, Claude, and Gemini.
    
    Quick start:
        client = LLMClient()
        resp = client.chat("Explain dropout regularization")
        print(resp)
    """

    def __init__(self):
        # Sync clients
        self._openai    = OpenAI()
        self._anthropic = Anthropic()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

        # Async clients (for parallel calls)
        self._async_openai    = AsyncOpenAI()
        self._async_anthropic = AsyncAnthropic()

        self.stats = UsageStats()

    # ── Core: single chat call ───────────────────────────────

    def chat(
        self,
        prompt: str,
        provider: Provider = Provider.OPENAI,
        model: str | None = None,
        system: str = "You are a helpful assistant.",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> LLMResponse:
        """
        Send a message to any provider.
        
        Args:
            prompt:      The user message
            provider:    Provider.OPENAI / Provider.CLAUDE / Provider.GEMINI
            model:       Override the default model
            system:      System prompt (instructions)
            max_tokens:  Max response length in tokens
            temperature: 0=deterministic, 1=creative
            stream:      Print tokens as they arrive

        Returns:
            LLMResponse with text, tokens, cost, and latency
        """
        model = model or DEFAULT_MODELS[provider]
        t0 = time.time()

        if provider == Provider.OPENAI:
            resp = self._openai_call(prompt, system, model, max_tokens, temperature, stream)
        elif provider == Provider.CLAUDE:
            resp = self._claude_call(prompt, system, model, max_tokens, temperature, stream)
        elif provider == Provider.GEMINI:
            resp = self._gemini_call(prompt, system, model, max_tokens, temperature, stream)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        resp.latency_s = round(time.time() - t0, 3)
        self.stats.record(resp)
        return resp

    # ── Fallback: try providers in order ────────────────────

    def chat_with_fallback(
        self,
        prompt: str,
        providers: list[Provider] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Try each provider in order. Use the first one that succeeds.
        Default order: OpenAI → Claude → Gemini.
        """
        providers = providers or [Provider.OPENAI, Provider.CLAUDE, Provider.GEMINI]

        for p in providers:
            try:
                return self.chat(prompt, provider=p, **kwargs)
            except Exception as e:
                print(f"[{p.value}] failed: {type(e).__name__} — trying next provider")
        
        raise RuntimeError("All providers failed")

    # ── Parallel: run multiple prompts concurrently ──────────

    async def parallel(
        self,
        prompts: list[str],
        provider: Provider = Provider.OPENAI,
        max_concurrent: int = 5,
        **kwargs,
    ) -> list[LLMResponse]:
        """
        Run multiple prompts in parallel (async).
        Up to max_concurrent calls run simultaneously.
        
        Usage:
            results = await client.parallel(["q1", "q2", "q3"])
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _one(prompt: str) -> LLMResponse:
            async with semaphore:
                model = kwargs.get("model") or DEFAULT_MODELS[provider]
                system = kwargs.get("system", "You are a helpful assistant.")
                max_tok = kwargs.get("max_tokens", 512)

                if provider == Provider.OPENAI:
                    return await self._async_openai_call(prompt, system, model, max_tok)
                elif provider == Provider.CLAUDE:
                    return await self._async_claude_call(prompt, system, model, max_tok)
                else:
                    raise ValueError("Gemini async not supported in this client")

        tasks = [_one(p) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, LLMResponse):
                self.stats.record(r)

        return results

    # ── Token counting ───────────────────────────────────────

    def count_tokens(self, text: str) -> int:
        """Count tokens in a string (approximate for non-GPT models)"""
        return _count_tokens(text)

    def estimate_cost(self, prompt: str, model: str = "gpt-4o-mini") -> str:
        """Estimate cost before making a call (based on input tokens only)"""
        n = _count_tokens(prompt)
        p = PRICING.get(model, {"input": 0, "output": 0})
        cost = (n / 1_000_000) * p["input"]
        return f"~{n} tokens, estimated input cost: ${cost:.6f}"

    # ── Internal provider implementations ───────────────────

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(5),
    )
    def _openai_call(self, prompt, system, model, max_tokens, temperature, stream) -> LLMResponse:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ]

        if stream:
            print(f"[{model}] ", end="")
            chunks = []
            resp_stream = self._openai.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens,
                temperature=temperature, stream=True,
            )
            for chunk in resp_stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    print(delta.content, end="", flush=True)
                    chunks.append(delta.content)
            print()
            text = "".join(chunks)
            # Token counts unavailable mid-stream; estimate from text
            in_tok  = _count_tokens(system + prompt)
            out_tok = _count_tokens(text)
        else:
            r = self._openai.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens,
                temperature=temperature,
            )
            text    = r.choices[0].message.content
            in_tok  = r.usage.prompt_tokens
            out_tok = r.usage.completion_tokens

        return LLMResponse(
            text=text, provider="openai", model=model,
            input_tokens=in_tok, output_tokens=out_tok,
            cost_usd=_calc_cost(model, in_tok, out_tok), latency_s=0,
        )

    @retry(
        retry=retry_if_exception_type((AnthropicRateLimitError, AnthropicAPIConnectionError)),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(5),
    )
    def _claude_call(self, prompt, system, model, max_tokens, temperature, stream) -> LLMResponse:
        if stream:
            print(f"[{model}] ", end="")
            chunks = []
            with self._anthropic.messages.stream(
                model=model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": prompt}],
            ) as s:
                for t in s.text_stream:
                    print(t, end="", flush=True)
                    chunks.append(t)
            print()
            text    = "".join(chunks)
            in_tok  = _count_tokens(system + prompt)
            out_tok = _count_tokens(text)
        else:
            r = self._anthropic.messages.create(
                model=model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            text    = r.content[0].text
            in_tok  = r.usage.input_tokens
            out_tok = r.usage.output_tokens

        return LLMResponse(
            text=text, provider="claude", model=model,
            input_tokens=in_tok, output_tokens=out_tok,
            cost_usd=_calc_cost(model, in_tok, out_tok), latency_s=0,
        )

    def _gemini_call(self, prompt, system, model, max_tokens, temperature, stream) -> LLMResponse:
        g_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system,
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_tokens, temperature=temperature,
            ),
        )
        if stream:
            print(f"[{model}] ", end="")
            chunks = []
            for chunk in g_model.generate_content(prompt, stream=True):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    chunks.append(chunk.text)
            print()
            text    = "".join(chunks)
            in_tok  = _count_tokens(system + prompt)
            out_tok = _count_tokens(text)
        else:
            r       = g_model.generate_content(prompt)
            text    = r.text
            in_tok  = r.usage_metadata.prompt_token_count
            out_tok = r.usage_metadata.candidates_token_count

        return LLMResponse(
            text=text, provider="gemini", model=model,
            input_tokens=in_tok, output_tokens=out_tok,
            cost_usd=_calc_cost(model, in_tok, out_tok), latency_s=0,
        )

    async def _async_openai_call(self, prompt, system, model, max_tokens) -> LLMResponse:
        t0 = time.time()
        r  = await self._async_openai.chat.completions.create(
            model=model, max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
        )
        return LLMResponse(
            text=r.choices[0].message.content, provider="openai", model=model,
            input_tokens=r.usage.prompt_tokens, output_tokens=r.usage.completion_tokens,
            cost_usd=_calc_cost(model, r.usage.prompt_tokens, r.usage.completion_tokens),
            latency_s=round(time.time() - t0, 3),
        )

    async def _async_claude_call(self, prompt, system, model, max_tokens) -> LLMResponse:
        t0 = time.time()
        r  = await self._async_anthropic.messages.create(
            model=model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return LLMResponse(
            text=r.content[0].text, provider="claude", model=model,
            input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens,
            cost_usd=_calc_cost(model, r.usage.input_tokens, r.usage.output_tokens),
            latency_s=round(time.time() - t0, 3),
        )
