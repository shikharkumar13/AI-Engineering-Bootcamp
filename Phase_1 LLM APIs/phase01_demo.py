"""
demo.py — Run this to see the Universal LLM Client in action.

Make sure you have a .env file with:
  OPENAI_API_KEY=...
  ANTHROPIC_API_KEY=...
  GOOGLE_API_KEY=...

Then run: python demo.py
"""

import asyncio, time
from llm_client import LLMClient, Provider


def demo_basic():
    """Demo 1: Simple calls to each provider"""
    
    print("\n" + "═" * 60)
    print("DEMO 1: Basic calls to all 3 providers")
    print("═" * 60)
    
    client = LLMClient()
    question = "Explain the vanishing gradient problem in 2 sentences."
    
    for provider in Provider:
        print(f"\n▸ {provider.value.upper()}")
        resp = client.chat(question, provider=provider)
        print(f"  {resp.text}")
        print(f"  [{resp.input_tokens} in, {resp.output_tokens} out, ${resp.cost_usd:.6f}]")


def demo_streaming():
    """Demo 2: Streaming from different providers"""
    
    print("\n" + "═" * 60)
    print("DEMO 2: Streaming (tokens appear live)")
    print("═" * 60)
    
    client = LLMClient()
    prompt = "Explain attention mechanism in transformers in 3 sentences."
    
    print("\n▸ OpenAI streaming:")
    client.chat(prompt, provider=Provider.OPENAI, stream=True)
    
    print("\n▸ Claude streaming:")
    client.chat(prompt, provider=Provider.CLAUDE, stream=True)


def demo_token_tracking():
    """Demo 3: Cost and token tracking"""
    
    print("\n" + "═" * 60)
    print("DEMO 3: Token and cost tracking")
    print("═" * 60)
    
    client = LLMClient()
    
    prompts = [
        "What is dropout?",
        "What is batch normalization?",
        "What is the difference between L1 and L2 regularization?",
    ]
    
    for prompt in prompts:
        estimate = client.estimate_cost(prompt)
        print(f"\nPrompt: '{prompt}'")
        print(f"  Pre-call estimate: {estimate}")
        
        resp = client.chat(prompt, provider=Provider.OPENAI)
        print(f"  Actual:  {resp.input_tokens} in + {resp.output_tokens} out = ${resp.cost_usd:.6f}")
    
    # Print session totals
    client.stats.report()


def demo_fallback():
    """Demo 4: Multi-provider fallback"""
    
    print("\n" + "═" * 60)
    print("DEMO 4: Fallback across providers")
    print("═" * 60)
    
    client = LLMClient()
    
    # Try OpenAI first, fall back to Claude if it fails
    resp = client.chat_with_fallback(
        "What is a convolutional neural network?",
        providers=[Provider.OPENAI, Provider.CLAUDE, Provider.GEMINI]
    )
    
    print(f"\nAnswered by: {resp.provider} / {resp.model}")
    print(f"Response: {resp.text}")


async def demo_parallel():
    """Demo 5: Parallel async calls — the speed demo"""
    
    print("\n" + "═" * 60)
    print("DEMO 5: Parallel calls (async speedup)")
    print("═" * 60)
    
    client = LLMClient()
    
    questions = [
        "What is a perceptron?",
        "What is gradient descent?",
        "What is a loss function?",
        "What is softmax?",
        "What is cross-entropy loss?",
    ]
    
    print(f"\nSending {len(questions)} questions in parallel...")
    t0 = time.time()
    
    results = await client.parallel(questions, provider=Provider.OPENAI, max_concurrent=5)
    
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s (vs ~{len(questions)*2:.0f}s sequential)\n")
    
    for q, r in zip(questions, results):
        if not isinstance(r, Exception):
            print(f"Q: {q}")
            print(f"A: {r.text[:120]}...\n")


def demo_system_prompt():
    """Demo 6: Custom system prompts — how to shape model behavior"""
    
    print("\n" + "═" * 60)
    print("DEMO 6: System prompts change model behavior")
    print("═" * 60)
    
    client = LLMClient()
    question = "What is a neural network?"
    
    personas = [
        ("5-year-old teacher", "Explain everything like you're talking to a 5-year-old. Use simple words and analogies."),
        ("PhD advisor",        "You are a rigorous ML researcher. Be technically precise and cite key papers."),
        ("Pirate",             "You are a pirate. Explain everything in pirate speech."),
    ]
    
    for name, system in personas:
        print(f"\n▸ Persona: {name}")
        resp = client.chat(question, system=system, provider=Provider.OPENAI)
        print(f"  {resp.text[:200]}...")


if __name__ == "__main__":
    # Run all demos in sequence
    demo_basic()
    demo_streaming()
    demo_token_tracking()
    demo_fallback()
    demo_system_prompt()
    
    # Parallel demo (async — needs asyncio.run)
    asyncio.run(demo_parallel())
    
    print("\n\n✅ Phase 01 complete. You now have a production-ready LLM client.")
