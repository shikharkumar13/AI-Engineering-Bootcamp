# Phase 03 — LangChain & Orchestration

> **Prerequisites:** Phase 01 (LLM APIs) and Phase 02 (Prompt Engineering) complete.  
> **What you'll learn:** LangChain's core abstractions; composing pipelines with LCEL;
> conversation memory; loading and splitting documents; tracing with LangSmith.  
> **Project:** A multi-turn document Q&A system with session memory and LangSmith tracing.

---

## Table of Contents

1. [The Big Picture — Why LangChain?](#1-the-big-picture--why-langchain)
2. [Setup & Installation](#2-setup--installation)
3. [Core Abstractions — Prompts, Models, Parsers](#3-core-abstractions--prompts-models-parsers)
4. [LangChain Expression Language (LCEL)](#4-langchain-expression-language-lcel)
5. [Conversation Memory](#5-conversation-memory)
6. [Document Loaders](#6-document-loaders)
7. [Text Splitters](#7-text-splitters)
8. [LangSmith — Tracing & Observability](#8-langsmith--tracing--observability)
9. [Key Takeaways](#9-key-takeaways)
10. [Practice Exercises](#10-practice-exercises)

---

## 1. The Big Picture — Why LangChain?

In Phase 01 you learned to call LLM APIs. In Phase 02 you learned to craft precise
prompts and extract structured data. But as soon as your application has more than one
LLM call, you face a new set of problems:

**Problem 1: Chaining calls manually is verbose.**
```python
# Without LangChain — calling two prompts in sequence
response1 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": f"Summarize this: {document}"}]
)
summary = response1.choices[0].message.content

response2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": f"Translate to French: {summary}"}]
)
result = response2.choices[0].message.content
```

**Problem 2: Conversation history must be managed manually.**  
Every call is stateless. You have to append messages to a history list yourself, worry
about the context window filling up, and pass history to every call.

**Problem 3: Document loading is non-trivial.**  
PDFs have structure, HTML has tags, CSVs have headers, Word docs need parsing.
Writing loaders for each format is boring and error-prone.

**Problem 4: No observability.**  
You cannot see inside what is happening: which prompt was sent, what came back, how
many tokens were used across a multi-step pipeline.

LangChain solves all four of these. It is a framework of composable, interchangeable
building blocks for LLM applications:

- **Prompt templates:** reusable prompt objects with variable substitution
- **Model wrappers:** consistent interface across all providers
- **Output parsers:** extract structured data from model responses
- **LCEL:** compose all of the above with the `|` operator
- **Memory:** manage conversation history automatically
- **Document loaders:** load any file type into a standard format
- **Text splitters:** split large documents into chunks
- **LangSmith:** trace, debug, and evaluate everything

---

### 1.1 When NOT to use LangChain

LangChain adds abstraction. Abstraction has a cost: more to learn, more indirection, 
more things that can break in non-obvious ways. Do not reach for it automatically.

| Use LangChain | Stick to raw API calls |
|---|---|
| Multi-step pipelines with 3+ LLM calls | Single LLM call scripts |
| Need conversation memory across sessions | One-off questions |
| Loading and processing documents | Already have processed text |
| Need LangSmith tracing on a team | Solo prototyping |
| Building something others will maintain | Quick experiments |

---

### 1.2 LangChain package structure (v0.3+)

LangChain is split into focused packages. Install only what you need:

```
langchain-core        → Runnables, prompts, parsers, base classes
langchain-openai      → ChatOpenAI, OpenAIEmbeddings
langchain-anthropic   → ChatAnthropic
langchain-community   → 3rd party integrations (loaders, memory backends, etc.)
langchain-text-splitters → Text splitting utilities
langchain             → High-level convenience imports
langsmith             → Tracing and evaluation platform
```

This split happened in v0.1 and was completed by v0.3. If you see old tutorials importing
`from langchain.chat_models import ChatOpenAI`, that is the legacy monolithic package.
The current import is `from langchain_openai import ChatOpenAI`.

---

## 2. Setup & Installation

```bash
pip install langchain langchain-openai langchain-community langchain-text-splitters \
            langsmith pypdf beautifulsoup4 python-dotenv tiktoken
```

Update your `.env` file:

```bash
# Existing from Phase 01
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# New for Phase 03: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...          # from smith.langchain.com
LANGCHAIN_PROJECT=phase03-doc-chat  # optional — groups your traces
```

Get your LangSmith API key at: https://smith.langchain.com (free tier available).  
Setting `LANGCHAIN_TRACING_V2=true` is all you need. Every LangChain call is then
automatically traced, no code changes required.

---

## 3. Core Abstractions — Prompts, Models, Parsers

### 3.1 The ChatModel

`ChatOpenAI` is LangChain's wrapper around the OpenAI chat API. It is a `Runnable`,
meaning it can be composed with other Runnables using `|`.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Initialize the model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=1024,
)

# Call directly with a list of messages (just like the raw API)
response = llm.invoke([
    SystemMessage(content="You are a concise ML tutor."),
    HumanMessage(content="What is a perceptron?"),
])

print(type(response))         # → AIMessage
print(response.content)       # → "A perceptron is the simplest neural network..."
print(response.response_metadata)  # → token counts, model name, finish reason
```

**The Message types:**

| Class | Role | When you create it |
|---|---|---|
| `SystemMessage` | System instructions | You create it (developer) |
| `HumanMessage` | User input | You create it (or from user) |
| `AIMessage` | Model response | Returned by the model |

These map directly to the `{"role": "system/user/assistant", "content": "..."}` dicts
you used in Phase 01. LangChain uses typed objects instead.

---

### 3.2 ChatPromptTemplate

A `ChatPromptTemplate` is a reusable prompt with variable slots. It produces a list of
messages when you call `.format_messages()` or `.invoke()`.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Basic template with a variable {topic}
template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}. Be concise and precise."),
    ("human",  "Explain {topic} in simple terms."),
])

# Format the template with actual values
messages = template.format_messages(
    domain="machine learning",
    topic="backpropagation"
)
# → [SystemMessage("You are an expert in machine learning..."),
#    HumanMessage("Explain backpropagation in simple terms.")]

# Or invoke with a dict (this is the LCEL way, used when composing with |)
messages = template.invoke({
    "domain": "machine learning",
    "topic": "backpropagation"
})
```

**Template variable syntax:**
- `{variable_name}`: required variable, must be provided
- There is no optional variable syntax; use Python logic before invoking if needed

**The `MessagesPlaceholder`:**  
This is how you inject a list of messages (like conversation history) into a template.
The placeholder is replaced with the actual message list at runtime:

```python
from langchain_core.prompts import MessagesPlaceholder

# Template with a slot for conversation history
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Context: {context}"),
    MessagesPlaceholder(variable_name="history"),   # ← history goes here
    ("human", "{question}"),
])

# When invoked, history is injected between system and human messages
messages = template.invoke({
    "context": "...",
    "history": [
        HumanMessage(content="What is a transformer?"),
        AIMessage(content="A transformer is..."),
    ],
    "question": "How does attention work in transformers?"
})
# → [SystemMessage, HumanMessage("What is a transformer?"),
#    AIMessage("A transformer is..."), HumanMessage("How does attention work...")]
```

This is the core mechanism behind conversation memory: MessagesPlaceholder injects
the history into the prompt before each model call.

---

### 3.3 Output Parsers

Output parsers transform the raw `AIMessage` from the model into a more useful type.

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# StrOutputParser: extracts .content from AIMessage as a plain string
# This is what you use 90% of the time
str_parser = StrOutputParser()

raw_response: AIMessage  = llm.invoke([HumanMessage(content="Say hello")])
clean_string: str        = str_parser.invoke(raw_response)
# raw_response → AIMessage(content="Hello! How can I help you?", ...)
# clean_string → "Hello! How can I help you?"


# JsonOutputParser: parses JSON from the model's string output
json_parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Always respond with valid JSON only."),
    ("human", "Give me info about {animal} as JSON with keys: name, diet, habitat")
])

chain = prompt | llm | json_parser  # ← we'll explain this | syntax in Section 4
result = chain.invoke({"animal": "red fox"})
# → {"name": "Red Fox", "diet": "omnivore", "habitat": "forests and urban areas"}
print(type(result))  # → dict  (not string!)
```

**Available output parsers:**

| Parser | Output type | Use case |
|---|---|---|
| `StrOutputParser` | `str` | Most chat and text generation tasks |
| `JsonOutputParser` | `dict` | When you need a dict but don't have a Pydantic model |
| `PydanticOutputParser` | Pydantic model | When you need validated structured output |
| `CommaSeparatedListOutputParser` | `list[str]` | When model outputs a comma-separated list |

> **When to use `JsonOutputParser` vs `instructor`?**  
> `instructor` (from Phase 02) is more reliable because it uses function calling.
> `JsonOutputParser` just asks the model to format as JSON and parses it, so it can fail
> if the model adds commentary. Use `instructor` for critical extraction; use
> `JsonOutputParser` when you want LangChain integration and the structure is simple.

---

## 4. LangChain Expression Language (LCEL)

LCEL is the modern way to build LangChain pipelines. It was introduced in v0.1 and is
now the primary interface. The old `LLMChain`, `ConversationChain`, `SequentialChain`
classes were deprecated for years and have since been removed from current LangChain
releases entirely. LCEL replaces them all.

### 4.1 The `|` operator — composing Runnables

The `|` pipe operator composes two Runnables: the output of the left becomes the input
of the right. This is borrowed from Unix pipes (`cat file | grep "error" | head -n 5`).

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise technical writer."),
    ("human",  "Summarize this in one sentence: {text}"),
])
parser = StrOutputParser()

# Compose a chain with |
# Data flows left to right: dict → prompt → messages → llm → AIMessage → str
chain = prompt | llm | parser

# Invoke the chain
result = chain.invoke({"text": "Backpropagation is an algorithm for training..."})
print(result)  # → "Backpropagation trains neural networks by..."
print(type(result))  # → str
```

**Every component in the chain is a Runnable.** Runnables share a common interface:

| Method | What it does | Returns |
|---|---|---|
| `.invoke(input)` | Run once, return result | Single output |
| `.stream(input)` | Run and yield tokens as they arrive | Generator |
| `.batch([input1, input2, ...])` | Run multiple inputs in parallel | List of outputs |
| `.ainvoke(input)` | Async version of invoke | Awaitable |
| `.astream(input)` | Async version of stream | Async generator |

```python
# .invoke — single call
result = chain.invoke({"text": "Transformers use self-attention..."})
print(result)

# .stream — token by token (great for chat UIs)
for token in chain.stream({"text": "Transformers use self-attention..."}):
    print(token, end="", flush=True)
print()

# .batch — multiple inputs, run in parallel internally
results = chain.batch([
    {"text": "Gradient descent minimizes the loss function..."},
    {"text": "Dropout randomly zeros activations during training..."},
    {"text": "Batch normalization normalizes layer inputs..."},
])
for r in results:
    print(r)
```

---

### 4.2 RunnablePassthrough — passing input unchanged

`RunnablePassthrough` forwards its input as-is. This sounds useless, but it is essential
when you need to keep the original input alongside a transformed version.

```python
from langchain_core.runnables import RunnablePassthrough

# Adding the original question to the output alongside the answer
chain = (
    RunnablePassthrough.assign(
        answer=prompt | llm | parser  # runs the chain and assigns result to "answer"
    )
)

result = chain.invoke({"text": "Transformers use self-attention..."})
# → {
#     "text": "Transformers use self-attention...",  ← original input preserved
#     "answer": "Transformers process sequences..."   ← chain output added
#   }
```

---

### 4.3 RunnableParallel — multiple branches at once

`RunnableParallel` runs multiple chains on the same input simultaneously and returns a
dict of results. This is how you build "fan-out" pipelines.

```python
from langchain_core.runnables import RunnableParallel

# Two different chains on the same input, run in parallel
summary_chain    = ChatPromptTemplate.from_messages([
    ("system", "Summarize in one sentence."),
    ("human",  "{text}")
]) | llm | parser

key_points_chain = ChatPromptTemplate.from_messages([
    ("system", "Extract 3 key concepts as bullet points."),
    ("human",  "{text}")
]) | llm | parser

# Run both chains on the same input simultaneously
parallel_chain = RunnableParallel(
    summary    = summary_chain,
    key_points = key_points_chain,
)

result = parallel_chain.invoke({"text": "Attention mechanism allows..."})
print(result["summary"])     # → "Attention..."
print(result["key_points"])  # → "• Query-Key-Value\n• Scaled dot-product\n..."
```

---

### 4.4 RunnableLambda — custom Python functions in chains

Any Python function can be wrapped as a Runnable using `RunnableLambda`. This lets you
inject custom processing steps between model calls.

```python
from langchain_core.runnables import RunnableLambda

def word_count(text: str) -> dict:
    """Custom processing step: add word count to the output."""
    return {"text": text, "word_count": len(text.split())}

def format_report(data: dict) -> str:
    """Another custom step: format the final output."""
    return f"[{data['word_count']} words] {data['text']}"

# Wrap functions as Runnables and compose them
processing_chain = (
      (prompt | llm | parser)           # chain output is a string
    | RunnableLambda(word_count)        # passes string to word_count, gets dict back
    | RunnableLambda(format_report)     # passes dict to format_report, gets string back
)

result = processing_chain.invoke({"text": "Explain neural networks."})
print(result)  # → "[47 words] A neural network is a..."
```

---

### 4.5 Building multi-step pipelines

Here is a realistic multi-step pipeline that shows LCEL's power:

```python
# Pipeline: translate then summarize
translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate the following text to English. Output only the translation."),
    ("human",  "{original_text}"),
])

summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize this in one sentence."),
    ("human",  "{english_text}"),
])

# Chain them: first translate, then summarize
pipeline = (
    translate_prompt | llm | parser                             # step 1: translate
    | RunnableLambda(lambda t: {"english_text": t})             # reformat output as dict
    | summarize_prompt | llm | parser                           # step 2: summarize
)

result = pipeline.invoke({"original_text": "La red neuronal aprende patrones..."})
print(result)  # → "Neural networks learn patterns from data..."
```

---

### 4.6 Why LCEL over the old chain classes

The old LangChain had `LLMChain`, `SequentialChain`, `ConversationChain`, etc. These were
deprecated for years and have since been removed from `langchain.chains` entirely in
current LangChain releases (`from langchain.chains import LLMChain` now raises
`ModuleNotFoundError`), so there is no fallback to the old way even if you wanted one.
Here is why LCEL replaced them:

| Feature | Old LLMChain | LCEL |
|---|---|---|
| Streaming | Required special setup | Built into every Runnable |
| Async | Required `arun()` separately | `.ainvoke()`, `.astream()` built-in |
| Batching | Not built-in | `.batch()` built-in |
| Type safety | Loose (strings everywhere) | Typed inputs/outputs |
| Composability | Special chain classes for each pattern | One `|` operator for everything |
| Debugging | Black box | LangSmith traces every step |

```python
# OLD WAY — removed from current LangChain; shown only for contrast
from langchain.chains import LLMChain  # ModuleNotFoundError today
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(text="explain backprop")  # returns string, no streaming

# NEW WAY — LCEL
chain  = prompt | llm | StrOutputParser()
result = chain.invoke({"text": "explain backprop"})  # full control, streaming works
```

---

## 5. Conversation Memory

### 5.1 The problem: models are stateless

Every API call is independent. The model has no memory of previous calls. To have a
conversation, you must include the full history in every request. As you saw in Phase 01:

```python
history = []

# Turn 1
history.append({"role": "user", "content": "What is a transformer?"})
response = call_api(history)
history.append({"role": "assistant", "content": response})

# Turn 2
history.append({"role": "user", "content": "How does attention work in it?"})
response = call_api(history)  # sends ALL history every time
```

LangChain automates this with `InMemoryChatMessageHistory` and
`RunnableWithMessageHistory`.

---

### 5.2 InMemoryChatMessageHistory

`InMemoryChatMessageHistory` is a simple container that stores a list of messages in
RAM. When the process restarts, the history is gone.

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# Create a history store for one session
history = InMemoryChatMessageHistory()

# Add messages to it
history.add_user_message("What is a transformer model?")
history.add_ai_message("A transformer model uses self-attention to process sequences.")
history.add_user_message("How does attention work?")

# Read the messages back
for msg in history.messages:
    role = "User" if isinstance(msg, HumanMessage) else "AI"
    print(f"{role}: {msg.content}")

# Clear if needed
history.clear()
```

For **multiple sessions** (e.g. different users), you maintain a dict keyed by session_id:

```python
# Session store — maps session_id to its history object
session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Get or create a history object for this session."""
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# Usage
history_alice = get_session_history("alice-123")
history_bob   = get_session_history("bob-456")
# Each user has their own isolated conversation history
```

---

### 5.3 RunnableWithMessageHistory — memory-aware chains

> **Note:** LangChain has since marked `RunnableWithMessageHistory` deprecated in favor
> of LangGraph's built-in persistence (constructing one now prints a
> `LangChainDeprecationWarning`). It still works, and this phase teaches it deliberately:
> LangGraph itself is the subject of Phase 05, not this one. Once you reach Phase 05,
> revisit this section and compare the two approaches.

`RunnableWithMessageHistory` wraps any LCEL chain and automatically:
1. Loads the conversation history before each call
2. Injects it into the `MessagesPlaceholder` in your prompt
3. Saves the new human message and AI response to history after each call

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()

# Step 1: Build a prompt with a history placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),  # ← history injected here
    ("human",  "{question}"),
])

# Step 2: Build the base chain (no memory yet)
base_chain = prompt | llm | parser

# Step 3: Wrap with memory management
session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# This wrapper handles loading/saving history automatically
chain_with_memory = RunnableWithMessageHistory(
    base_chain,
    get_history,
    input_messages_key="question",   # which key in your input dict is the user message
    history_messages_key="history",  # which MessagesPlaceholder variable to fill
)

# Step 4: Use it — the session_id goes in config, not the input
config = {"configurable": {"session_id": "user-alice"}}

reply1 = chain_with_memory.invoke({"question": "What is gradient descent?"}, config=config)
print("Turn 1:", reply1)

reply2 = chain_with_memory.invoke({"question": "What is the learning rate?"}, config=config)
print("Turn 2:", reply2)
# The model remembers the context from Turn 1 — it knows you are discussing optimization

# Different user — completely separate history
config_bob = {"configurable": {"session_id": "user-bob"}}
reply = chain_with_memory.invoke({"question": "What is self-attention?"}, config=config_bob)
print("Bob's first message:", reply)
```

---

### 5.4 Memory strategies

**Buffer memory (keep everything):**  
The default: every message is kept. Simple, but the context window fills up over time.

```python
# What is in the history after N turns:
# [Human: "What is X?", AI: "X is...", Human: "And Y?", AI: "Y is...", ...]
# Context window fills proportionally to conversation length
```

**Token-trimmed memory (keep last N tokens):**  
When the history gets too long, trim the oldest messages to stay within budget.

```python
from langchain_core.messages import trim_messages

def get_trimmed_history(session_id: str, max_tokens: int = 2000):
    """Get history trimmed to the last max_tokens tokens."""
    history = get_history(session_id)
    trimmed = trim_messages(
        history.messages,
        max_tokens=max_tokens,
        token_counter=llm,              # uses the model's tokenizer
        strategy="last",                # keep the most recent messages
        start_on="human",               # always start with a human message
        include_system=True,            # keep the system message
    )
    return trimmed
```

**Summary memory (summarize old turns):**  
When history grows large, summarize older turns into a single message. This compresses
history while retaining the gist of the conversation.

```python
async def summarize_and_trim(history: InMemoryChatMessageHistory, keep_last: int = 4):
    """
    If history is long, summarize all but the last `keep_last` messages
    and replace them with a single summary message.
    """
    messages = history.messages
    if len(messages) <= keep_last:
        return  # not long enough to need summarizing
    
    to_summarize = messages[:-keep_last]
    recent       = messages[-keep_last:]
    
    summarize_chain = ChatPromptTemplate.from_messages([
        ("system", "Summarize this conversation briefly for context:"),
        ("human",  "{conversation}"),
    ]) | llm | StrOutputParser()
    
    conversation_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in to_summarize
    )
    summary = await summarize_chain.ainvoke({"conversation": conversation_text})
    
    # Replace history with summary + recent messages
    history.clear()
    history.add_ai_message(f"[Earlier conversation summary: {summary}]")
    for msg in recent:
        history.messages.append(msg)
```

---

### 5.5 Streaming with memory

Memory works seamlessly with streaming:

```python
# Stream the response token by token while still maintaining history
config = {"configurable": {"session_id": "user-alice"}}

print("Response: ", end="")
for token in chain_with_memory.stream({"question": "Explain dropout in detail."}, config=config):
    print(token, end="", flush=True)
print()
# History is updated automatically after the stream completes
```

---

## 6. Document Loaders

### 6.1 The Document object

When you load a file with LangChain, you get a list of `Document` objects. Understanding
this object is fundamental to everything in this phase and Phase 04 (RAG).

```python
from langchain_core.documents import Document

# Every loader produces this type
doc = Document(
    page_content="The transformer architecture was introduced in 2017...",
    metadata={
        "source": "/path/to/file.pdf",
        "page": 1,
        # any other metadata the loader adds
    }
)

print(doc.page_content)  # the text
print(doc.metadata)      # source info, page numbers, etc.
```

Loaders return `list[Document]`: one Document per page (for PDFs), one per row (for
CSV), or one for the whole file (for text).

---

### 6.2 TextLoader — plain text files

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader(
    file_path="paper.txt",
    encoding="utf-8",  # specify encoding if needed
)

docs = loader.load()  # returns list[Document]

print(f"Loaded {len(docs)} document(s)")
print(f"Characters: {len(docs[0].page_content)}")
print(f"Metadata:   {docs[0].metadata}")
# → Metadata: {'source': 'paper.txt'}
```

---

### 6.3 PyPDFLoader — PDF files

PDFs are the most common document format in enterprise workflows. `PyPDFLoader` splits
the PDF into one `Document` per page.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("report.pdf")
docs   = loader.load()

print(f"Pages: {len(docs)}")
for i, doc in enumerate(docs):
    print(f"Page {i+1}: {len(doc.page_content)} chars, metadata: {doc.metadata}")
# → Page 1: 2840 chars, metadata: {'source': 'report.pdf', 'page': 0}
# → Page 2: 3120 chars, ...

# Access specific page
page_3_text = docs[2].page_content
```

> **`PyPDFLoader` vs `PDFMinerLoader` vs `PyMuPDFLoader`:**  
> `PyPDFLoader` is the simplest and handles most PDFs. For scanned PDFs (images),
> you need an OCR-based loader. For better layout preservation, try `PyMuPDFLoader`.
> Start with `PyPDFLoader`; switch only if text extraction quality is poor.

---

### 6.4 WebBaseLoader — web pages

`WebBaseLoader` fetches a URL and extracts its text content, stripping HTML tags.

```python
from langchain_community.document_loaders import WebBaseLoader

# Single URL
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)")
docs   = loader.load()

print(f"Loaded {len(docs)} document(s)")
print(f"Content preview: {docs[0].page_content[:300]}")

# Multiple URLs at once
loader = WebBaseLoader([
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
])
docs = loader.load()
print(f"Loaded {len(docs)} documents from {len(docs)} URLs")
```

**Note on web loading:** `WebBaseLoader` works well for article-style pages. For
JavaScript-heavy SPAs (React/Vue apps), it will only get the initial HTML, not rendered
content. For those, you need a browser-based loader like `SeleniumURLLoader`.

---

### 6.5 CSVLoader — CSV files

```python
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    file_path="data.csv",
    csv_args={"delimiter": ",", "quotechar": '"'},
    # Each row becomes a Document; columns are formatted as "key: value\nkey: value"
    source_column="url",  # use this column as the source in metadata
)

docs = loader.load()
print(f"Loaded {len(docs)} rows")
print(docs[0].page_content)
# → "name: John Smith\nemail: john@example.com\nsubject: Need help with billing"
```

---

### 6.6 DirectoryLoader — loading entire folders

```python
from langchain_community.document_loaders import DirectoryLoader

# Load all .txt files in a directory and its subdirectories
loader = DirectoryLoader(
    path="./documents/",
    glob="**/*.txt",         # pattern to match files
    loader_cls=TextLoader,   # which loader to use for matched files
    show_progress=True,      # show a progress bar
    use_multithreading=True, # load files in parallel (faster for many files)
)

docs = loader.load()
print(f"Loaded {len(docs)} documents from directory")
```

---

## 7. Text Splitters

### 7.1 Why splitting is necessary

Documents are almost always too long to fit in a single prompt. Even with large context
windows (128k+ tokens), there are reasons to split:

1. **Cost:** Sending 50,000 tokens every turn is expensive
2. **Quality:** Models perform better on relevant chunks than on diluted large contexts
3. **RAG (Phase 04):** You can only retrieve relevant chunks if you have chunks to retrieve
4. **Memory:** For conversation context, you only want the relevant piece, not the whole doc

The goal is to split into chunks that are:
- Small enough to fit in context along with the question and answer
- Large enough to be semantically meaningful
- Overlapping slightly so context is not lost at chunk boundaries

---

### 7.2 RecursiveCharacterTextSplitter — the default choice

This is the recommended splitter for most use cases. It tries to split on natural
boundaries (paragraphs → sentences → words → characters) in order, only going smaller
when necessary.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # target max characters per chunk
    chunk_overlap=200,     # characters to overlap between chunks
    
    # Separators tried in order — splits on \n\n first (paragraphs),
    # then \n (lines), then ". " (sentences), then " " (words), then "" (chars)
    separators=["\n\n", "\n", ". ", " ", ""],
    
    length_function=len,   # how to measure length (len = characters)
    # use tiktoken-based length_function for token-accurate splitting:
    # from langchain_text_splitters import TokenTextSplitter
)

text = """
Transformers have revolutionized natural language processing. The key innovation
is the self-attention mechanism, which allows the model to weigh the importance
of different words when processing a sequence.

The attention score between two tokens is computed as the dot product of their
query and key vectors, scaled by the square root of the dimension. This scaling
prevents the dot products from becoming too large.

Modern large language models like GPT and Claude are based on the decoder-only
transformer architecture. They predict the next token given all previous tokens.
"""

chunks = splitter.split_text(text)
print(f"Chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1} ({len(chunk)} chars):")
    print(chunk[:100] + "...")
```

**Splitting Documents (not raw text):**

When you load documents and then split, use `split_documents()`: it preserves the
metadata (source file, page number) in each chunk.

```python
# Load then split — metadata is preserved
loader  = PyPDFLoader("research_paper.pdf")
docs    = loader.load()            # list of Documents, one per page
chunks  = splitter.split_documents(docs)  # list of Documents, smaller chunks

print(f"Pages: {len(docs)} → Chunks: {len(chunks)}")
# Every chunk has the source and page from its parent Document
print(chunks[0].metadata)  # → {'source': 'research_paper.pdf', 'page': 0}
print(chunks[0].page_content[:200])
```

---

### 7.3 chunk_size and chunk_overlap — the critical tradeoff

```
         chunk_overlap
         │←────────────→│
         
┌─────────────────────────────┐
│         Chunk 1             │  chunk_size = 1000 chars
└─────────────────────────────┘
               ┌──────────────────────────────┐
               │         Chunk 2              │
               └──────────────────────────────┘
               │←────────────→│
                chunk_overlap
```

**Why overlap?** Without overlap, a sentence split across two chunks loses context.
With overlap, each chunk contains the tail of the previous chunk, so no sentence is
stranded without context.

**Choosing chunk_size:**

| Use case | Recommended chunk_size |
|---|---|
| Q&A over technical docs | 500–1000 chars (~125–250 tokens) |
| Summarization | 2000–4000 chars (~500–1000 tokens) |
| Chat over book chapters | 1500–3000 chars |
| Code files | 500–1500 chars (split by function) |

**Choosing chunk_overlap:**  
A common rule: 10-20% of `chunk_size`. For `chunk_size=1000`, use `chunk_overlap=150`
to `chunk_overlap=200`.

---

### 7.4 Token-based splitting

Character count is not the same as token count. For precision, split by tokens:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

# Use tiktoken to count tokens instead of characters
tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

def token_length(text: str) -> int:
    return len(tokenizer.encode(text))

token_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,         # 256 TOKENS per chunk (not characters)
    chunk_overlap=32,       # 32 token overlap
    length_function=token_length,  # ← swap character counting for token counting
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks = token_splitter.split_text(long_text)
# Each chunk is now at most 256 tokens — precise for context window math
```

---

## 8. LangSmith — Tracing & Observability

### 8.1 What is LangSmith and why you need it

When you call an LLM API directly, you see the input and output. But in a multi-step
LangChain pipeline, you need to see:

- What exactly was sent to the model at each step (the formatted prompt, with all
  variables substituted)
- What came back at each step
- How many tokens each step used and what it cost
- How long each step took
- Where exactly a pipeline failed when something goes wrong

LangSmith is LangChain's own observability platform. It captures all of this
automatically, with zero code changes, just by setting environment variables.

---

### 8.2 Setup (two environment variables)

```bash
# In your .env file
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your_key_here
LANGCHAIN_PROJECT=my-project-name   # optional — groups traces together
```

That is all. Every `chain.invoke()`, `chain.stream()`, every model call, and every
retriever call is traced automatically. Go to https://smith.langchain.com to see them.

```python
from dotenv import load_dotenv
load_dotenv()

# Everything below is automatically traced — no other changes needed
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm    = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
chain  = prompt | llm | StrOutputParser()

# This call appears in LangSmith with full input/output/token/latency details
result = chain.invoke({"question": "What is attention?"})
```

---

### 8.3 What you see in LangSmith

For every traced run, LangSmith shows:

```
Run: chain.invoke(...)
├── ChatPromptTemplate.invoke(...)      ← formatted prompt with variables substituted
│   └── input:  {"question": "What is attention?"}
│   └── output: [SystemMessage(...), HumanMessage("What is attention?")]
│
├── ChatOpenAI.invoke(...)              ← model call
│   └── input:  [SystemMessage, HumanMessage]
│   └── output: AIMessage("Attention is a mechanism...")
│   └── tokens: 45 in + 87 out = 132 total
│   └── cost:   $0.0000198
│   └── time:   1.23s
│
└── StrOutputParser.invoke(...)         ← parser
    └── input:  AIMessage("Attention is a mechanism...")
    └── output: "Attention is a mechanism..."
```

This visibility is invaluable when debugging. Instead of "why is my chain giving wrong
answers", you can see exactly what prompt was sent to the model.

---

### 8.4 Adding metadata to traces

You can add custom metadata to group and filter traces:

```python
# Add run name and tags for filtering in LangSmith
result = chain.invoke(
    {"question": "What is attention?"},
    config={
        "run_name": "doc-chat-production",   # shows in LangSmith as the run name
        "tags": ["production", "v2.1"],       # for filtering
        "metadata": {                          # custom fields
            "user_id": "alice-123",
            "document": "transformer_paper.pdf",
        }
    }
)
```

---

### 8.5 The @traceable decorator — tracing custom functions

For functions that are not LangChain Runnables (e.g., your own preprocessing code),
use the `@traceable` decorator:

```python
from langsmith import traceable

@traceable(name="preprocess-document", tags=["preprocessing"])
def preprocess(text: str) -> str:
    """Custom preprocessing — will appear as a step in LangSmith traces."""
    text = text.strip()
    text = " ".join(text.split())   # normalize whitespace
    return text

@traceable(name="full-pipeline")
def full_pipeline(raw_text: str, question: str) -> str:
    clean_text = preprocess(raw_text)   # appears as a sub-trace
    result     = chain.invoke({"context": clean_text, "question": question})
    return result
```

---

## 9. Key Takeaways

After completing this phase, you understand:

1. **LangChain solves the composition problem.** It provides standard building blocks
   (prompts, models, parsers, memory, loaders) and the `|` operator to compose them.
   Do not use it for simple single-call scripts.

2. **LCEL is the way.** The old chain classes (`LLMChain`, `ConversationChain`) are
   deprecated. Every new LangChain code should use LCEL with the `|` operator.

3. **Every LangChain component is a Runnable.** They all share `.invoke()`, `.stream()`,
   `.batch()`. Runnables compose: `A | B | C` means C(B(A(input))).

4. **Memory is a wrapper, not a chain feature.** `RunnableWithMessageHistory` wraps
   any LCEL chain. The chain itself is stateless; the wrapper adds state. Session
   management is your responsibility: use a dict keyed by session_id.

5. **Documents are `page_content` + `metadata`.** Every loader produces this object.
   Every splitter consumes and produces it. Metadata (source, page) is preserved
   through the splitting process, which is crucial for citations in Phase 04.

6. **`RecursiveCharacterTextSplitter` is the default.** Start with `chunk_size=1000`,
   `chunk_overlap=200`. Adjust based on your model's context window and task.

7. **Two env vars is all LangSmith needs.** `LANGCHAIN_TRACING_V2=true` and
   `LANGCHAIN_API_KEY`. After that, every chain call is automatically traced with
   full input/output/token/cost/latency breakdown. Use it from day one.

---

## 10. Practice Exercises

### Exercise 1 — Prompt Translation Pipeline (Easy)
Build an LCEL pipeline that takes a user's text and: (1) detects the language, (2) if
not English, translates it to English, (3) then answers the English version of the
question. Use `RunnableParallel` for step 1+2 and `RunnablePassthrough` to merge.

### Exercise 2 — Session-aware CLI Chat (Medium)
Build a command-line chatbot using `RunnableWithMessageHistory`. Each run should accept
a `--session` argument to continue an existing conversation. Save and load history from
a JSON file so sessions persist across process restarts. (Hint: write a custom
`BaseChatMessageHistory` subclass or use `FileChatMessageHistory` from LangChain.)

### Exercise 3 — Multi-format Document Loader (Medium)
Build a `SmartLoader` class with a single `load(source: str) -> list[Document]` method
that auto-detects the source type:
- If it starts with `http://` or `https://` → use `WebBaseLoader`
- If it ends with `.pdf` → use `PyPDFLoader`
- If it ends with `.csv` → use `CSVLoader`
- Otherwise → use `TextLoader`
Then split all loaded documents with `RecursiveCharacterTextSplitter` and return chunks.

### Exercise 4 — Context-Window-Aware Memory (Hard)
Extend the `DocChat` class from the project with automatic memory management. Before
each chain call, calculate: `total_tokens = doc_tokens + history_tokens + question_tokens`.
If `total_tokens > 0.8 × context_window_limit`:
1. Summarize the oldest half of the history into one `AIMessage`
2. Replace those messages with the summary
3. Then proceed with the call

Log a warning whenever the summarization triggers.

---

*Next: Phase 04, RAG & Vector Databases*  
*You will give LLMs access to large document collections that would never fit in a
context window, using vector embeddings and semantic search.*
