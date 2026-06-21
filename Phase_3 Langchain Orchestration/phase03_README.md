# Phase 03 — Doc Chat with Session Memory

## What this project does
A multi-turn Q&A system over any document (PDF, text, URL, CSV) with:
- Persistent conversation memory per session
- LCEL chain composition (`prompt | llm | parser`)
- Automatic LangSmith tracing of every call

## Project structure
```
phase03_project/
├── doc_chat.py      ← DocChat class + standalone LCEL demos
├── phase03_demo.py          ← 8 demo scenarios
└── phase03_requirements.txt
```

## Quick start

### 1. Install
```bash
pip install -r phase03_requirements.txt
```

### 2. Configure .env
```bash
OPENAI_API_KEY=sk-...

# Optional but recommended — enables LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=phase03-doc-chat
```

### 3. Run
```bash
python phase03_demo.py
```

## API reference

```python
from doc_chat import DocChat

# Create a chat session
chat = DocChat(session_id="alice", model="gpt-4o-mini")

# Load any document
chat.load("paper.pdf")        # PDF file
chat.load("notes.txt")        # text file
chat.load("https://...")      # web page
chat.load("data.csv")         # CSV file

# Chat — memory is maintained automatically
reply = chat.chat("What is the main point?")
reply = chat.chat("Can you elaborate?")  # model remembers context

# Stream token by token
for token in chat.stream("Summarize the key findings."):
    print(token, end="", flush=True)

# Multiple sessions on the same document
chat.chat("What is X?", session_id="alice")
chat.chat("What is Y?", session_id="bob")   # separate history from alice

# Inspect history
chat.show_history()
history = chat.get_history()  # list of {role, content} dicts

# Session management
chat.clear_history()
chat.session_stats()
```

## Concepts demonstrated

| Demo | Concept |
|------|---------|
| Demo 1 | LCEL `\|` operator, RunnableParallel, streaming |
| Demo 2 | TextLoader, WebBaseLoader — loading different source types |
| Demo 3 | RecursiveCharacterTextSplitter, chunk_size/overlap tradeoffs |
| Demo 4 | Document Q&A with streaming output |
| Demo 5 | Multi-turn memory, follow-up questions, show_history() |
| Demo 6 | Multiple independent sessions on the same document |
| Demo 7 | Loading and chatting with a live web page |
| Demo 8 | LangSmith tracing setup and verification |

## The limitation this phase reveals

`DocChat` loads the entire document into the context window. For large documents
this either truncates (losing content) or exceeds the context window limit.

**Phase 04 (RAG)** solves this by indexing documents in a vector database and
retrieving only the relevant chunks for each question — no context window limit.
