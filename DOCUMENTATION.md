# 📚 Project Documentation — Personal AI Assistant

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [File-by-File Breakdown](#3-file-by-file-breakdown)
   - [main.py](#mainpy)
   - [agents.py](#agentspy)
   - [tools.py](#toolspy)
   - [rag.py](#ragpy)
   - [memory.py](#memorypy)
4. [Data & Message Flow](#4-data--message-flow)
5. [LangChain & LangGraph Usage](#5-langchain--langgraph-usage)
6. [Key Design Decisions](#6-key-design-decisions)

---

## 1. Project Overview

A multi-agent AI assistant built with Python, LangChain, LangGraph, and Streamlit.
Users can chat, upload documents (PDF/TXT), and receive answers from a specialized agent
selected automatically by a supervisor. Every step is logged live in the UI.

**Entry point:** `streamlit run main.py`

---

## 2. Architecture

```
User types a question
        │
        ▼
┌───────────────────────────────────────────────┐
│                   main.py                      │
│  1. Check if document is loaded               │
│  2. Show MemorySaver status                   │
│  3. Call supervisor → pick agent              │
│  4. Run chosen agent with thread_id           │
│  5. Parse & display tool calls in UI          │
└───────────────┬───────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────┐
│           Supervisor (agents.py → route())     │
│  Sends question to Gemini with routing prompt  │
│  Returns one word: research / finance /        │
│                    document / utility          │
└───────┬───────────────────────────────────────┘
        │
        ▼  (based on routing decision)
┌───────────────────────────────────────────────────────────┐
│                                                            │
│  🔍 Research Agent    💰 Finance Agent                    │
│     search_wikipedia     calculator                        │
│     get_weather          compare_numbers                   │
│                          get_crypto_price                  │
│                          get_exchange_rate                 │
│                                                            │
│  📝 Document Agent    🛠️ Utility Agent                    │
│     search_document      get_current_date                  │
│     word_counter                                           │
│     summarize_request                                      │
│     bullet_list_formatter                                  │
│     keyword_extractor                                      │
│                                                            │
└────────────────────────────────────────────────────────────┘
        │
        ▼
  Each agent runs a ReAct loop (LangGraph):
  Think → Call tool → Observe result → Think again → Answer
```

---

## 3. File-by-File Breakdown

---

### `main.py`

**Role:** Orchestrator — manages the full pipeline and Streamlit UI.

#### SSL Configuration (lines 1–5)
```python
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = certifi.where()
```
Must be set **before any imports** that trigger network calls.
Tells Python, requests, httpx, and gRPC (used by Google SDK) where to find trusted SSL certificates.
Required on Windows where Python does not use the OS certificate store by default.

#### `extract_text(content)` function
```python
def extract_text(content) -> str:
```
Gemini 2.5 Flash returns content in two possible formats:
- Plain string: `"Hello"`
- List of parts: `[{"type": "text", "text": "Hello", "extras": {"signature": ...}}]`

This function normalizes both formats into a plain string.

#### Session State
```python
st.session_state.thread_id   # UUID — unique ID per conversation session
st.session_state.messages    # list of {role, content} dicts for UI display
st.session_state.retriever   # FAISS retriever (None if no document uploaded)
st.session_state.agents      # dict of built LangGraph agents
```

#### Document Upload Flow
```
User uploads file
  → write to temp file
  → get_retriever(tmp_path)         ← rag.py builds FAISS index
  → make_search_document_tool(...)  ← tools.py creates tool bound to retriever
  → build_agents(llm, search_tool)  ← agents.py rebuilds Document Agent with new tool
  → delete temp file
```
The agents are **rebuilt** on every upload so the Document Agent gets the new retriever.

#### Per-request Pipeline (inside chat input block)
| Step | What happens |
|------|-------------|
| 1 | Show document status |
| 2 | Show MemorySaver thread info |
| 3 | Log full prompt in expander |
| 4 | Call `route()` → supervisor picks agent |
| 5 | `agent.invoke(messages, config={thread_id})` |
| 6 | Log all raw response messages |
| 7 | Parse and display each tool call + result |
| 8 | Extract final text answer, update UI |

#### Clear Conversation
```python
st.session_state.thread_id = str(uuid.uuid4())  # new UUID = fresh memory
st.session_state.agents = build_agents(llm)      # fresh MemorySaver
```

---

### `agents.py`

**Role:** Defines all agents, the supervisor prompt, and the routing logic.

#### `AGENT_REGISTRY`
A dict mapping agent keys to their configuration:
```python
{
    "research": { "emoji", "name", "description", "tools", "prompt" },
    "finance":  { ... },
    "document": { ... },
    "utility":  { ... },
}
```
The `prompt` field is the system prompt injected into each agent's LangGraph graph.
It explicitly tells the agent **which tool to use for which situation**.

#### `SUPERVISOR_PROMPT`
A strict prompt that asks Gemini to return **only one word** — the agent name.
Example output: `research`

#### `build_agents(llm, search_document_tool=None)`
```python
checkpointer = MemorySaver()
agents[key] = create_react_agent(llm, tools, prompt=..., checkpointer=checkpointer)
```
- Creates one shared `MemorySaver` instance for all agents
- All agents share the same `thread_id` namespace
- If `search_document_tool` is passed → prepended to Document Agent's tool list

#### `route(llm, user_input)`
```python
response = llm.invoke([SystemMessage(SUPERVISOR_PROMPT), HumanMessage(user_input)])
# parse response → return agent_key
```
- Sends **only the raw user question** to the supervisor (not history, not RAG)
- Parses the response, defaults to `"research"` if unrecognized

---

### `tools.py`

**Role:** Defines all 11 tools available to agents.

#### `make_search_document_tool(retriever)` — Factory function
```python
def make_search_document_tool(retriever):
    @tool
    def search_document(query: str) -> str:
        docs = retriever.invoke(query)
        ...
    return search_document
```
Uses a **closure** — the inner function captures the `retriever` variable from the outer scope.
This allows creating a new tool each time a document is uploaded, bound to the correct FAISS retriever.

#### `@tool` decorator (LangChain)
Converts a regular Python function into a LangChain tool:
1. The **function name** becomes the tool name Gemini calls
2. The **docstring** is sent to Gemini as the tool description — this is how the model decides when to use it
3. The **type hints** are converted to JSON Schema for structured calling

#### Tools by agent

| Tool | Agent | API / Logic |
|------|-------|-------------|
| `search_document` | 📝 Document | FAISS vector search |
| `search_wikipedia` | 🔍 Research | Wikipedia REST API |
| `get_weather` | 🔍 Research | wttr.in API |
| `calculator` | 💰 Finance | Python `eval()` (sandboxed) |
| `compare_numbers` | 💰 Finance | Pure Python |
| `get_crypto_price` | 💰 Finance | CoinGecko API |
| `get_exchange_rate` | 💰 Finance | open.er-api.com |
| `get_current_date` | 🛠️ Utility | `datetime.datetime.now()` |
| `word_counter` | 📝 Document | Pure Python |
| `summarize_request` | 📝 Document | Returns instruction string |
| `bullet_list_formatter` | 📝 Document | Pure Python |
| `keyword_extractor` | 📝 Document | Frequency analysis, stopwords |

---

### `rag.py`

**Role:** Builds a FAISS vector store from an uploaded document and returns a retriever.

#### Pipeline
```
file_path
    │
    ▼
load_documents()
    PyPDFLoader (PDF) or TextLoader (TXT)
    → list of Document objects
    │
    ▼
RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    → splits into ~500-character chunks
    → 50-character overlap prevents cutting sentences mid-thought
    │
    ▼
GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    → converts each chunk to a vector (numerical representation of meaning)
    │
    ▼
FAISS.from_documents(chunks, embeddings)
    → stores all vectors in an in-memory FAISS index
    │
    ▼
vector_store.as_retriever(search_kwargs={"k": 3})
    → returns top 3 most semantically similar chunks per query
```

#### Key parameters
| Parameter | Value | Effect |
|-----------|-------|--------|
| `chunk_size` | 500 | Max characters per chunk |
| `chunk_overlap` | 50 | Characters shared between adjacent chunks |
| `k` | 3 | Number of chunks returned per query |
| Embedding model | `gemini-embedding-001` | Google's text embedding model |

---

### `memory.py`

**Role:** Legacy module — no longer used by the main pipeline.

Was replaced by LangGraph's `MemorySaver` checkpointer in `agents.py`.
The file remains in the project but its functions (`create_memory`, `format_history`, `add_to_memory`) are no longer called.

---

## 4. Data & Message Flow

### Full example: "Who was Einstein?"

```
User types: "Who was Einstein?"
    │
    ▼
main.py:
  thread_id = "a3f2c1b9-..."
  full_input = "Who was Einstein?"
    │
    ▼
route(llm, "Who was Einstein?")
  → Supervisor prompt + question sent to Gemini
  → Gemini returns: "research"
    │
    ▼
agents["research"].invoke(
    {"messages": [("human", "Who was Einstein?")]},
    config={"configurable": {"thread_id": "a3f2c1b9-..."}}
)
    │
    ▼ LangGraph ReAct loop:

[HumanMessage]  "Who was Einstein?"
    ↓
[AIMessage]     tool_calls: [search_wikipedia("Albert Einstein")]
    ↓
[ToolMessage]   "Wikipedia — Albert Einstein: physicist, born 1879..."
    ↓
[AIMessage]     "Albert Einstein was a German-born physicist..."
    │
    ▼
main.py extracts final AIMessage content → displays answer
MemorySaver stores all 4 messages under thread_id
```

### Next question in same session: "What did he discover?"

LangGraph automatically loads the previous 4 messages from `MemorySaver`
and prepends them. The agent already knows who "he" refers to.

---

## 5. LangChain & LangGraph Usage

### LangGraph
| Feature | Where used | What it does |
|---------|-----------|-------------|
| `create_react_agent` | `agents.py` | Builds a full ReAct graph (Think→Act→Observe loop) |
| `MemorySaver` | `agents.py` | Persists message history per thread_id automatically |
| `ToolMessage`, `AIMessage`, `HumanMessage` | `main.py` | Message types in the graph state |

### LangChain
| Feature | Where used | What it does |
|---------|-----------|-------------|
| `@tool` decorator | `tools.py` | Converts functions to LLM-callable tools with schema |
| `ChatGoogleGenerativeAI` | `main.py` | Gemini LLM wrapper |
| `GoogleGenerativeAIEmbeddings` | `rag.py` | Text → vector conversion |
| `FAISS` | `rag.py` | Vector similarity search |
| `PyPDFLoader`, `TextLoader` | `rag.py` | Document ingestion |
| `RecursiveCharacterTextSplitter` | `rag.py` | Smart text chunking |

---

## 6. Key Design Decisions

### Why a supervisor pattern instead of one big agent?
Each specialist agent has a **focused system prompt and a small tool set**.
This improves tool selection accuracy — the Finance Agent only sees finance tools,
so it can't accidentally call `search_wikipedia` when it should call `calculator`.

### Why is RAG a tool instead of pre-processing?
When RAG was done in `main.py` before calling the agent, it always searched the document
regardless of whether the question was about the document.
As a `search_document` tool, the Document Agent calls it **only when relevant**.

### Why `make_search_document_tool` (factory) instead of a class?
Each document upload produces a different retriever.
A factory function using a closure is the simplest way to bind a specific retriever
to a tool at runtime without global state.

### Why `MemorySaver` instead of manual memory?
Manual memory required formatting history as a string and injecting it into every prompt.
`MemorySaver` with `thread_id` lets LangGraph handle this automatically —
the agent receives the full message history as structured `HumanMessage`/`AIMessage` objects,
which is more reliable and token-efficient.

### Why `extract_text()`?
Gemini 2.5 Flash returns content as a list of typed parts (`[{"type":"text","text":"...","extras":{...}}]`)
instead of a plain string. Without this normalization function, raw JSON would appear in the UI.
