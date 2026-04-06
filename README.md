<div align="center">

# 🤖 Personal AI Assistant

### A multi-agent personal assistant powered by LangChain, LangGraph, Google Gemini & Streamlit

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Latest-green?style=for-the-badge&logo=chainlink&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0+-yellow?style=for-the-badge&logo=graphql&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red?style=for-the-badge&logo=streamlit&logoColor=white)
![Gemini](https://img.shields.io/badge/Google_Gemini-2.5_Flash-orange?style=for-the-badge&logo=google&logoColor=white)
![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-blueviolet?style=for-the-badge&logo=langchain&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

</div>

---

## ✨ What is this?

A fully functional AI-powered personal assistant built with a **production-grade multi-agent architecture**.
Each question is automatically routed by a supervisor to the most suitable specialist agent.
The entire pipeline is transparent — you can see every decision, tool call, input and output in real time.

- 📄 **Read and understand your documents** (PDF & TXT) using Hybrid RAG
- 🧠 **Shared conversation memory** across all agents — the conversation never feels "cut off"
- 🎯 **Structured supervisor routing** — picks the right agent(s) with a confidence score and reason
- 🔀 **Multi-hop routing** — routes to multiple agents in one turn for complex questions
- 🛠 **Specialist agents** — each with a focused set of real-world tools
- 🔍 **Full transparency** — logs show every decision, tool call, input & output
- 📡 **LangSmith integration** — external tracing, token usage & performance monitoring
- 💬 **Responds in your language** — Hebrew, English, or any other

---

## 🏗 Architecture

```
User Question
      │
      ▼
┌─────────────────────────────────────────────────────┐
│                     main.py                          │
│         (orchestrator — manages full pipeline)       │
│                                                      │
│  1. Document status  →  hybrid RAG ready?            │
│  2. History inject   →  full conversation to agents  │
│  3. Supervisor       →  RoutingDecision (structured) │
│  4. Specialist Agents → call the right tools         │
│  5. Combine answers  →  multi-hop merge              │
│  6. Logging          →  show every step in the UI    │
└─────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│          🎯 Supervisor (Gemini + Structured Output)                  │
│   Returns: { agents: [...], confidence: 0.95, reason: "..." }        │
│   Supports multi-hop → routes to multiple agents per turn            │
└──────────┬──────────────────────────────────────────────────────────┘
           │
     ┌─────┴──────────────────────────┐
     ▼            ▼          ▼        ▼
┌─────────┐ ┌─────────┐ ┌───────┐ ┌────────┐
│🔍 Resea-│ │💰 Finan-│ │📝 Doc │ │🛠️ Util │
│  rch    │ │   ce    │ │  ument│ │  ity   │
├─────────┤ ├─────────┤ ├───────┤ ├────────┤
│Wikipedia│ │Calculat.│ │search │ │Date    │
│Weather  │ │Crypto   │ │_doc   │ │& Time  │
│         │ │Exchange │ │Words  │ │        │
│         │ │Compare  │ │Bullets│ │        │
│         │ │         │ │Keywrds│ │        │
└─────────┘ └─────────┘ └───────┘ └────────┘
       ↑ all agents share full conversation history via MemorySaver
```

---

## 🤖 The Four Agents

| Agent | Emoji | Handles | Tools |
|-------|-------|---------|-------|
| **Research Agent** | 🔍 | General knowledge, world facts, weather | `search_wikipedia`, `get_weather` |
| **Finance & Math Agent** | 💰 | Calculations, crypto, currencies, comparisons | `calculator`, `get_crypto_price`, `get_exchange_rate`, `compare_numbers` |
| **Document & Text Agent** | 📝 | Document Q&A, text analysis, summarizing | `search_document` *(RAG)*, `word_counter`, `summarize_request`, `keyword_extractor`, `bullet_list_formatter` |
| **Utility Agent** | 🛠️ | Date and time | `get_current_date` |

---

## 🛠 All Available Tools

| Tool | Agent | Description |
|------|-------|-------------|
| 🧮 `calculator` | 💰 Finance | Evaluates mathematical expressions |
| ⚖️ `compare_numbers` | 💰 Finance | Compares two numbers and shows difference |
| 📈 `get_crypto_price` | 💰 Finance | Real-time cryptocurrency prices via CoinGecko |
| 💱 `get_exchange_rate` | 💰 Finance | Live currency exchange rates |
| 🔍 `search_wikipedia` | 🔍 Research | Wikipedia summaries — two-step search with URL encoding |
| 🌤 `get_weather` | 🔍 Research | Live weather for any city worldwide |
| 📄 `search_document` | 📝 Document | Hybrid RAG search over the uploaded document |
| 📝 `word_counter` | 📝 Document | Count words and characters in text |
| 🔑 `keyword_extractor` | 📝 Document | Extract key terms from any passage |
| 📋 `bullet_list_formatter` | 📝 Document | Format text into bullet points |
| 📄 `summarize_request` | 📝 Document | Summarize a topic from a document |
| 📅 `get_current_date` | 🛠️ Utility | Current date and time |

---

## 🧩 Key Features in Depth

### 🎯 Structured Supervisor Routing
The supervisor LLM returns a typed `RoutingDecision` Pydantic object — not plain text:
```python
class RoutingDecision(BaseModel):
    agents: List[Literal["research", "finance", "document", "utility"]]
    confidence: float   # 0.0 – 1.0
    reason: str         # one-sentence explanation
```
If confidence is low, the system automatically falls back to the Research Agent.
Multi-hop: for complex questions (e.g. *"What is Bitcoin and how much is it?"*), it routes to **both** Finance and Research in a single turn.

---

### 🧠 Shared Conversation Memory
Each agent is powered by LangGraph's `MemorySaver` checkpointer — but agents also receive the **full session history** injected into their prompt. This means:
- No agent ever loses context of the overall conversation
- Internal tool-calling loops remain clean per agent
- The conversation feels continuous even when a different agent answers

---

### 📚 Hybrid RAG (FAISS + BM25)
Document search combines two retrieval strategies:

| Strategy | Type | Strength |
|----------|------|----------|
| **FAISS** | Semantic (vector) | Finds conceptually similar content |
| **BM25** | Keyword (exact) | Finds specific terms and names |

Results are interleaved — semantic results first (higher weight), then keyword results — and deduplicated before returning the top-k chunks.

---

### 📡 LangSmith Observability
Add your LangSmith key to `.env` for external tracing:
```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=personal-assistant
LANGCHAIN_API_KEY=your_langsmith_key_here
```
At [smith.langchain.com](https://smith.langchain.com) you can inspect every LLM call, token usage, latency, and tool invocations with a full visual trace.

---

## 🔍 Live Debug Logging

Every response includes a full breakdown inside the Streamlit UI:

```
🧠 Multi-agent pipeline running...
  📭 No document uploaded — skipping RAG
  🆕 No previous conversation history
  📤 Full message sent to agents (with history)   ← expandable
  🎯 Supervisor deciding which agent...
  ┌─────────────────────────────────────────┐
  │ Agents:     🔍 research                  │
  │ Confidence: ████████░░ 0.92              │
  │ Reason:     "General knowledge question" │
  └─────────────────────────────────────────┘
  🔍 Running Research Agent...
  📨 Full raw response — 3 messages           ← expandable
  🔧 Tool called: search_wikipedia
     ↳ search_wikipedia — input & output      ← expandable (exact I/O)
✅ 🔍 Research Agent — Done in 2.4s
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/personal-ai-assistant.git
cd personal-ai-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API keys

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional — for LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=personal-assistant
LANGCHAIN_API_KEY=your_langsmith_key_here
```

> Get your free Gemini API key at [aistudio.google.com](https://aistudio.google.com)
> Get your free LangSmith key at [smith.langchain.com](https://smith.langchain.com)

### 5. Run the app

```bash
streamlit run main.py
```

Open your browser at `http://localhost:8501` 🎉

---

## 📁 Project Structure

```
personal-ai-assistant/
│
├── main.py          # Streamlit UI + full pipeline orchestration
├── agents.py        # Multi-agent definitions + structured supervisor routing
├── rag.py           # Hybrid RAG: document loading, FAISS + BM25 retrieval
├── tools.py         # All 12 custom tools (with SSL & timeout hardening)
├── memory.py        # Legacy memory module (replaced by MemorySaver)
├── requirements.txt # Project dependencies
└── .env             # API keys (not committed to git)
```

---

## 🧠 How Hybrid RAG Works

1. **Load** — reads your PDF or TXT file
2. **Split** — divides text into 500-character chunks with 50-char overlap
3. **Embed** — converts each chunk into a vector using `gemini-embedding-001`
4. **Store** — saves all vectors in FAISS + builds a BM25 keyword index
5. **Retrieve** — merges FAISS (semantic) + BM25 (keyword) results, deduplicates, returns top-k chunks
6. **Inject** — the Document Agent receives the chunks as context via `search_document` tool

---

## 💡 Example Routing

| Question | Agents | Tool Used |
|----------|--------|-----------|
| "Who was Einstein?" | 🔍 Research | `search_wikipedia` |
| "What is Bitcoin worth?" | 💰 Finance | `get_crypto_price` |
| "How much is 18% of 5000?" | 💰 Finance | `calculator` |
| "What's the weather in NY?" | 🔍 Research | `get_weather` |
| "What day is it today?" | 🛠️ Utility | `get_current_date` |
| "Summarize my document" | 📝 Document | `search_document` |
| "What is BTC and how much?" | 🔍 + 💰 Multi-hop | `search_wikipedia` + `get_crypto_price` |

---

## ⚙️ Technical Notes

### SSL on Windows
Google's API requires SSL verification. The app configures `certifi` certificates at startup **and** all outbound HTTP requests in `tools.py` explicitly pass `verify=certifi.where()`:
```python
# main.py — process-level env vars
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = certifi.where()

# tools.py — request-level (belt + suspenders)
requests.get(url, verify=certifi.where(), timeout=10, headers=_HEADERS)
```

### Wikipedia Tool — Two-Step Search
The `search_wikipedia` tool uses a two-step approach for reliability:
1. **Title lookup** via the MediaWiki search API (handles Hebrew, special characters, disambiguation)
2. **Summary fetch** via the REST API using the exact resolved title

All queries are URL-encoded with `urllib.parse.quote` and sent with a proper `User-Agent` header to avoid `403 Forbidden` errors.

### Gemini Content Format
Gemini 2.5 Flash returns content as a list of parts instead of a plain string. The app normalizes both formats with a custom `extract_text()` function.

### Free Tier Limits
- ~15 requests per minute
- ~1,500 requests per day
- Model: `gemini-2.5-flash` (chat) + `gemini-embedding-001` (embeddings)

---

## 🔒 Security Notes

- Never commit your `.env` file
- The `.gitignore` excludes all sensitive files automatically
- API keys are loaded at runtime via `python-dotenv`

---

## 🛣 Roadmap

- [x] Single agent with tools
- [x] RAG — document upload & search
- [x] Conversation memory (LangGraph `MemorySaver`)
- [x] Multi-agent architecture with structured supervisor routing
- [x] Multi-hop routing — multiple agents per turn
- [x] Hybrid RAG — FAISS + BM25 combined retrieval
- [x] Structured output with confidence scores
- [x] Shared conversation context across agents
- [x] LangSmith integration for external tracing
- [x] SSL hardening for all HTTP requests
- [x] Full debug logging in the UI
- [ ] Support for multiple documents simultaneously
- [ ] Export conversation history
- [ ] Voice input support
- [ ] Deploy to Streamlit Cloud

---

## 👩‍💻 Built With

- [LangChain](https://langchain.com) — LLM framework & tool definitions
- [LangGraph](https://langgraph.com) — Agent orchestration (`create_react_agent`, `MemorySaver`)
- [Google Gemini](https://aistudio.google.com) — LLM (`gemini-2.5-flash`) & Embeddings (`gemini-embedding-001`)
- [FAISS](https://faiss.ai) — Vector similarity search
- [BM25 (rank-bm25)](https://github.com/dorianbrown/rank_bm25) — Keyword-based retrieval
- [LangSmith](https://smith.langchain.com) — Observability & tracing
- [Streamlit](https://streamlit.io) — Web UI framework
- [Certifi](https://pypi.org/project/certifi/) — SSL certificate management

---

<div align="center">

Built by **Yael Frank** — Software Developer & AI Engineer @ [Helpe](https://helpe.ai)

Made with ❤️ and lots of ☕

</div>
