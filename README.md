<div align="center">

# 🤖 Personal AI Assistant

### A multi-agent personal assistant powered by LangChain, Google Gemini & Streamlit

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Latest-green?style=for-the-badge&logo=chainlink&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0+-yellow?style=for-the-badge&logo=graphql&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red?style=for-the-badge&logo=streamlit&logoColor=white)
![Gemini](https://img.shields.io/badge/Google_Gemini-2.5_Flash-orange?style=for-the-badge&logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

</div>

---

## ✨ What is this?

A fully functional AI-powered personal assistant with a **multi-agent architecture** — each type of question is routed to a specialist agent with the most relevant tools. Includes real-time debug logging so you can see exactly what happens at every step.

- 📄 **Read and understand your documents** (PDF & TXT) using RAG
- 🧠 **Remember your conversation** across multiple turns
- 🎯 **Supervisor routing** — automatically picks the right agent for each question
- 🛠 **Specialist agents** — each with its own focused set of tools
- 🔍 **Full transparency** — logs show every decision, tool call, input & output
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
│  1. RAG  →  searches uploaded document               │
│  2. Memory  →  loads conversation history            │
│  3. Supervisor  →  decides which agent to use        │
│  4. Specialist Agent  →  calls the right tools       │
│  5. Logging  →  shows every step in the UI           │
└─────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────┐
│              🎯 Supervisor (Gemini LLM)              │
│     Reads the question → returns one agent name      │
└──────────┬──────────────────────────────────────────┘
           │
     ┌─────┴──────────────────────────┐
     ▼            ▼          ▼        ▼
┌─────────┐ ┌─────────┐ ┌───────┐ ┌────────┐
│🔍 Resea-│ │💰 Finan-│ │📝 Doc │ │🛠️ Util │
│  rch    │ │   ce    │ │  ument│ │  ity   │
├─────────┤ ├─────────┤ ├───────┤ ├────────┤
│Wikipedia│ │Calculat.│ │Words  │ │Date    │
│Weather  │ │Crypto   │ │Summary│ │        │
│         │ │Exchange │ │Keywrds│ │        │
│         │ │Compare  │ │Bullets│ │        │
└─────────┘ └─────────┘ └───────┘ └────────┘
```

---

## 🤖 The Four Agents

| Agent | Emoji | Handles | Tools |
|-------|-------|---------|-------|
| **Research Agent** | 🔍 | General knowledge, world facts, weather | `search_wikipedia`, `get_weather` |
| **Finance & Math Agent** | 💰 | Calculations, crypto, currencies, comparisons | `calculator`, `get_crypto_price`, `get_exchange_rate`, `compare_numbers` |
| **Document & Text Agent** | 📝 | Text analysis, summarizing, formatting | `word_counter`, `summarize_request`, `keyword_extractor`, `bullet_list_formatter` |
| **Utility Agent** | 🛠️ | Date and time | `get_current_date` |

---

## 🛠 All Available Tools

| Tool | Agent | Description |
|------|-------|-------------|
| 🧮 `calculator` | 💰 Finance | Evaluates mathematical expressions |
| ⚖️ `compare_numbers` | 💰 Finance | Compares two numbers and shows difference |
| 📈 `get_crypto_price` | 💰 Finance | Real-time cryptocurrency prices via CoinGecko |
| 💱 `get_exchange_rate` | 💰 Finance | Live currency exchange rates |
| 🔍 `search_wikipedia` | 🔍 Research | Wikipedia summaries on any topic |
| 🌤 `get_weather` | 🔍 Research | Live weather for any city worldwide |
| 📝 `word_counter` | 📝 Document | Count words and characters in text |
| 🔑 `keyword_extractor` | 📝 Document | Extract key terms from any passage |
| 📋 `bullet_list_formatter` | 📝 Document | Format text into bullet points |
| 📄 `summarize_request` | 📝 Document | Summarize a topic from a document |
| 📅 `get_current_date` | 🛠️ Utility | Current date and time |

---

## 🔍 Live Debug Logging

Every response shows a full breakdown of what happened:

```
🧠 Multi-agent pipeline running...
  📭 No document uploaded — skipping RAG
  🆕 No previous conversation history
  📤 Full user message sent to agent      ← expandable
  🎯 Supervisor deciding which agent...
  🎯 Supervisor decision                  ← expandable (shows routing reason)
  🔍 Running Research Agent...
  📨 Full raw response — 3 messages       ← expandable (all LLM messages)
  🔧 Tool called: search_wikipedia
     ↳ search_wikipedia — input & output  ← expandable (exact input + output)
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

### 4. Set up your API key

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

> Get your free API key at [aistudio.google.com](https://aistudio.google.com)

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
├── agents.py        # Multi-agent definitions + supervisor routing logic
├── rag.py           # Document loading + FAISS vector store
├── tools.py         # All 11 custom tools
├── memory.py        # Conversation memory (last 10 message pairs)
├── requirements.txt # Project dependencies
└── .env             # API keys (not committed to git)
```

---

## 🧠 How RAG Works

1. **Load** — reads your PDF or TXT file
2. **Split** — divides text into 500-character chunks with 50-char overlap
3. **Embed** — converts each chunk into a vector using `gemini-embedding-001`
4. **Store** — saves all vectors in FAISS for fast similarity search
5. **Retrieve** — finds the 3 most relevant chunks for each question

---

## 💡 Example Routing

```
"Who was Einstein?"          → 🔍 Research Agent  → search_wikipedia
"What is Bitcoin worth?"     → 💰 Finance Agent   → get_crypto_price
"How much is 18% of 5000?"   → 💰 Finance Agent   → calculator
"What's the weather in NY?"  → 🔍 Research Agent  → get_weather
"What day is it today?"      → 🛠️ Utility Agent   → get_current_date
"Summarize this text: ..."   → 📝 Document Agent  → summarize_request
```

---

## ⚙️ Technical Notes

### SSL on Windows
Google's API requires SSL verification. The app automatically configures `certifi` certificates at startup:
```python
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = certifi.where()
```

### Gemini Content Format
Gemini 2.5 Flash returns content as a list of parts instead of a plain string. The app handles this with `extract_text()` which normalizes both formats.

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
- [x] Conversation memory
- [x] Multi-agent architecture with supervisor routing
- [x] Full debug logging in the UI
- [ ] Support for multiple documents simultaneously
- [ ] Export conversation history
- [ ] Voice input support
- [ ] Deploy to Streamlit Cloud

---

## 👩‍💻 Built With

- [LangChain](https://langchain.com) — LLM framework & tool definitions
- [LangGraph](https://langgraph.com) — Agent orchestration (`create_react_agent`)
- [Google Gemini](https://aistudio.google.com) — LLM & Embeddings
- [FAISS](https://faiss.ai) — Vector similarity search
- [Streamlit](https://streamlit.io) — Web UI framework

---

<div align="center">
Made with ❤️ and lots of ☕
</div>
