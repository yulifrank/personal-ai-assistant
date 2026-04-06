<div align="center">

# 🤖 Personal AI Assistant

### A smart personal assistant powered by LangChain, Google Gemini & Streamlit

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.2+-green?style=for-the-badge&logo=chainlink&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red?style=for-the-badge&logo=streamlit&logoColor=white)
![Gemini](https://img.shields.io/badge/Google_Gemini-Flash-orange?style=for-the-badge&logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

</div>

---

## ✨ What is this?

A fully functional AI-powered personal assistant that can:
- 📄 **Read and understand your documents** (PDF & TXT) using RAG
- 🧠 **Remember your conversation** across multiple turns
- 🛠 **Use real tools** like weather, crypto prices, calculator and more
- 🌐 **Connect to live APIs** for up-to-date information
- 💬 **Answer in your language** — Hebrew or English

---

## 🏗 Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────────┐
│           LangGraph Agent           │
│                                     │
│  ┌─────────┐  ┌──────┐  ┌────────┐ │
│  │   RAG   │  │Tools │  │Memory  │ │
│  │ (FAISS) │  │      │  │ (k=10) │ │
│  └─────────┘  └──────┘  └────────┘ │
└─────────────────────────────────────┘
    │
    ▼
Google Gemini 1.5 Flash
    │
    ▼
Answer to User
```

---

## 🛠 Available Tools

| Tool | Description |
|------|-------------|
| 🧮 `calculator` | Evaluates mathematical expressions |
| 🌤 `get_weather` | Live weather for any city worldwide |
| 📈 `get_crypto_price` | Real-time cryptocurrency prices |
| 💱 `get_exchange_rate` | Currency exchange rates |
| 🔍 `search_wikipedia` | Wikipedia summaries on any topic |
| 📝 `word_counter` | Count words and characters in text |
| 🔑 `keyword_extractor` | Extract key terms from any passage |
| 📋 `bullet_list_formatter` | Format text into bullet points |
| 📅 `get_current_date` | Current date and time |
| ⚖️ `compare_numbers` | Compare two numbers with difference |

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
├── main.py          # Streamlit UI + agent orchestration
├── rag.py           # Document loading + FAISS vector store
├── tools.py         # All custom tools
├── memory.py        # Conversation memory management
├── requirements.txt # Project dependencies
├── docs/            # Drop your documents here
└── .env             # API keys (not committed to git)
```

---

## 💡 Example Usage

```
You:        "What are my working hours according to the contract?"
Assistant:  "According to section 4.2, working hours are Sunday–Thursday, 9:00–18:00"

You:        "What is the weather in Tel Aviv?"
Assistant:  "Temperature: 24°C, Clear sky, Humidity: 58%"

You:        "How much is 15% of 20000?"
Assistant:  "Result: 3000.0"

You:        "What is Bitcoin worth right now?"
Assistant:  "Bitcoin: $67,432 — up 2.3% in the last 24h"
```

---

## 🧠 How RAG Works

1. **Load** — reads your PDF or TXT file
2. **Split** — divides text into 500-character chunks with 50-char overlap
3. **Embed** — converts each chunk into a vector using Gemini Embeddings
4. **Store** — saves all vectors in FAISS for fast search
5. **Retrieve** — finds the 3 most relevant chunks for each question

---

## 🔒 Security Notes

- Never commit your `.env` file
- The `.gitignore` excludes all sensitive files automatically
- API keys are loaded at runtime via `python-dotenv`

---

## 🛣 Roadmap

- [ ] Support for multiple documents simultaneously
- [ ] Export conversation history to PDF
- [ ] Voice input support
- [ ] LangGraph multi-agent architecture
- [ ] Deploy to Streamlit Cloud

---

## 👩‍💻 Built With

- [LangChain](https://langchain.com) — LLM framework
- [LangGraph](https://langgraph.com) — Agent orchestration
- [Google Gemini](https://aistudio.google.com) — LLM & Embeddings
- [FAISS](https://faiss.ai) — Vector similarity search
- [Streamlit](https://streamlit.io) — Web UI framework

---

<div align="center">
Made with ❤️ and lots of ☕
</div>
