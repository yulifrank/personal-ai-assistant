import certifi
import os
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = certifi.where()

import streamlit as st
import tempfile
import time
import uuid
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ToolMessage, AIMessage
from rag import get_retriever
from agents import AGENT_REGISTRY, build_agents, route
from tools import make_search_document_tool

load_dotenv()

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp { background-color: #f0f2f8; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f5f6fa 100%);
    border-right: 1px solid #e2e5f0;
}

/* Title area */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
}
.main-header h1 { margin: 0; font-size: 2rem; }
.main-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.95rem; }

/* Agent badges */
.agent-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 2px;
}
.badge-research  { background:#1e3a5f; color:#60a5fa; border:1px solid #2563eb; }
.badge-finance   { background:#1a3a2a; color:#34d399; border:1px solid #059669; }
.badge-document  { background:#3a2a1a; color:#fb923c; border:1px solid #ea580c; }
.badge-utility   { background:#2a1a3a; color:#c084fc; border:1px solid #9333ea; }

/* Metric cards */
.metric-row {
    display: flex;
    gap: 10px;
    margin: 0.5rem 0;
}
.metric-card {
    flex: 1;
    background: #ffffff;
    border: 1px solid #e2e5f0;
    border-radius: 10px;
    padding: 10px 14px;
    text-align: center;
}
.metric-card .value { font-size: 1.4rem; font-weight: 700; color: #7c3aed; }
.metric-card .label { font-size: 0.7rem; color: #6b7280; margin-top: 2px; }

/* Confidence bar */
.conf-bar-wrap { background:#e5e7eb; border-radius:8px; height:8px; margin:6px 0; overflow:hidden; }
.conf-bar      { height:8px; border-radius:8px; transition:width 0.4s ease; }

/* Tool pill */
.tool-pill {
    display: inline-block;
    background: #f3f4f6;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-family: monospace;
    color: #374151;
    margin: 2px;
}

/* Chat bubbles */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    margin-bottom: 8px;
}

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid #e2e5f0 !important;
    border-radius: 10px !important;
    background: #ffffff !important;
}

/* Status */
[data-testid="stStatus"] {
    border: 1px solid #e2e5f0;
    border-radius: 12px;
    background: #ffffff;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    border: 1px solid #d1d5db;
    background: #ffffff;
    color: #374151;
    transition: all 0.2s;
}
.stButton > button:hover {
    border-color: #7c3aed;
    background: #f5f3ff;
    color: #7c3aed;
}
</style>
""", unsafe_allow_html=True)


def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def confidence_bar(value: float) -> str:
    pct = int(value * 100)
    if value >= 0.8:
        color = "#10b981"
    elif value >= 0.5:
        color = "#f59e0b"
    else:
        color = "#ef4444"
    return (
        f'<div class="conf-bar-wrap">'
        f'<div class="conf-bar" style="width:{pct}%;background:{color};"></div>'
        f'</div>'
        f'<span style="font-size:0.85rem;color:{color};font-weight:600;">{pct}%</span>'
    )


def build_message_with_history(user_input: str, history: list[dict]) -> str:
    """Prepends conversation history so every agent is aware of prior context."""
    if not history:
        return user_input
    lines = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    history_text = "\n".join(lines)
    return (
        f"[Conversation so far]\n{history_text}\n\n"
        f"[Current question]\n{user_input}"
    )


def run_agent_and_log(agent, user_input: str, thread_id: str, agent_key: str) -> tuple[str, list]:
    cfg = AGENT_REGISTRY[agent_key]
    # Each agent gets its own thread namespace to avoid tool-call cross-contamination
    config = {"configurable": {"thread_id": f"{thread_id}-{agent_key}"}}
    response = agent.invoke({"messages": [("human", user_input)]}, config=config)
    all_messages = response.get("messages", [])

    with st.expander(f"📨 {cfg['emoji']} Raw messages — {len(all_messages)} total"):
        for i, msg in enumerate(all_messages):
            msg_type = type(msg).__name__
            color = {"HumanMessage": "#3b82f6", "AIMessage": "#8b5cf6", "ToolMessage": "#f59e0b"}.get(msg_type, "#6b7280")
            st.markdown(
                f'<span style="font-size:0.78rem;background:{color}22;color:{color};'
                f'padding:2px 8px;border-radius:4px;font-weight:600;">[{i}] {msg_type}</span>',
                unsafe_allow_html=True,
            )
            if hasattr(msg, "content") and msg.content:
                st.code(extract_text(msg.content), language="markdown")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                st.markdown("🔧 **Tool calls:**")
                st.json(msg.tool_calls)
            if hasattr(msg, "tool_call_id"):
                st.caption(f"tool_call_id: `{msg.tool_call_id}`")
            st.divider()

    tool_calls_made = []
    for msg in all_messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_made.append(tc)
        if isinstance(msg, ToolMessage):
            matching = next((tc for tc in tool_calls_made if tc["id"] == msg.tool_call_id), None)
            tool_name = matching["name"] if matching else "unknown"
            tool_input = matching["args"] if matching else {}
            st.markdown(
                f'🔧 **Tool:** <span class="tool-pill">{tool_name}</span>',
                unsafe_allow_html=True,
            )
            with st.expander(f"↳ `{tool_name}` — input & output"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📥 Input**")
                    st.json(tool_input)
                with col2:
                    st.markdown("**📤 Output**")
                    st.code(extract_text(msg.content))

    if not tool_calls_made:
        st.warning(
            f"⚠️ **Guardrail:** {cfg['name']} answered without calling any tool — possible hallucination."
        )

    return extract_text(all_messages[-1].content), all_messages


# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Personal AI Assistant", page_icon="🤖", layout="wide")

# ─── Session state ─────────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0
if "agents" not in st.session_state:
    st.session_state.agents = None
if "total_calls" not in st.session_state:
    st.session_state.total_calls = 0
if "total_tools" not in st.session_state:
    st.session_state.total_tools = 0

# ─── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,
)
if st.session_state.agents is None:
    st.session_state.agents = build_agents(llm)

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 Personal AI Assistant</h1>
    <p>Multi-agent system powered by Gemini 2.5 Flash · LangGraph · RAG · LangSmith</p>
</div>
""", unsafe_allow_html=True)

# ─── Stats row ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("💬 Messages", len(st.session_state.messages))
with col2:
    st.metric("🤖 Agent calls", st.session_state.total_calls)
with col3:
    st.metric("🔧 Tool calls", st.session_state.total_tools)
with col4:
    doc_status = f"{st.session_state.num_chunks} chunks" if st.session_state.retriever else "No document"
    st.metric("📄 Document", doc_status)

st.divider()

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # LangSmith
    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    langsmith_key = os.getenv("LANGCHAIN_API_KEY", "")
    if tracing and langsmith_key and langsmith_key != "your_langsmith_key_here":
        st.success("📊 LangSmith **ON**")
        st.caption(f"Project: `{os.getenv('LANGCHAIN_PROJECT', 'default')}`")
    else:
        st.info("📊 LangSmith **OFF**")
        st.caption("Add `LANGCHAIN_API_KEY` to `.env`")

    st.divider()
    st.markdown("## 📄 Document")
    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

    if uploaded_file:
        with st.spinner("Indexing with hybrid search..."):
            suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            retriever, num_chunks = get_retriever(tmp_path)
            st.session_state.retriever = retriever
            st.session_state.num_chunks = num_chunks
            search_tool = make_search_document_tool(st.session_state.retriever)
            st.session_state.agents = build_agents(llm, search_document_tool=search_tool)
            os.unlink(tmp_path)
        st.success(f"✅ {uploaded_file.name}")
        st.markdown(
            f'<div style="font-size:0.78rem;color:#9ca3af;">'
            f'🔢 {num_chunks} chunks &nbsp;|&nbsp; '
            f'🔍 FAISS 60% + BM25 40% &nbsp;|&nbsp; top-4'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("## 🤖 Agents")

    agent_colors = {
        "research": ("badge-research", "#3b82f6"),
        "finance":  ("badge-finance",  "#10b981"),
        "document": ("badge-document", "#f97316"),
        "utility":  ("badge-utility",  "#a855f7"),
    }

    for key, cfg in AGENT_REGISTRY.items():
        badge_class, color = agent_colors[key]
        with st.expander(f"{cfg['emoji']} {cfg['name']}"):
            st.caption(cfg["description"])
            tools_html = "".join(
                f'<span class="tool-pill">{t.name}</span>' for t in cfg["tools"]
            )
            if key == "document":
                if st.session_state.retriever:
                    tools_html += '<span class="tool-pill" style="border-color:#10b981;color:#10b981;">search_document ✅</span>'
                else:
                    tools_html += '<span class="tool-pill" style="opacity:0.4;">search_document ⬜</span>'
            st.markdown(tools_html, unsafe_allow_html=True)

    st.divider()
    st.markdown(
        f'<div style="font-size:0.75rem;color:#6b7280;">🔑 thread: <code>{st.session_state.thread_id[:8]}...</code></div>',
        unsafe_allow_html=True,
    )
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.agents = build_agents(llm)
        st.session_state.total_calls = 0
        st.session_state.total_tools = 0
        st.rerun()

# ─── Chat display ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── Empty state ───────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;color:#4b5563;">
        <div style="font-size:3rem;">💬</div>
        <div style="font-size:1.1rem;font-weight:600;color:#9ca3af;margin-top:0.5rem;">Start a conversation</div>
        <div style="font-size:0.85rem;margin-top:0.3rem;">
            Try: <i>"What's the weather in Tel Aviv?"</i> · <i>"Calculate 15% of 20000"</i> · <i>"Who was Einstein?"</i>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Chat input ────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask me anything..."):

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.status("🧠 Running multi-agent pipeline...", expanded=True) as status:
            start_time = time.time()

            # Step 1 — context
            if st.session_state.retriever:
                st.markdown(
                    f'📄 **Hybrid retriever** — {st.session_state.num_chunks} chunks active',
                )
            else:
                st.markdown("📭 No document loaded")

            msg_count = len(st.session_state.messages)
            st.markdown(
                f'🧩 **MemorySaver** — thread `{st.session_state.thread_id[:8]}...` · {msg_count} messages'
            )

            # Step 2 — supervisor
            st.markdown("🎯 **Supervisor routing...**")
            decision = route(llm, user_input)
            st.session_state.total_calls += 1

            conf_pct = int(decision.confidence * 100)
            with st.expander("🎯 Routing decision"):
                agents_selected = " → ".join(
                    f"{AGENT_REGISTRY[a]['emoji']} `{a}`" for a in decision.agents
                )
                st.markdown(f"**Route:** {agents_selected}")
                st.markdown("**Confidence:**")
                st.markdown(confidence_bar(decision.confidence), unsafe_allow_html=True)
                st.markdown(f"**Reason:** _{decision.reason}_")
                if len(decision.agents) > 1:
                    st.info("🔄 Multi-hop — agents will run in sequence")

            if decision.confidence < 0.4:
                st.warning("⚠️ Low confidence — defaulting to Research Agent")
                decision.agents = ["research"]

            # Step 3 — run agents
            # Build message with history so every agent knows prior conversation
            prior_history = st.session_state.messages[:-1]  # exclude current user msg
            message_with_history = build_message_with_history(user_input, prior_history)

            with st.expander("📤 Full message sent to agents (with history)"):
                st.code(message_with_history, language="markdown")

            all_answers = []
            for agent_key in decision.agents:
                cfg = AGENT_REGISTRY[agent_key]
                _, color = agent_colors[agent_key]
                st.markdown(
                    f'<div style="display:inline-flex;align-items:center;gap:6px;'
                    f'background:{color}18;border:1px solid {color}44;border-radius:8px;'
                    f'padding:4px 12px;font-size:0.85rem;font-weight:600;color:{color};">'
                    f'{cfg["emoji"]} Running {cfg["name"]}...</div>',
                    unsafe_allow_html=True,
                )

                answer, all_messages = run_agent_and_log(
                    st.session_state.agents[agent_key],
                    message_with_history,
                    st.session_state.thread_id,
                    agent_key,
                )

                tool_count = sum(
                    1 for m in all_messages if isinstance(m, ToolMessage)
                )
                st.session_state.total_tools += tool_count
                all_answers.append((agent_key, answer))

            # Step 4 — combine
            if len(all_answers) == 1:
                final_answer = all_answers[0][1]
            else:
                parts = []
                for ak, ans in all_answers:
                    c = AGENT_REGISTRY[ak]
                    parts.append(f"**{c['emoji']} {c['name']}:**\n{ans}")
                final_answer = "\n\n---\n\n".join(parts)

            elapsed = round(time.time() - start_time, 2)
            agents_label = " + ".join(
                f"{AGENT_REGISTRY[a]['emoji']} {AGENT_REGISTRY[a]['name']}"
                for a in decision.agents
            )
            status.update(
                label=f"✅ {agents_label} · {elapsed}s · confidence {conf_pct}%",
                state="complete",
                expanded=False,
            )

        st.markdown(final_answer)

    st.session_state.messages.append({"role": "assistant", "content": final_answer})
