import certifi
import os
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = certifi.where()

import streamlit as st
import tempfile
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ToolMessage, AIMessage
from rag import get_retriever
from agents import AGENT_REGISTRY, build_agents, route
from memory import create_memory, format_history, add_to_memory

load_dotenv()


def extract_text(content) -> str:
    """Handles both plain string and Gemini list-of-parts content format."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


# ─── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="Personal AI Assistant", page_icon="🤖", layout="wide")

st.title("🤖 Personal AI Assistant")
st.caption("Multi-agent system — upload a document and ask me anything!")

# ─── Session state init ────────────────────────────────────────
if "memory" not in st.session_state:
    st.session_state.memory = create_memory()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ─── LLM (shared across all agents) ───────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,
)
agents = build_agents(llm)

# ─── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

    if uploaded_file:
        with st.spinner("Reading and indexing document..."):
            suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            st.session_state.retriever = get_retriever(tmp_path)
            os.unlink(tmp_path)
        st.success(f"✅ Loaded: {uploaded_file.name}")

    st.divider()

    st.header("🤖 Available Agents")
    for key, cfg in AGENT_REGISTRY.items():
        with st.expander(f"{cfg['emoji']} {cfg['name']}"):
            st.caption(cfg["description"])
            st.markdown("**Tools:**")
            for t in cfg["tools"]:
                st.markdown(f"- `{t.name}`")

    st.divider()

    if st.button("🗑 Clear conversation"):
        st.session_state.messages = []
        st.session_state.memory = create_memory()
        st.rerun()

# ─── Chat display ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── Chat input ────────────────────────────────────────────────
if user_input := st.chat_input("Ask me anything..."):

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.status("🧠 Multi-agent pipeline running...", expanded=True) as status:
            start_time = time.time()

            # ── Step 1: RAG ──────────────────────────────────────
            full_input = user_input
            if st.session_state.retriever:
                st.write("📂 **Searching document...**")
                docs = st.session_state.retriever.invoke(user_input)
                if docs:
                    st.write(f"📄 Found **{len(docs)}** relevant chunk(s)")
                    context = "\n\n".join([d.page_content for d in docs])
                    full_input = (
                        f"Use the following document context to help answer:\n\n"
                        f"{context}\n\nUser question: {user_input}"
                    )
                else:
                    st.write("📭 No relevant document content found")
            else:
                st.write("📭 No document uploaded — skipping RAG")

            # ── Step 2: Memory ───────────────────────────────────
            history = format_history(st.session_state.memory)
            if history != "No conversation history yet.":
                msg_count = len(st.session_state.memory.get("messages", []))
                st.write(f"🧩 **Loaded history** — {msg_count} messages")
                full_input = f"Conversation so far:\n{history}\n\n{full_input}"
            else:
                st.write("🆕 No previous conversation history")

            # ── Step 3: Full prompt log ───────────────────────────
            with st.expander("📤 Full message sent to agent"):
                st.code(full_input, language="markdown")

            # ── Step 4: Supervisor routing ───────────────────────
            st.write("🎯 **Supervisor deciding which agent to use...**")
            chosen_key, supervisor_raw = route(llm, user_input)
            cfg = AGENT_REGISTRY[chosen_key]

            with st.expander("🎯 Supervisor decision"):
                st.markdown(f"**Supervisor replied:** `{supervisor_raw}`")
                st.markdown(f"**→ Routing to:** {cfg['emoji']} **{cfg['name']}**")
                st.markdown(f"**Agent tools available:** {', '.join(f'`{t.name}`' for t in cfg['tools'])}")
                st.markdown("**Agent system prompt:**")
                st.code(cfg["prompt"], language="markdown")

            st.write(f"{cfg['emoji']} **Running {cfg['name']}...**")

            # ── Step 5: Run chosen agent ─────────────────────────
            response = agents[chosen_key].invoke({"messages": [("human", full_input)]})
            all_messages = response.get("messages", [])

            # ── Step 6: Full raw response ─────────────────────────
            with st.expander(f"📨 Full raw response — {len(all_messages)} message(s)"):
                for i, msg in enumerate(all_messages):
                    msg_type = type(msg).__name__
                    st.markdown(f"**[{i}] {msg_type}**")
                    if hasattr(msg, "content") and msg.content:
                        st.code(extract_text(msg.content), language="markdown")
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        st.markdown("🔧 **Tool calls requested:**")
                        st.json(msg.tool_calls)
                    if hasattr(msg, "tool_call_id"):
                        st.caption(f"tool_call_id: `{msg.tool_call_id}`")
                    st.divider()

            # ── Step 7: Tool calls summary ───────────────────────
            tool_calls_made = []
            for msg in all_messages:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls_made.append(tc)
                if isinstance(msg, ToolMessage):
                    matching = next(
                        (tc for tc in tool_calls_made if tc["id"] == msg.tool_call_id),
                        None,
                    )
                    tool_name = matching["name"] if matching else "unknown"
                    tool_input = matching["args"] if matching else {}
                    st.write(f"🔧 **Tool called:** `{tool_name}`")
                    with st.expander(f"↳ `{tool_name}` — input & output"):
                        st.markdown("**Input sent to tool:**")
                        st.json(tool_input)
                        st.markdown("**Output returned from tool:**")
                        st.code(extract_text(msg.content))

            if not tool_calls_made:
                st.write("💬 No tools were called — answered from model knowledge")

            # ── Step 8: Final answer ─────────────────────────────
            answer = extract_text(all_messages[-1].content)
            elapsed = round(time.time() - start_time, 2)
            status.update(
                label=f"✅ {cfg['emoji']} {cfg['name']} — Done in {elapsed}s",
                state="complete",
                expanded=False,
            )

        st.markdown(answer)

    add_to_memory(st.session_state.memory, user_input, answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
