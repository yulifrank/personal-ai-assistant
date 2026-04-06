from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from tools import (
    calculator, compare_numbers, get_crypto_price, get_exchange_rate,
    word_counter, summarize_request, bullet_list_formatter, keyword_extractor,
    get_current_date, get_weather, search_wikipedia,
)

# ── Agent definitions ──────────────────────────────────────────────────────────
AGENT_REGISTRY = {
    "research": {
        "emoji": "🔍",
        "name": "Research Agent",
        "description": "Wikipedia searches, weather, general world knowledge",
        "tools": [search_wikipedia, get_weather],
        "prompt": (
            "You are a research specialist. "
            "For ANY factual question about a person, place, concept, or event → use search_wikipedia. "
            "For weather questions → use get_weather. "
            "Always use your tools — never answer from memory alone. "
            "Respond in the same language the user wrote in."
        ),
    },
    "finance": {
        "emoji": "💰",
        "name": "Finance & Math Agent",
        "description": "Math calculations, number comparison, crypto prices, currency exchange rates",
        "tools": [calculator, compare_numbers, get_crypto_price, get_exchange_rate],
        "prompt": (
            "You are a finance and math specialist. "
            "For any math or calculation → use calculator. "
            "To compare two numbers → use compare_numbers. "
            "For cryptocurrency prices → use get_crypto_price. "
            "For currency exchange rates → use get_exchange_rate. "
            "Respond in the same language the user wrote in."
        ),
    },
    "document": {
        "emoji": "📝",
        "name": "Document & Text Agent",
        "description": "Word counting, text summarizing, keyword extraction, bullet point formatting",
        "tools": [word_counter, summarize_request, bullet_list_formatter, keyword_extractor],
        "prompt": (
            "You are a text and document specialist. "
            "For word/character counts → use word_counter. "
            "To summarize a topic → use summarize_request. "
            "To format text as bullet points → use bullet_list_formatter. "
            "To extract keywords → use keyword_extractor. "
            "Respond in the same language the user wrote in."
        ),
    },
    "utility": {
        "emoji": "🛠️",
        "name": "Utility Agent",
        "description": "Current date and time",
        "tools": [get_current_date],
        "prompt": (
            "You are a utility assistant. "
            "For any date or time question → use get_current_date. "
            "Respond in the same language the user wrote in."
        ),
    },
}

SUPERVISOR_PROMPT = """You are a routing supervisor. Your only job is to decide which specialist agent should handle the user's question.

Available agents:
- research   → Wikipedia lookups, weather, general world knowledge about people/places/events
- finance    → math calculations, number comparison, crypto prices, currency exchange
- document   → word counting, text summarizing, keyword extraction, bullet point formatting
- utility    → current date and time

Reply with ONLY one word — the agent name: research, finance, document, or utility
Do NOT explain. Do NOT add punctuation. Just the agent name."""


def build_agents(llm: ChatGoogleGenerativeAI) -> dict:
    """Builds and returns all specialist agents."""
    agents = {}
    for key, cfg in AGENT_REGISTRY.items():
        agents[key] = create_react_agent(llm, cfg["tools"], prompt=cfg["prompt"])
    return agents


def route(llm: ChatGoogleGenerativeAI, user_input: str) -> tuple[str, str]:
    """
    Asks the supervisor LLM which agent to use.
    Returns (agent_key, reasoning_text).
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    response = llm.invoke([
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=user_input),
    ])
    raw = response.content
    if isinstance(raw, list):
        raw = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in raw)
    agent_key = raw.strip().lower().split()[0] if raw.strip() else "research"
    if agent_key not in AGENT_REGISTRY:
        agent_key = "research"
    return agent_key, raw.strip()
