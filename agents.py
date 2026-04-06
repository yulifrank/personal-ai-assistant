from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from tools import (
    calculator, compare_numbers, get_crypto_price, get_exchange_rate,
    word_counter, summarize_request, bullet_list_formatter, keyword_extractor,
    get_current_date, get_weather, search_wikipedia,
)

# ── Structured routing decision ────────────────────────────────────────────────
class RoutingDecision(BaseModel):
    """Structured output from the supervisor LLM."""
    agents: List[Literal["research", "finance", "document", "utility"]] = Field(
        description="Ordered list of agents to run. Use multiple for complex questions."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="How confident you are in this routing decision (0.0-1.0)."
    )
    reason: str = Field(
        description="One sentence explaining why you chose these agents."
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
            "IMPORTANT: always write the search_wikipedia query in English, even if the user wrote in Hebrew or another language. "
            "For example: if asked 'מי ראש הממשלה בישראל' → search_wikipedia('Prime Minister of Israel'). "
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
            "ALWAYS use the calculator tool for ANY arithmetic — even simple ones like 2+3 or 10*5. Never calculate in your head. "
            "To compare two numbers → use compare_numbers. "
            "For cryptocurrency prices → use get_crypto_price. "
            "For currency exchange rates → use get_exchange_rate. "
            "If the question is not math/finance related, say so clearly. "
            "Respond in the same language the user wrote in."
        ),
    },
    "document": {
        "emoji": "📝",
        "name": "Document & Text Agent",
        "description": "Document search (RAG), word counting, summarizing, keyword extraction, bullet formatting",
        "tools": [word_counter, summarize_request, bullet_list_formatter, keyword_extractor],
        "prompt": (
            "You are a text and document specialist. "
            "If the user asks about content from their uploaded document → use search_document to find relevant information. "
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

SUPERVISOR_PROMPT = """You are a routing supervisor for a multi-agent AI assistant.
Your job is to decide which specialist agent(s) should handle the user's question.

Available agents:
- research   → Wikipedia lookups, weather, general world knowledge about people/places/events
- finance    → math calculations, number comparison, crypto prices, currency exchange
- document   → document search, word counting, summarizing, keyword extraction, bullet formatting
- utility    → current date and time

Rules:
- For simple questions, return ONE agent.
- For complex questions that need multiple types of knowledge, return MULTIPLE agents in logical order.
- Set confidence between 0.0 and 1.0 based on how clear the routing is.
- Write one short sentence explaining your decision in the reason field.

Examples:
- "Who was Einstein?" → agents: ["research"], confidence: 0.99
- "What is Bitcoin price and who created it?" → agents: ["finance", "research"], confidence: 0.95
- "Summarize section 3 and calculate the total" → agents: ["document", "finance"], confidence: 0.90"""


def build_agents(llm: ChatGoogleGenerativeAI, search_document_tool=None) -> dict:
    """
    Builds all specialist agents with MemorySaver checkpointer.
    Optionally injects search_document tool into the Document Agent.
    """
    checkpointer = MemorySaver()
    agents = {}
    for key, cfg in AGENT_REGISTRY.items():
        tools = list(cfg["tools"])
        if key == "document" and search_document_tool is not None:
            tools = [search_document_tool] + tools
        agents[key] = create_react_agent(
            llm,
            tools,
            prompt=cfg["prompt"],
            checkpointer=checkpointer,
        )
    return agents


def route(llm: ChatGoogleGenerativeAI, user_input: str) -> RoutingDecision:
    """
    Uses structured output to get a typed routing decision from the supervisor.
    Returns a RoutingDecision with agents list, confidence score, and reason.
    Falls back to research agent if structured output fails.
    """
    structured_llm = llm.with_structured_output(RoutingDecision)
    try:
        decision = structured_llm.invoke([
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(content=user_input),
        ])
        # Validate agents list is not empty
        if not decision.agents:
            decision.agents = ["research"]
        return decision
    except Exception:
        return RoutingDecision(
            agents=["research"],
            confidence=0.0,
            reason="Fallback: structured output failed, defaulting to research agent.",
        )
