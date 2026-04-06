from langchain_core.messages import HumanMessage, AIMessage


def create_memory():
    return {"messages": []}


def format_history(memory) -> str:
    messages = memory.get("messages", [])
    if not messages:
        return "No conversation history yet."

    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            history.append(f"Assistant: {msg.content}")

    return "\n".join(history)


def add_to_memory(memory, user_msg: str, ai_msg: str):
    messages = memory.setdefault("messages", [])
    messages.append(HumanMessage(content=user_msg))
    messages.append(AIMessage(content=ai_msg))

    # Keep only the last 10 pairs (20 messages)
    if len(messages) > 20:
        memory["messages"] = messages[-20:]
