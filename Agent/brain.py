"""The agent: a LangGraph loop  llm -> (tool calls?) -> tools -> llm ... -> answer.

    from Agent.brain import ask
    print(ask("What is the hotel limit in Dubai?", thread_id="anas"))
"""
import os

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from . import llm, rag
from .context import SYSTEM_PROMPT
from .memory import checkpointer
from .tools import TOOLS

MODEL = llm.MODEL                                  # YOTH_MODEL env; any Ollama chat model
MODE = os.getenv("YOTH_MODE", "rag")               # "rag" = retrieve-then-answer (Arabic gov docs); "agent" = tool-calling graph


def build_graph(model: str = MODEL):
    llm = ChatOllama(model=model, temperature=0.1, num_ctx=8192, reasoning=False).bind_tools(TOOLS)

    def call_llm(state: MessagesState):
        return {"messages": [llm.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])]}

    g = StateGraph(MessagesState)
    g.add_node("llm", call_llm)
    g.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
    g.add_edge(START, "llm")
    g.add_conditional_edges("llm", tools_condition)     # tool calls -> "tools", otherwise END
    g.add_edge("tools", "llm")
    return g.compile(checkpointer=checkpointer)


_graph = None


def graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def ask(question: str, thread_id: str = "default") -> str:
    if MODE == "rag":
        return rag.answer(question, thread_id)["answer"]
    out = graph().invoke({"messages": [("user", question)]},
                         {"configurable": {"thread_id": thread_id}})
    return out["messages"][-1].content


def stream(question: str, thread_id: str = "default"):
    """Yield ('tool', name) when a tool is called and ('text', chunk) for the answer.
    In rag mode the answer is checked (citations / numbers) before it is released, then sent in word groups,
    followed by ('sources', [...])."""
    if MODE == "rag":
        yield "tool", "search_documents"
        r = rag.answer(question, thread_id)
        words = r["answer"].split(" ")
        for i in range(0, len(words), 4):
            yield "text", " ".join(words[i:i + 4]) + (" " if i + 4 < len(words) else "")
        yield "sources", r["sources"]
        return
    cfg = {"configurable": {"thread_id": thread_id}}
    for msg, meta in graph().stream({"messages": [("user", question)]}, cfg, stream_mode="messages"):
        if meta.get("langgraph_node") == "llm":
            for call in getattr(msg, "tool_call_chunks", []) or []:
                if call.get("name"):
                    yield "tool", call["name"]
            if msg.content:
                yield "text", msg.content
