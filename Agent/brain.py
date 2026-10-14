"""The agent: a LangGraph loop  llm -> (tool calls?) -> tools -> llm ... -> answer.

    from Agent.brain import ask
    print(ask("كم رسوم تجديد جواز السفر؟", thread_id="anas"))
"""
import os

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from DB.arabic import normalize_display

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
    """Yield ('tool', name) when a tool is called and ('text', chunk) for the answer as the model writes it.
    In rag mode the finished answer is then checked (citations / numbers / refusal); if the check changed it,
    ('replace', checked_answer) follows. Then ('sources', [...])."""
    if MODE == "rag":
        yield "tool", "search_documents"
        streamed = []
        for kind, val in rag.answer_stream(question, thread_id):
            if kind == "token":
                streamed.append(val)
                yield "text", val
            elif kind == "done":
                if normalize_display("".join(streamed)) != val["answer"]:
                    yield "replace", val["answer"]
                yield "sources", val["sources"]
        return
    cfg = {"configurable": {"thread_id": thread_id}}
    for msg, meta in graph().stream({"messages": [("user", question)]}, cfg, stream_mode="messages"):
        if meta.get("langgraph_node") == "llm":
            for call in getattr(msg, "tool_call_chunks", []) or []:
                if call.get("name"):
                    yield "tool", call["name"]
            if msg.content:
                yield "text", msg.content
