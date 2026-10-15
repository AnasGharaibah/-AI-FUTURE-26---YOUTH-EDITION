"""Entry point for the terminal chat, voice mode, eval and the web API.

    YOTH_MODE=agent (default): Agent/agent.py - router + tool calling + guardrails
    YOTH_MODE=rag            : Agent/rag.py   - retrieve-then-answer (also the agent's automatic fallback)

    from Agent.brain import ask
    print(ask("كم رسوم تجديد جواز السفر؟", thread_id="anas"))
"""
import os

from DB.arabic import normalize_display

from . import agent, llm, rag

MODEL = llm.MODEL                                  # model at import time (banner); calls read llm.MODEL
MODE = os.getenv("YOTH_MODE", "agent")             # "agent" | "rag"


def ask(question: str, thread_id: str = "default") -> str:
    if MODE == "rag":
        return rag.answer(question, thread_id)["answer"]
    return agent.answer(question, thread_id)["answer"]


def ask_traced(question: str, thread_id: str = "default", model: str | None = None) -> dict:
    """Agent mode with a trace: {"answer", "tools": [tool names in call order], "model_calls", "retrieved",
    "route", "flags", "sources"}."""
    return agent.answer(question, thread_id, model=model)


def stream(question: str, thread_id: str = "default"):
    """Yield ('tool', name) when a tool is called and ('text', chunk) for the answer.
    rag mode: tokens as the model writes them; if the guards changed the answer, ('replace', checked_answer).
    agent mode: the checked answer in one piece (tool calls first). Then ('sources', [...])."""
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
    for kind, val in agent.run(question, thread_id):
        if kind == "tool":
            yield "tool", val
        elif kind == "done":
            yield "text", val["answer"]
            yield "sources", val["sources"]
