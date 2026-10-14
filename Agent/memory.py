"""Conversation memory: LangGraph checkpointer keyed by thread_id (one thread = one chat)."""
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()          # in RAM; swap for SqliteSaver to survive restarts
