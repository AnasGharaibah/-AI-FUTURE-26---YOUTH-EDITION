"""Conversation memory: LangGraph checkpointer keyed by thread_id (one thread = one chat)."""
from langgraph.checkpoint.memory import MemorySaver

# TODO(#15): MemorySaver keeps every thread forever (unbounded RAM in a long-running server); swap for SqliteSaver
#            or prune old threads.
checkpointer = MemorySaver()          # in RAM; swap for SqliteSaver to survive restarts
