"""Tools the agent can call. Each @tool docstring is what the LLM reads to decide when to use it."""
import ast
import datetime as dt
import operator
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool

from DB.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def _db() -> Pipeline:
    return Pipeline()


@tool
def search_documents(query: str) -> str:
    """Search the company documents (policies, employee handbook, IT FAQ, board update).
    Use for any question about rules, limits, benefits, procedures or IT help.
    Returns numbered sources with file names."""
    text, hits = _db().context(query)
    return text if hits else "No relevant documents found."


@tool
def query_projects(status: str = "", manager: str = "", client: str = "") -> str:
    """Look up projects in the projects spreadsheet. All filters are optional, case-insensitive
    'contains' matches. status is e.g. Active, Completed, On hold. Returns a table with budgets in JOD."""
    import pandas as pd
    df = pd.read_excel(ROOT / "data/raw/projects.xlsx")
    for col, val in (("Status", status), ("Manager", manager), ("Client", client)):
        if val:
            df = df[df[col].astype(str).str.contains(val, case=False)]
    if df.empty:
        return "No matching projects."
    return df.to_string(index=False) + f"\n\nTotal budget: {df['Budget (JOD)'].sum()} JOD"


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '3 * 120 + 45.5'. Supports + - * / ** %."""
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Could not calculate: {e}"


@tool
def get_current_date() -> str:
    """Today's date and weekday."""
    return dt.date.today().strftime("%A, %Y-%m-%d")


TOOLS = [search_documents, query_projects, calculate, get_current_date]
