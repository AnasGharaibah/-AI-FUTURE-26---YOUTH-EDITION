"""Tools the agent can call. Each @tool docstring is what the LLM reads to decide when to use it."""
import ast
import datetime as dt
import operator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

from langchain_core.tools import tool

from DB.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent


def _db() -> Pipeline:
    return Pipeline.shared()                  # same instance as Agent.rag: one Chroma client, one BM25 index


@tool
def search_documents(query: str) -> str:
    """Search the official government documents (service guides, regulations, FAQs; Arabic and English).
    Use for any question about a service: fees, required documents, steps, durations, eligibility or articles.
    Returns numbered sources with file names."""
    text, hits = _db().context(query)
    return text if hits else "No relevant documents found."


@tool
def query_projects(status: str = "", manager: str = "", client: str = "") -> str:
    """Look up projects in the projects register spreadsheet (project, client, manager, budget, status).
    All filters are optional, case-insensitive 'contains' matches. status is e.g. Active, Completed, On hold.
    Returns a table with budgets in JOD."""
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


MAX_EXPONENT = 100
MAX_NUMBER = 1e12
CALC_TIMEOUT = 2.0                      # seconds
_calc_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="calculate")


def _check(x):
    if abs(x) > MAX_NUMBER:
        raise ValueError(f"number too large (limit {MAX_NUMBER:.0e})")
    return x


def _eval(node):
    """Arithmetic only. Every operand and intermediate result must stay <= MAX_NUMBER and exponents <= MAX_EXPONENT,
    so input like 9**9**9 is rejected instead of freezing the process."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return _check(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"exponent too large (limit {MAX_EXPONENT})")
        return _check(_OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _check(_OPS[type(node.op)](_eval(node.operand)))
    raise ValueError("unsupported expression")


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '3 * 120 + 45.5'. Supports + - * / ** %."""
    try:
        tree = ast.parse(expression, mode="eval").body
        return str(_calc_pool.submit(_eval, tree).result(timeout=CALC_TIMEOUT))
    except FutureTimeout:
        return f"Could not calculate: took longer than {CALC_TIMEOUT:g} s"
    except Exception as e:
        return f"Could not calculate: {e}"


@tool
def get_current_date() -> str:
    """Today's date and weekday."""
    return dt.date.today().strftime("%A, %Y-%m-%d")


TOOLS = [search_documents, query_projects, calculate, get_current_date]
