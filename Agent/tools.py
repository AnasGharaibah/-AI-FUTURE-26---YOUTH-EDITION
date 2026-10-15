"""The agent's three tools (Ollama native tool calling). Descriptions in Arabic + English, government wording.

    search_documents(query, file=None) -> numbered sources from the hybrid search (optionally one file)
    get_document_summary(file)        -> the stored summary of one document (DB/summaries.py)
    list_services()                   -> the Service Map
"""
from DB import summaries as S
from DB.arabic import normalize_search
from DB.pipeline import Pipeline

from . import llm


def _db() -> Pipeline:
    return Pipeline.shared()                  # same instance as Agent.rag: one Chroma client, one BM25 index


def service_files() -> list[str]:
    """File names the agent may filter on: the summarized documents, else whatever is indexed."""
    return sorted(S.load()) or sorted(_db().store.sources())


def resolve_file(name: str | None) -> str | None:
    """The model's file argument -> a real file name (exact, case-insensitive, or by service name); None = no filter."""
    if not name or not str(name).strip():
        return None
    files, want = service_files(), normalize_search(str(name).strip().strip("`'\""))
    for f in files:
        if normalize_search(f) == want or normalize_search(f.rsplit(".", 1)[0]) == want:
            return f
    for f, s in S.load().items():
        if want and (want in normalize_search(f) or want in normalize_search(s.get("service", ""))
                     or normalize_search(s.get("service", "")) in want):
            return f
    return None


def search_documents(query: str, file: str | None = None, start: int = 1, model: str | None = None,
                     scope: list[str] | None = None) -> tuple[str, list[dict]]:
    """-> (numbered sources text for the model, hits). Numbers continue from `start` so that several searches in one
    turn never reuse [1]. scope: the files the user limited the chat to (web UI folder / document)."""
    from . import rag
    pipe = _db()
    pipe.cfg["rewriter"] = lambda q: llm.rewrite_to_msa(q, model)
    target = resolve_file(file)
    files = [target] if target else None
    if scope is not None:
        files = [f for f in (files or scope) if f in scope]
        if not files:
            return "لا توجد نتائج: الملف المطلوب خارج نطاق البحث المحدد.", []
    hits = pipe.search(query, sources=files)
    if not hits:
        return "لا توجد نتائج مطابقة في الوثائق.", []
    return rag.sources_block(hits, start), hits


def get_document_summary(file: str) -> str:
    target = resolve_file(file)
    s = S.load().get(target) if target else None
    if not s:
        return f"لا يوجد ملخص للملف {file!r}. الملفات المتاحة: " + "، ".join(service_files())
    facts = "\n".join(f"- {f['fact']}" + (f" (صفحة {f['source_page']})" if f.get("source_page") else "")
                      for f in s["key_facts"])
    return (f"الخدمة: {s['service']}\nالملف: {s['file']}\nالملخص: {s['summary']}\n"
            f"المواضيع: {'، '.join(s['topics'])}\nالأقسام: {'، '.join(s['sections'])}\n"
            f"الحالات: {'؛ '.join(s['cases']) or '-'}\nحقائق أساسية:\n{facts or '-'}")


def list_services() -> str:
    return S.MAP_PATH.read_text(encoding="utf-8") if S.MAP_PATH.exists() else "خريطة الخدمات غير متوفرة بعد."


def _spec(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required}}}


def tool_specs() -> list[dict]:
    files = service_files()
    file_prop = {"type": "string", "description": "اسم الملف من خريطة الخدمات / file name from the Service Map",
                 **({"enum": files} if files else {})}
    return [
        _spec("search_documents",
              "ابحث في الوثائق الرسمية للخدمات الحكومية (الرسوم، الوثائق المطلوبة، الخطوات، الشروط، المدد، المواد). "
              "يعيد مصادر مرقّمة. / Search the official government-service documents (fees, required documents, "
              "steps, conditions, durations, articles). Returns numbered sources.",
              {"query": {"type": "string", "description": "سؤال البحث / the search query"}, "file": file_prop},
              ["query"]),
        _spec("get_document_summary",
              "ملخص وثيقة رسمية كاملة: الخدمة، المواضيع، الحالات، الحقائق الأساسية. / Summary of one official "
              "document: service, topics, cases, key facts.",
              {"file": file_prop}, ["file"]),
        _spec("list_services",
              "قائمة كل الخدمات والوثائق المتوفرة (خريطة الخدمات). / List every service and document covered "
              "(the Service Map).", {}, []),
    ]


TOOL_NAMES = ("search_documents", "get_document_summary", "list_services")
