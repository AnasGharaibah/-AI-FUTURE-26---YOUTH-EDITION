"""Web app: the UI (UI/index.html) + the API it talks to.

    uv run uvicorn api:app            -> http://127.0.0.1:8000        (API docs at /docs)

Documents live in data/raw/ (the library); uploads are saved there and indexed with DB.pipeline.
Answers come from Agent.rag. Folders and stars are kept in data/library.json.
"""
import hashlib
import json
import re
import threading
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from Agent import llm, rag
from DB.arabic import is_arabic, normalize_search, tokens
from DB.parse import PAGE_RE, Parser

ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "UI"
RAW = ROOT / "data/raw"
PARSED = ROOT / "data/parsed"
LIB_PATH = ROOT / "data/library.json"
SUPPORTED = Parser.MARKER_EXT | Parser.OFFICE_EXT | Parser.PLAIN_EXT
LOCK = threading.Lock()                       # one library write / re-index at a time

app = FastAPI(title="Yoth AI")


# ================================================================ library
def doc_id(name: str) -> str:
    return hashlib.md5(name.encode()).hexdigest()[:10]


def _meta() -> dict:
    """Folders + per-file folder memberships and stars (the files themselves are data/raw/)."""
    if LIB_PATH.exists():
        return json.loads(LIB_PATH.read_text(encoding="utf-8"))
    return {"folders": ["General"], "docs": {}}


def _save_meta(meta: dict):
    LIB_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


@lru_cache(maxsize=256)
def _pages(name: str, mtime: float) -> list[str]:
    md = (PARSED / f"{name}.md").read_text(encoding="utf-8")
    parts = PAGE_RE.split(md)                 # [before, "0", page0, "1", page1, ...]
    if len(parts) == 1:
        return [md.strip()]
    by_num = {int(parts[i]): parts[i + 1].strip() for i in range(1, len(parts), 2)}
    pages = [by_num.get(i, "") for i in range(max(by_num) + 1)]
    if parts[0].strip():
        pages[0] = parts[0].strip() + "\n\n" + pages[0]
    return pages


def pages(name: str) -> list[str]:
    """Text of each page of a parsed document (chunk page N = pages[N - 1]; no page markers = one page)."""
    p = PARSED / f"{name}.md"
    return _pages(name, p.stat().st_mtime) if p.exists() else []


def raw_files() -> list[Path]:
    RAW.mkdir(parents=True, exist_ok=True)
    return sorted(f for f in RAW.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED)


def library() -> dict:
    meta, docs = _meta(), {}
    for f in raw_files():
        info = meta["docs"].get(f.name, {})
        folders = info.get("folders") or ["General"]
        docs[doc_id(f.name)] = {"id": doc_id(f.name), "name": f.name, "folder": folders[0], "folders": folders,
                                "pages": max(1, len(pages(f.name))), "ext": f.suffix.lower(),
                                "starred": bool(info.get("starred"))}
    return {"folders": meta["folders"], "docs": docs}


def doc_name(id_: str) -> str:
    for f in raw_files():
        if doc_id(f.name) == id_:
            return f.name
    raise HTTPException(404, "Document not found.")


def reindex():
    rag.db().ingest()


@app.get("/api/library")
def get_library():
    return library()


@app.post("/api/upload")
def upload(files: list[UploadFile] = File(...), folder: str = Form("General")):
    saved, out = [], []
    with LOCK:
        meta = _meta()
        for f in files:
            name = Path(f.filename or "").name
            if Path(name).suffix.lower() not in SUPPORTED:
                out.append({"ok": False, "name": name, "error": "Unsupported file type. Use " +
                            ", ".join(sorted(SUPPORTED)) + "."})
                continue
            (RAW / name).write_bytes(f.file.read())
            meta["docs"][name] = {**meta["docs"].get(name, {}), "folders": [folder]}
            if folder not in meta["folders"]:
                meta["folders"].append(folder)
            saved.append(name)
        _save_meta(meta)
        if saved:
            reindex()
    manifest_path = PARSED / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    for name in saved:
        status = manifest.get(name, {}).get("status", "failed")
        if status in ("ok", "partial", "fallback"):
            out.append({"ok": True, "doc": library()["docs"][doc_id(name)]})
        else:                                 # unreadable (scanned / broken): keep it out of the library
            with LOCK:
                (RAW / name).unlink(missing_ok=True)
                meta = _meta()
                meta["docs"].pop(name, None)
                _save_meta(meta)
            out.append({"ok": False, "name": name, "error": f"could not read the text ({status})."})
    return out


@app.delete("/api/docs/{id_}")
def delete_doc(id_: str):
    with LOCK:
        name = doc_name(id_)
        (RAW / name).unlink(missing_ok=True)
        meta = _meta()
        meta["docs"].pop(name, None)
        _save_meta(meta)
        reindex()
    return library()


@app.patch("/api/docs/{id_}")
def patch_doc(id_: str, body: dict):
    with LOCK:
        name, meta = doc_name(id_), _meta()
        info = meta["docs"].setdefault(name, {})
        if "starred" in body:
            info["starred"] = bool(body["starred"])
        if "folder" in body:
            target = str(body["folder"]).strip()
            if not target:
                raise HTTPException(400, "Folder name is required.")
            info["folders"] = [target]
            if target not in meta["folders"]:
                meta["folders"].append(target)
        _save_meta(meta)
    return library()


@app.post("/api/folders")
def new_folder(body: dict):
    name = str(body.get("name", "")).strip()[:40]
    if not name:
        raise HTTPException(400, "Folder name is required.")
    with LOCK:
        meta = _meta()
        if name not in meta["folders"]:
            meta["folders"].append(name)
            _save_meta(meta)
    return library()


@app.delete("/api/folders/{name}")
def delete_folder(name: str):
    if name == "General":
        raise HTTPException(400, "Cannot delete General.")
    with LOCK:
        meta = _meta()
        for info in meta["docs"].values():
            if name in info.get("folders", []):
                info["folders"] = [f for f in info["folders"] if f != name] or ["General"]
        meta["folders"] = [f for f in meta["folders"] if f != name]
        _save_meta(meta)
    return library()


# ---------------------------------------------------------------- viewer
@app.get("/api/doc/{id_}/page/{n}")
def doc_page(id_: str, n: int):
    ps = pages(doc_name(id_)) or [""]
    if not 1 <= n <= len(ps):
        raise HTTPException(404, "No such page.")
    return {"text": ps[n - 1], "total": len(ps)}


@app.get("/api/doc/{id_}/file")
def doc_file(id_: str):
    name = doc_name(id_)
    return FileResponse(RAW / name, content_disposition_type="inline", filename=name)


# ================================================================ chat
class ChatIn(BaseModel):
    question: str
    folders: list[str] = []
    doc_ids: list[str] = []
    history: list[dict] = []            # [{"role": "user"|"assistant", "content": str}, ...] kept by the browser
    language: str = "auto"


_CASUAL = re.compile(
    r"^(مرحبا|اهلا|اهلا وسهلا|السلام عليكم|وعليكم السلام|صباح الخير|مساء الخير|شكرا|شكرا لك|مع السلامه|كيف حالك"
    r"|hi|hello|hey|good (morning|afternoon|evening)|thanks|thank you|bye|goodbye|how are you|who are you"
    r"|what can you do)[!?.؟ ]*$")


def _pairs(history: list[dict]) -> list[tuple[str, str]]:
    out, q = [], None
    for m in history:
        if m.get("role") == "user":
            q = m.get("content", "")
        elif m.get("role") == "assistant" and q is not None:
            out.append((q, m.get("content", "")))
            q = None
    return out


def _scope(req: ChatIn) -> list[str] | None:
    """File names to search in; None = everything, [] = an empty folder."""
    docs = library()["docs"]
    if req.doc_ids:
        return [docs[i]["name"] for i in req.doc_ids if i in docs]
    if req.folders:
        return [d["name"] for d in docs.values() if set(d["folders"]) & set(req.folders)]
    return None


def _locate(page: str, chunk: str) -> tuple[int, int]:
    """Character span of a chunk inside its page text (for highlighting); (0, 0) when not found."""
    spans = []
    for line in chunk.splitlines():
        line = line.strip().strip("*").strip()
        for probe in (line, line[:40]):
            if len(probe) >= 8 and (i := page.find(probe)) >= 0:
                spans.append((i, i + len(probe)))
                break
    if not spans:
        return 0, 0
    start = min(s for s, _ in spans)
    end = max(e for s, e in spans if s - start < 2 * len(chunk) + 200)
    return start, end


def _source(s: dict) -> dict:
    name, page = s["file"], s.get("page") or 1
    ps = pages(name)
    start, end = _locate(ps[page - 1], s["text"]) if 1 <= page <= len(ps) else (0, 0)
    return {"n": s["n"], "doc_id": doc_id(name), "file": name, "page": page, "section": s.get("section") or "",
            "start": start, "end": end, "quote": s["text"][:700]}


def _verify(answer: str, sources: list[dict]) -> list[dict]:
    """Per sentence: do its cited sources share most of its words? (a support signal, not a guarantee)"""
    text_of = {s["n"]: set(tokens(s["text"])) for s in sources}
    out = []
    for sent in re.split(r"(?<=[.!?؟])\s+|\n+", answer):
        cited = [int(x) for x in re.findall(r"\[(\d+)\]", sent)]
        words = set(tokens(re.sub(r"\[\d+\]", " ", sent)))
        if not words:
            continue
        best = max((len(words & text_of.get(n, set())) / len(words) for n in cited), default=0.0)
        status = "supported" if cited and best >= 0.32 else "weak" if cited else "unsupported"
        out.append({"sentence": re.sub(r"\[\d+\]", "", sent).strip(), "status": status,
                    "citations": cited, "support": round(best, 2)})
    return out


def _line(obj) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


@app.post("/api/chat")
def chat(req: ChatIn):
    """Streams newline-delimited JSON events: sources / token / verification / not_found / done / error."""
    def events():
        try:
            q = req.question.strip()
            if _CASUAL.match(normalize_search(q)):
                ar = is_arabic(q)
                reply = llm.chat([
                    {"role": "system", "content": "You are Yoth AI, a friendly assistant for government-service "
                     "documents. Reply briefly and warmly to this greeting, in " + ("Arabic." if ar else "English.")},
                    {"role": "user", "content": q}], max_tokens=80)
                yield _line({"type": "token", "text": reply})
                yield _line({"type": "done", "mode": "chat"})
                return

            names = _scope(req)
            if names == []:
                yield _line({"type": "not_found", "text": rag.REFUSAL_AR if is_arabic(q) else rag.REFUSAL_EN})
                yield _line({"type": "done", "mode": "documents", "found": False})
                return
            r = rag.answer(q, sources=names, history=_pairs(req.history))
            yield _line({"type": "trace", "trace": [], "query": r.get("msa") or r["standalone"]})
            if not r["sources"]:
                yield _line({"type": "not_found", "text": r["answer"]})
                yield _line({"type": "done", "mode": "documents", "found": False})
                return
            sources = [_source(s) for s in r["sources"]]
            yield _line({"type": "sources", "sources": sources})
            yield _line({"type": "token", "text": r["answer"]})
            yield _line({"type": "verification", "items": _verify(r["answer"], r["sources"])})
            yield _line({"type": "done", "mode": "documents", "found": True})
        except Exception as e:
            yield _line({"type": "error", "text": f"{e} (is Ollama running?)"})

    return StreamingResponse(events(), media_type="application/x-ndjson")


# ================================================================ export to Word
class ExportItem(BaseModel):
    question: str = ""
    answer: str = ""
    sources: list[dict] = []


class ExportIn(BaseModel):
    title: str = ""
    items: list[ExportItem]


def _rtl(paragraph, rtl: bool):
    """Arabic paragraphs: right-to-left direction (w:bidi) and RTL runs.
    No explicit alignment: in a bidi paragraph "start" is already the right edge ("right" would flip it left)."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    if not rtl:
        return paragraph
    ppr = paragraph._p.get_or_add_pPr()
    if ppr.find(qn("w:bidi")) is None:                 # Word wants <w:bidi> in its schema position
        after = {qn("w:" + t) for t in ("adjustRightInd", "snapToGrid", "spacing", "ind", "contextualSpacing",
                 "mirrorIndents", "suppressOverlap", "jc", "textDirection", "textAlignment", "textboxTightWrap",
                 "outlineLvl", "divId", "cnfStyle", "rPr", "sectPr", "pPrChange")}
        nxt = next((c for c in ppr if c.tag in after), None)
        bidi = OxmlElement("w:bidi")
        nxt.addprevious(bidi) if nxt is not None else ppr.append(bidi)
    for run in paragraph.runs:
        rpr = run._r.get_or_add_rPr()
        rpr.get_or_add_rFonts().set(qn("w:cs"), "Arial")   # a font that has Arabic letters
        rpr.get_or_add_rtl()
    return paragraph


def _clean(text: str) -> str:
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text or "")
    return re.sub(r"\*\*|__|`", "", text).strip()


@app.post("/api/export/docx")
def export_docx(req: ExportIn):
    """Questions + answers (+ their sources) -> a .docx download. Arabic text is written right-to-left."""
    import io
    from urllib.parse import quote
    try:
        import docx
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        from docx.shared import Pt, RGBColor
    except ImportError:
        raise HTTPException(500, "python-docx is missing. Run: uv add python-docx")
    items = [i for i in req.items if i.answer.strip()]
    if not items:
        raise HTTPException(400, "There is nothing to export yet.")

    all_text = req.title + " ".join(i.question + i.answer for i in items)
    ar = is_arabic(all_text)
    title = _clean(req.title)[:80] or ("محادثة يوث" if ar else "Yoth AI chat")

    doc = docx.Document()
    style = doc.styles["Normal"]
    style.font.name, style.font.size = "Arial", Pt(12)
    _rtl(doc.add_heading(title, level=1), ar)

    for n, item in enumerate(items, 1):
        q_ar, a_ar = is_arabic(item.question or item.answer), is_arabic(item.answer)
        if item.question.strip() and not (len(items) == 1 and _clean(item.question)[:80] == title):
            label = ("السؤال" if q_ar else "Question") + (f" {n}" if len(items) > 1 else "")
            _rtl(doc.add_heading(f"{label}: {_clean(item.question)}", level=2), q_ar)
        for block in re.split(r"\n\s*\n|\n(?=\s*(?:\d+[.)]|[-•*])\s)", _clean(item.answer)):
            if block.strip():
                _rtl(doc.add_paragraph(block.strip()), a_ar)
        seen, refs = set(), []
        for s in item.sources:
            file, page = str(s.get("file", "")).strip(), s.get("page")
            if file and (file, page) not in seen:
                seen.add((file, page))
                n_ = s.get("n")
                where = (f"، صفحة {page}" if a_ar else f", page {page}") if page else ""
                refs.append((f"[{n_}] " if n_ else "") + file + where)
        if refs:
            p = doc.add_paragraph()
            run = p.add_run("المصادر:" if a_ar else "Sources:")
            run.bold = True
            run._r.get_or_add_rPr().get_or_add_bCs()                   # bold for Arabic letters too
            _rtl(p, a_ar)
            for r in refs:
                p = _rtl(doc.add_paragraph(r, style="List Bullet"), a_ar)
                for run in p.runs:
                    run.font.size, run.font.color.rgb = Pt(10), RGBColor(0x55, 0x55, 0x55)
                    sz = run._r.rPr.find(qn("w:sz"))
                    nxt = sz.getnext() if sz is not None else None
                    if sz is not None and (nxt is None or nxt.tag != qn("w:szCs")):
                        cs = OxmlElement("w:szCs")                     # same size for Arabic letters
                        cs.set(qn("w:val"), "20")
                        sz.addnext(cs)

    buf = io.BytesIO()
    doc.save(buf)
    name = re.sub(r'[\\/:*?"<>|\r\n]+', " ", title).strip()[:60] or "chat"
    return Response(buf.getvalue(),
                    media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    headers={"Content-Disposition": f"attachment; filename=\"export.docx\"; "
                                                    f"filename*=UTF-8''{quote(name + '.docx')}"})


# ================================================================ voice
class TTSIn(BaseModel):
    text: str
    language: str | None = None           # "ar" | "en" | None = detect


@app.post("/api/stt")
def stt(audio: UploadFile = File(...), language: str = Form("auto")):
    """Audio upload (webm / wav / mp3 ...) -> {"text", "language", "seconds"}."""
    from Agent.voice import transcribe
    return transcribe(audio.file.read(), None if language in ("", "auto") else language[:2])


@app.post("/api/tts")
def tts(body: TTSIn):
    """Text -> wav. Header X-TTS-Engine says which engine produced it."""
    from Agent.voice import synthesize
    wav, engine, _ = synthesize(body.text, body.language)
    if not wav:
        return Response(status_code=204)
    return Response(wav, media_type="audio/wav", headers={"X-TTS-Engine": engine})


@app.get("/api/health")
def health():
    return {"status": "ok", "model": llm.MODEL}


app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
