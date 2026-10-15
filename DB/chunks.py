"""Step 2 - Markdown -> chunks.


Two algorithms:
  "structure" (default, best for unknown documents)
      follows headings / tables / lists / slides / pages, picks a size per
      document type, repeats table headers, adds a file > section context line,
      keeps a parent section for every chunk
  "fixed"     plain sliding window (baseline to compare against)


Usage:
    from DB.chunks import Chunker
    chunker = Chunker("structure")
    chunks, parents = chunker.chunk_text(markdown, source="report.pdf")
"""

import hashlib
import re
import sys
from pathlib import Path

if __package__:
    from .arabic import normalize_search
else:                                   # run directly: python DB/chunks.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from DB.arabic import normalize_search

PAGE_RE = re.compile(r"^\{(\d+)\}-{48}\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")
LIST_RE = re.compile(r"^\s*([-*+•]|\d+\s*[.)\-–—]|[أ-ي]\s*[-–)])\s+")
SENT_RE = re.compile(r"(?<=[.!?؟؛،۔。])\s+|(?<=[.!?؟])(?=[A-Z؀-ۿ])|\n+")


_AR_WORD = re.compile(r"[\u0600-\u06ff\u0750-\u077f]")


def ntok(text: str) -> int:
    """Rough token count: Arabic words tokenize heavier (x1.8) than Latin ones (x1.35)."""
    words = text.split()
    ar = sum(1 for w in words if _AR_WORD.search(w))
    return max(1, int(ar * 1.8 + (len(words) - ar) * 1.35))


# ---- Arabic government-document structure -------------------------------------------------
# level: lower number = bigger heading.  (باب 1, فصل 2, مادة 3, بند 4, أولاً... 5, "الرسوم:" labels 6)
ORDINALS = ("اول", "ثاني", "ثالث", "رابع", "خامس", "سادس", "سابع", "ثامن", "تاسع", "عاشر")
LABELS = ("الوثائق المطلوبه", "المستندات المطلوبه", "الاوراق المطلوبه", "الوثائق", "المستندات", "الرسوم", "رسوم",
          "الشروط", "شروط", "الخطوات", "الاجراءات", "خطوات", "مده الانجاز", "مدة الانجاز", "مكان التقديم",
          "قنوات التقديم", "طريقه التقديم", "الفئه المستفيده", "الفئات المستفيده", "ملاحظات", "الغرامات",
          "العقوبات", "التعريفات", "الجهه المسؤوله", "الجهة المسوولة", "اوقات الدوام", "معلومات الاتصال")
_ARTICLE = re.compile(r"^(?:\*\*)?\s*(المادة|مادة)\s*[\(\[]?\s*(\d+|[ء-ي ]{2,20}?)\s*[\)\]]?\s*(?:\*\*)?\s*[-–—:.،]?\s*(.*)$")
_CHAPTER = re.compile(r"^(?:\*\*)?\s*(الباب|الفصل|البند|القسم)\s+([\w؀-ۿ]+)(?:\*\*)?\s*[-–—:.،]?\s*(.*)$")
_ORDINAL = re.compile(r"^(?:\*\*)?\s*([ء-ي\u064b-\u0652]{3,14})\s*[:：\-–—]\s*(?:\*\*)?\s*(.*)$")
_LABEL = re.compile(r"^(?:\*\*)?\s*([^:：\n|]{2,40}?)\s*(?:\*\*)?\s*[:：]\s*(?:\*\*)?\s*(.*)$")
_BOLD_ONLY = re.compile(r"^\*\*([^*]{2,60})\*\*$")


def arabic_heading(line: str):
    """Arabic structural line -> (level, heading_text, rest_of_line) or None."""
    t = line.strip()
    if not t or t.startswith("|") or len(t.split()) > 14:
        return None
    m = _ARTICLE.match(t)
    if m:
        num, rest = m.group(2).strip(), m.group(3).strip()
        title = rest if rest and len(rest.split()) <= 6 and not re.search(r"[.؟!]", rest) else ""
        return 3, f"{m.group(1)} {num}" + (f" - {title}" if title else ""), "" if title else rest
    m = _CHAPTER.match(t)
    if m:
        lvl = {"الباب": 1, "القسم": 1, "الفصل": 2}.get(m.group(1), 4)
        title = m.group(3).strip()
        short = len(title.split()) <= 8 and not re.search(r"[.؟!]", title)
        return lvl, f"{m.group(1)} {m.group(2)}" + (f" - {title}" if title and short else ""), "" if short else title
    m = _ORDINAL.match(t)
    if m and normalize_search(m.group(1)).rstrip("ا").startswith(ORDINALS):
        rest = m.group(2).strip()
        short = len(rest.split()) <= 6 and not re.search(r"[.؟!]", rest)
        return 5, m.group(1) + (f": {rest}" if rest and short else ""), "" if short else rest
    m = _LABEL.match(t)
    if m:
        label, rest = m.group(1).strip(" *"), m.group(2).strip()
        known = normalize_search(label).startswith(tuple(normalize_search(x) for x in LABELS))
        if (known and len(label.split()) <= 4) or (not rest and len(label.split()) <= 6):
            return 6, label, rest
    m = _BOLD_ONLY.match(t)
    if m:
        return 6, m.group(1).strip(), ""
    return None


class Chunker:
    # (target, max) tokens per document profile
    PROFILES = {
        "prose":  (350, 500),    # reports, policies, articles
        "table":  (250, 400),    # spreadsheets, table-heavy documents
        "slides": (300, 450),    # one slide per chunk when it fits
        "short":  (200, 300),    # FAQs, forms, short-line documents
    }

    def __init__(self, chunk_algorithm: str = "structure", overlap_tokens: int = 50,
                 min_tokens: int = 25, fixed_size: int = 350):
        if chunk_algorithm not in ("structure", "fixed"):
            raise ValueError("chunk_algorithm must be 'structure' or 'fixed'")
        self.chunk_algorithm = chunk_algorithm
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens
        self.fixed_size = fixed_size

    # ================================================================ public
    def chunk_text(self, md: str, source: str):
        """Returns (chunks, parents). Each chunk is a dict with text + metadata."""
        if self.chunk_algorithm == "fixed":
            chunks, parents, profile = self._fixed(md, source), [], "fixed"
        else:
            profile, chunks, parents = self._structure(md, source)
        for c in chunks:
            c["profile"] = profile
            c["tokens"] = ntok(c["text"])
            where = f" p.{c['page']}" if c["page"] else ""
            c["embed_text"] = f"[{source}{where} > {c['section'] or 'start'}]\n{c['text']}"
        return chunks, parents

    def chunk_files(self, paths):
        """Chunk many parsed .md files; drops duplicate chunks; assigns stable ids."""
        all_chunks, all_parents, seen = [], [], set()
        for path in paths:
            source = path.name[:-3] if path.name.endswith(".md") else path.name
            chunks, parents = self.chunk_text(path.read_text(encoding="utf-8", errors="ignore"), source)
            for c in chunks:
                h = hashlib.md5(re.sub(r"\s+", " ", c["text"]).encode()).hexdigest()
                if h in seen:                 # repeated headers / footers
                    continue
                seen.add(h)
                c["id"] = f"{source}::{len(all_chunks)}"
                all_chunks.append(c)
            all_parents += parents
        return all_chunks, all_parents

    # ======================================================= fixed (baseline)
    def _fixed(self, md, source):
        pages = self._split_pages(md)
        chunks = []
        for page, text in pages:
            words = text.split()
            step = max(1, int((self.fixed_size - self.overlap_tokens) / 1.35))
            size = int(self.fixed_size / 1.35)
            for i in range(0, len(words), step):
                piece = " ".join(words[i:i + size])
                if piece.strip():
                    chunks.append({"source": source, "page": page, "section": "",
                                   "parent_id": "", "text": piece})
        return chunks

    @staticmethod
    def _split_pages(md):
        page, out, buf = None, [], []
        for line in md.splitlines():
            m = PAGE_RE.match(line.strip())
            if m:
                if buf:
                    out.append((page, "\n".join(buf)))
                page, buf = int(m.group(1)) + 1, []
            else:
                buf.append(line)
        if buf:
            out.append((page, "\n".join(buf)))
        return out

    # ===================================================== structure (best)
    @staticmethod
    def _to_blocks(md):
        """Markdown -> blocks: heading / table / list / code / para, each with its page."""
        blocks, page, buf, kind, in_code = [], None, [], None, False

        def flush():
            nonlocal buf, kind
            text = "\n".join(buf).strip()
            if text:
                blocks.append({"type": kind, "text": text, "page": page})
            buf, kind = [], None

        for line in md.splitlines():
            m = PAGE_RE.match(line.strip())
            if m and not in_code:
                flush()
                page = int(m.group(1)) + 1
                continue
            if line.strip().startswith("```"):
                if in_code:
                    buf.append(line); in_code = False; flush(); continue
                flush(); in_code, kind = True, "code"; buf.append(line); continue
            if in_code:
                buf.append(line); continue
            if not line.strip():
                flush(); continue
            h = HEADING_RE.match(line)
            if h:
                flush()
                blocks.append({"type": "heading", "level": len(h.group(1)),
                               "text": h.group(2).replace("**", "").strip("*_ "), "page": page})
                continue
            ah = arabic_heading(line) if not line.lstrip().startswith(("-", "*", "•")) else None
            if ah:
                flush()
                blocks.append({"type": "heading", "level": ah[0], "text": ah[1], "page": page})
                if ah[2]:
                    kind = "para"
                    buf.append(ah[2])
                continue
            if not line.lstrip().startswith("|"):
                line = line.replace("**", "")
            t = "table" if line.lstrip().startswith("|") else \
                "list" if LIST_RE.match(line) else "para"
            if kind and t != kind and not (kind == "list" and t == "para" and line.startswith("  ")):
                flush()
            kind = kind or t
            buf.append(line)
        flush()
        return blocks

    @staticmethod
    def _profile(source, blocks):
        ext = source.rsplit(".", 1)[-1].lower() if "." in source else ""
        if ext == "pptx":
            return "slides"
        body = [b for b in blocks if b["type"] != "heading"]
        if not body:
            return "prose"
        total = sum(ntok(b["text"]) for b in body)
        table_share = sum(ntok(b["text"]) for b in body if b["type"] == "table") / total
        if ext in ("xlsx", "csv") or table_share > 0.5:
            return "table"
        if total / len(body) < 40:
            return "short"
        return "prose"

    @staticmethod
    def _split_big(block, max_tok):
        """Split one oversized block along its natural seams."""
        text, kind = block["text"], block["type"]
        lines = text.splitlines()
        if kind == "table" and len(lines) > 2:
            header = lines[:2] if set(lines[1].replace("|", "").strip()) <= set("-: ") else lines[:1]
            pieces, cur = [], []
            for r in lines[len(header):]:
                if cur and ntok("\n".join(header + cur + [r])) > max_tok:
                    pieces.append("\n".join(header + cur)); cur = []
                cur.append(r)
            if cur:
                pieces.append("\n".join(header + cur))
            return pieces
        if kind == "list":
            units = re.split(r"\n(?=\s*(?:[-*+•]|\d+[.)])\s)", text)
        elif kind == "code":
            units = lines
        else:
            units = [s for s in SENT_RE.split(text) if s and s.strip()]
        pieces, cur = [], ""
        for u in units:
            while ntok(u) > max_tok:                  # one giant sentence
                words, cut = u.split(), int(max_tok / 1.35)
                pieces.append(" ".join(words[:cut])); u = " ".join(words[cut:])
            if cur and ntok(cur + " " + u) > max_tok:
                pieces.append(cur.strip()); cur = ""
            cur += ("\n" if kind in ("list", "code") else " ") + u
        if cur.strip():
            pieces.append(cur.strip())
        return pieces

    def _tail_overlap(self, text):
        if self.overlap_tokens <= 0 or text.lstrip().startswith("|"):
            return ""                                 # never overlap tables
        out = []
        for s in reversed([s for s in SENT_RE.split(text) if s.strip()]):
            if ntok(" ".join([s] + out)) > self.overlap_tokens:
                break
            out.insert(0, s.strip())
        return " ".join(out)

    def _structure(self, md, source):
        blocks = self._to_blocks(md)
        profile = self._profile(source, blocks)
        target, max_tok = self.PROFILES[profile]
        chunks, parents, path_stack = [], [], []
        sec = {"id": None, "blocks": [], "page": None}

        def section_path():
            return " > ".join(t for _, t in path_stack)

        def pack(blocks_, parent_id):
            cur, cur_page, carry = [], None, ""

            def emit():
                nonlocal cur, carry
                body = "\n\n".join(cur).strip()
                cur = []
                if not body:
                    return
                prev = chunks[-1] if chunks else None
                if (ntok(body) < self.min_tokens and prev and prev["parent_id"] == parent_id
                        and ntok(prev["text"] + body) <= max_tok):
                    prev["text"] += "\n\n" + body            # too small: merge back
                    return
                chunks.append({"source": source, "page": cur_page, "section": section_path(),
                               "parent_id": parent_id, "text": body})
                carry = self._tail_overlap(body)

            for b in blocks_:
                pieces = self._split_big(b, max_tok) if ntok(b["text"]) > max_tok else [b["text"]]
                if b["type"] == "list" and len(pieces) > 1 and path_stack:      # steps split: keep the heading
                    pieces = pieces[:1] + [f"{path_stack[-1][1]}:\n{x}" for x in pieces[1:]]
                for p in pieces:
                    page_change = profile == "slides" and cur and b["page"] != cur_page
                    if cur and (ntok("\n\n".join(cur + [p])) > target or page_change):
                        emit()
                        if (carry and b["type"] != "table" and not page_change
                                and ntok(carry + " " + p) <= max_tok):
                            cur.append(carry)
                    if not cur:
                        cur_page = b["page"]
                    cur.append(p)
            if cur:
                emit()

        def close_section():
            if not sec["blocks"]:
                return
            parents.append({"id": sec["id"], "source": source, "section": section_path(),
                            "page": sec["page"],
                            "text": "\n\n".join(b["text"] for b in sec["blocks"])})
            pack(sec["blocks"], sec["id"])
            sec["blocks"] = []

        for b in blocks:
            if b["type"] == "heading":
                close_section()
                while path_stack and path_stack[-1][0] >= b["level"]:
                    path_stack.pop()
                path_stack.append((b["level"], b["text"][:80]))
                sec["id"], sec["page"] = f"{source}#s{len(parents)}", b["page"]
                continue
            if sec["id"] is None:
                sec["id"], sec["page"] = f"{source}#s0", b["page"]
            if profile == "slides" and sec["blocks"] and b["page"] != sec["blocks"][-1]["page"]:
                close_section()
                sec["id"], sec["page"] = f"{source}#p{b['page']}", b["page"]
            sec["blocks"].append(b)
        close_section()
        return profile, chunks, parents


ch = Chunker          # keeps the short name `ch` your team started with


if __name__ == "__main__":
    # python DB/chunks.py [structure|fixed] [file.md ...]   (default: data/parsed/*.md)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    algo = args.pop(0) if args and args[0] in ("structure", "fixed") else "structure"
    files = [Path(a) for a in args] or sorted(
        (Path(__file__).resolve().parent.parent / "data" / "parsed").glob("*.md"))
    chunks, parents = Chunker(algo).chunk_files(files)
    for path in files:
        source = path.name[:-3]
        mine = [c for c in chunks if c["source"] == source]
        if mine:
            toks = [c["tokens"] for c in mine]
            print(f"{source:40} {mine[0]['profile']:7} {len(mine):4} chunks  "
                  f"tokens avg {sum(toks) // len(toks):4} max {max(toks):4}")
    print(f"\n{len(chunks)} chunks, {len(parents)} parent sections ({algo})")
    if chunks:
        print("\nexample:\n" + chunks[0]["embed_text"][:500])