"""Step 1 - documents -> Markdown.

PDF / images              -> Marker      (layout, tables)
DOCX/PPTX/XLSX/HTML/EPUB  -> MarkItDown  (no system libraries needed)
TXT / MD                  -> read directly

Usage:
    from DB.parse import Parser
    Parser().parse_all()          # data/raw -> data/parsed
"""
import json
import re
import time
import traceback
import warnings
from pathlib import Path

from .arabic import looks_reversed, normalize_display

warnings.filterwarnings("ignore", message="Couldn't find ffmpeg")

ROOT = Path(__file__).resolve().parent.parent
PAGE_SEP = "-" * 48                       # page marker: "{N}" + 48 dashes
PAGE_RE = re.compile(r"\{(\d+)\}-{48}")


_AR_WORD = re.compile("[\u0621-\u064a\u064b-\u0652\u0670]+")


def _letters_key(word: str) -> str:
    return "".join(sorted(re.sub("[\u064b-\u0652\u0670]", "", word)))


def repair_ligatures(md: str, reference: str) -> tuple[str, int]:
    """Marker/pdfium can return the letters of Arabic ligatures (لا لم لأ ...) in swapped order
    (املنتهي for المنتهي). `reference` is the same PDF's text from another extractor (pypdf) whose
    words are right but whose line order may be off. A word that is NOT in the reference vocabulary but
    is an anagram of exactly ONE reference word is replaced by it. Returns (text, words_fixed)."""
    ref = normalize_display(reference)
    vocab = set(_AR_WORD.findall(ref))
    by_key = {}
    for w in vocab:
        by_key.setdefault(_letters_key(w), set()).add(w)
    fixed = 0

    def sub(m):
        nonlocal fixed
        w = m.group(0)
        if w in vocab or len(w) < 3:
            return w
        cand = by_key.get(_letters_key(w), set())
        if len(cand) == 1:
            fixed += 1
            return next(iter(cand))
        return w
    return _AR_WORD.sub(sub, md), fixed


def fix_list_numbers(md: str) -> str:
    """Marker prints RTL list numbers back to front ('- .1 text' -> '1. text') and splits أولاً ('أو ًال')."""
    md = re.sub(r"(?m)^(\s*)[-*]\s+\.(\d+)\s+", r"\1\2. ", md)
    return re.sub("أو\\s*\u064b\\s*ال", "أولاً", md)


def scramble_ratio(md: str, reference: str) -> float:
    """Share of Arabic words (>= 3 letters) of `md` that the reference extractor never produced.
    High (> ~5%) = garbled text, typically RTL table cells."""
    vocab = set(_AR_WORD.findall(normalize_display(reference)))
    words = [w for w in _AR_WORD.findall(md) if len(w) >= 3]
    return sum(w not in vocab for w in words) / len(words) if words else 0.0


class Parser:
    MARKER_EXT = {".pdf", ".png", ".jpg", ".jpeg"}
    OFFICE_EXT = {".docx", ".pptx", ".xlsx", ".html", ".htm", ".epub"}
    PLAIN_EXT = {".txt", ".md"}

    def __init__(self, raw_dir=ROOT / "data/raw", parsed_dir=ROOT / "data/parsed",
                 use_ocr=False):
        self.raw_dir = Path(raw_dir)
        self.parsed_dir = Path(parsed_dir)
        self.use_ocr = use_ocr                # True needs llama-server installed
        self.manifest_path = self.parsed_dir / "_manifest.json"
        self._marker = None                   # loaded on first PDF only

    # ---------------------------------------------------------- converters
    def _load_marker(self):
        from marker.config.parser import ConfigParser
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        cp = ConfigParser({
            "output_format": "markdown",
            "paginate_output": True,           # page markers -> page citations
            "disable_image_extraction": True,
            "disable_ocr": not self.use_ocr,
        })
        return PdfConverter(config=cp.generate_config_dict(),
                            artifact_dict=create_model_dict(),
                            processor_list=cp.get_processors(),
                            renderer=cp.get_renderer())

    @staticmethod
    def _unload_ollama_models():
        """Marker needs GBs of RAM: never run it while Ollama holds a chat model (8 GB machines swap or crash).
        Unloads every loaded model; the embedder / chat model reload on their next request."""
        try:
            import ollama
            client = ollama.Client(timeout=30)
            for m in client.ps().get("models", []):
                name = m.get("model") or m.get("name")
                client.generate(model=name, prompt="", keep_alive=0)
                print(f"unloaded {name} from Ollama before running Marker")
        except Exception as e:                 # Ollama not running: nothing to unload
            print(f"(could not check Ollama models before Marker: {e})")

    def _marker_to_md(self, path: Path) -> str:
        if self._marker is None:
            self._unload_ollama_models()
            print("loading Marker models (slow the first time)...")
            try:
                self._marker = self._load_marker()
            except Exception:
                self._marker = False           # don't retry for every file
                raise
        if self._marker is False:
            raise RuntimeError("Marker unavailable")
        from marker.output import text_from_rendered
        text, _, _ = text_from_rendered(self._marker(str(path)))
        return text

    @staticmethod
    def _read_plain(path: Path) -> str:
        """UTF-8 (with or without BOM), UTF-16 with BOM, else Windows Arabic (cp1256). Never the OS default
        encoding: on Windows that is cp1252 and turns Arabic into mojibake."""
        raw = path.read_bytes()
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            return raw.decode("utf-16")
        try:
            return raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            return raw.decode("cp1256", errors="replace")

    @staticmethod
    def _office_to_md(path: Path) -> str:
        from markitdown import MarkItDown
        text = MarkItDown().convert(str(path)).text_content
        # PPTX "<!-- Slide number: 3 -->" -> page marker, so answers cite slides
        return re.sub(r"<!-- Slide number: (\d+) -->",
                      lambda m: f"\n\n{{{int(m.group(1)) - 1}}}{PAGE_SEP}\n\n", text)

    @staticmethod
    def _pdf_fallback(path: Path) -> str:
        from pypdf import PdfReader
        return "".join(f"\n\n{{{i}}}{PAGE_SEP}\n\n{p.extract_text() or ''}"
                       for i, p in enumerate(PdfReader(str(path)).pages))

    def release(self):
        """Drop the Marker models: they hold GBs of RAM that Ollama needs for the embedder / LLM."""
        if self._marker:
            import gc
            self._marker = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass

    @staticmethod
    def _pdf_empty_pages(path: Path) -> list:
        """1-based numbers of PDF pages without a text layer (scanned). We have no OCR, so only report."""
        try:
            from pypdf import PdfReader
            return [i + 1 for i, p in enumerate(PdfReader(str(path)).pages)
                    if len((p.extract_text() or "").strip()) < 10]
        except Exception:
            return []

    # --------------------------------------------------------------- public
    def out_path(self, src: Path) -> Path:
        return self.parsed_dir / f"{src.name}.md"       # report.pdf -> report.pdf.md

    def convert(self, path: Path):
        """One file -> (markdown, tool, status)."""
        ext = path.suffix.lower()
        try:
            if ext in self.PLAIN_EXT:
                return self._read_plain(path), "plain", "ok"
            if ext in self.OFFICE_EXT:
                return self._office_to_md(path), "markitdown", "ok"
            if ext in self.MARKER_EXT:
                return self._marker_to_md(path), "marker", "ok"
            return None, "", "unsupported"
        except Exception:
            traceback.print_exc()
            if ext == ".pdf":
                return self._pdf_fallback(path), "pypdf", "fallback"
            return None, "", "failed"

    def parse_all(self, force: bool = False) -> dict:
        """Parse new/changed files in raw_dir. Returns the manifest."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8")) if self.manifest_path.exists() else {}
        files = sorted(f for f in self.raw_dir.rglob("*")
                       if f.is_file() and not f.name.startswith("."))
        if not files:
            print(f"parse: no documents in {self.raw_dir}")
            return manifest

        names = {f.name for f in files}                  # forget deleted files
        for gone in [k for k in manifest if k not in names]:
            manifest.pop(gone)
            self.out_path(Path(gone)).unlink(missing_ok=True)

        for f in files:
            out = self.out_path(f)
            if not force and out.exists() and out.stat().st_mtime >= f.stat().st_mtime:
                continue
            t0 = time.time()
            text, tool, status = self.convert(f)
            if text is None:
                print(f"{status.upper():8} {f.name}")
                manifest[f.name] = {"status": status}
                continue
            text = normalize_display(text)               # NFKC, no tatweel, ASCII digits
            if tool == "marker":
                try:
                    ref = self._pdf_fallback(f)
                    text, n_fixed = repair_ligatures(text, ref)
                    text = fix_list_numbers(text).replace("<br>", " ")
                    if n_fixed:
                        print(f"repaired {n_fixed} words with swapped ligature letters in {f.name}")
                    garbled = scramble_ratio(text, ref)
                    if garbled > 0.05:
                        print(f"WARNING  {f.name}: {garbled:.0%} of Arabic words look garbled (RTL tables?) - check data/parsed")
                except Exception as e:
                    print(f"(ligature repair skipped for {f.name}: {e})")
            empty_pages = self._pdf_empty_pages(f) if f.suffix.lower() == ".pdf" else []
            if len(PAGE_RE.sub("", text).strip()) < 50:
                status = "empty"
                print(f"WARNING  {f.name} has no extractable text (scanned? no OCR available)")
            elif empty_pages:
                status = "partial"
                print(f"WARNING  {f.name}: pages {empty_pages} have no text layer (scanned) and are missing")
            rev = looks_reversed(text)
            if rev:
                print(f"WARNING  {f.name}: Arabic text looks REVERSED (bad PDF text layer) - answers may be wrong")
            out.write_text(text, encoding="utf-8")
            pages = len(PAGE_RE.findall(text))
            manifest[f.name] = {"status": status, "tool": tool, "pages": pages,
                                "chars": len(text), "seconds": round(time.time() - t0, 1)}
            if empty_pages:
                manifest[f.name]["empty_pages"] = empty_pages
            if rev:
                manifest[f.name]["reversed"] = True
            print(f"{status.upper():8} [{tool}] {f.name} ({pages} pages, {len(text):,} chars)")

        self.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        self.release()
        bad = [k for k, v in manifest.items() if v["status"] != "ok"]
        print(f"parse: {len(manifest) - len(bad)} ok, {len(bad)} need attention {bad or ''}")
        return manifest