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

warnings.filterwarnings("ignore", message="Couldn't find ffmpeg")

ROOT = Path(__file__).resolve().parent.parent
PAGE_SEP = "-" * 48                       # page marker: "{N}" + 48 dashes
PAGE_RE = re.compile(r"\{(\d+)\}-{48}")


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

    def _marker_to_md(self, path: Path) -> str:
        if self._marker is None:
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

    # --------------------------------------------------------------- public
    def out_path(self, src: Path) -> Path:
        return self.parsed_dir / f"{src.name}.md"       # report.pdf -> report.pdf.md

    def convert(self, path: Path):
        """One file -> (markdown, tool, status)."""
        ext = path.suffix.lower()
        try:
            if ext in self.PLAIN_EXT:
                return path.read_text(errors="ignore"), "plain", "ok"
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
        manifest = json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}
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
            if len(PAGE_RE.sub("", text).strip()) < 50:
                status = "empty"
                print(f"WARNING  {f.name} has no extractable text (scanned?)")
            out.write_text(text, encoding="utf-8")
            pages = len(PAGE_RE.findall(text))
            manifest[f.name] = {"status": status, "tool": tool, "pages": pages,
                                "chars": len(text), "seconds": round(time.time() - t0, 1)}
            print(f"{status.upper():8} [{tool}] {f.name} ({pages} pages, {len(text):,} chars)")

        self.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        bad = [k for k, v in manifest.items() if v["status"] != "ok"]
        print(f"parse: {len(manifest) - len(bad)} ok, {len(bad)} need attention {bad or ''}")
        return manifest