"""Convert every document in data/raw/ to Markdown in data/parsed/.

PDF / images              -> Marker      (layout, tables, OCR)
DOCX/PPTX/XLSX/HTML/EPUB  -> MarkItDown  (no system libraries needed)
TXT / MD                  -> read directly

usage:
    uv run python parse.py               # parse new/changed files only
    uv run python parse.py --force       # re-parse everything
    uv run python parse.py --force-ocr   # OCR every PDF page (scanned PDFs)
"""
import argparse, json, re, time, traceback, warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Couldn't find ffmpeg")

RAW = Path("data/raw")
PARSED = Path("data/parsed")
MANIFEST = PARSED / "_manifest.json"

MARKER_EXT = {".pdf", ".png", ".jpg", ".jpeg"}
OFFICE_EXT = {".docx", ".pptx", ".xlsx", ".html", ".htm", ".epub"}
PLAIN_EXT = {".txt", ".md"}
PAGE_SEP = "-" * 48          # page separator: {N} + 48 dashes


def build_converter(force_ocr: bool = False):
    from marker.config.parser import ConfigParser
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    cfg = {
        "output_format": "markdown",
        "paginate_output": True,           # page markers -> page citations
        "disable_image_extraction": True,  # text only
        "disable_ocr": True,               # no llama-server needed

    }
    if force_ocr:
        cfg["force_ocr"] = True
    cp = ConfigParser(cfg)
    return PdfConverter(
        config=cp.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=cp.get_processors(),
        renderer=cp.get_renderer(),
    )


def pdf_fallback(path: Path) -> str:
    """Text-only extraction if Marker fails; keeps the same page markers."""
    from pypdf import PdfReader
    pages = PdfReader(str(path)).pages
    return "".join(f"\n\n{{{i}}}{PAGE_SEP}\n\n{p.extract_text() or ''}"
                   for i, p in enumerate(pages))


def office_to_md(path: Path) -> str:
    from markitdown import MarkItDown
    text = MarkItDown().convert(str(path)).text_content
    # PPTX: "<!-- Slide number: 3 -->" -> "{2}------..." so slides become pages
    return re.sub(r"<!-- Slide number: (\d+) -->",
                  lambda m: f"\n\n{{{int(m.group(1)) - 1}}}{PAGE_SEP}\n\n", text)


def out_path(src: Path) -> Path:
    return PARSED / f"{src.name}.md"      # report.pdf -> report.pdf.md


def parse_all(force=False, force_ocr=False) -> dict:
    PARSED.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    converter = None

    files = sorted(f for f in RAW.rglob("*") if f.is_file() and not f.name.startswith("."))
    if not files:
        print("data/raw/ is empty — add some documents first.")
    for f in files:
        ext, out = f.suffix.lower(), out_path(f)
        if not force and out.exists() and out.stat().st_mtime >= f.stat().st_mtime:
            continue

        t0, status, tool = time.time(), "ok", ""
        try:
            if ext in PLAIN_EXT:
                text, tool = f.read_text(errors="ignore"), "plain"
            elif ext in OFFICE_EXT:
                text, tool = office_to_md(f), "markitdown"
            elif ext in MARKER_EXT:
                if converter is None:
                    print("loading Marker models (first run downloads them)...")
                    converter = build_converter(force_ocr)
                from marker.output import text_from_rendered
                text, _, _ = text_from_rendered(converter(str(f)))
                tool = "marker"
            else:
                print(f"SKIP     {f.name} (unsupported type)")
                manifest[f.name] = {"status": "unsupported"}
                continue
        except Exception:
            traceback.print_exc()
            if ext == ".pdf":
                text, status, tool = pdf_fallback(f), "fallback", "pypdf"
            else:
                manifest[f.name] = {"status": "failed"}
                print(f"FAIL     {f.name}")
                continue
        if len(re.sub(r"\{\d+\}-{48}", "", text).strip()) < 50:
            status = "empty"
            print(f"WARNING  {f.name} has no extractable text (scanned?)")
        out.write_text(text, encoding="utf-8")
        pages = text.count(PAGE_SEP)
        manifest[f.name] = {"status": status, "tool": tool, "output": out.name,
                            "pages": pages, "chars": len(text),
                            "seconds": round(time.time() - t0, 1)}
        print(f"{status.upper():8} [{tool}] {f.name} -> {out.name} "
              f"({pages} pages, {len(text):,} chars, {time.time() - t0:.1f}s)")

    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-parse every file")
    ap.add_argument("--force-ocr", action="store_true", help="OCR all PDF pages")
    a = ap.parse_args()
    m = parse_all(force=a.force, force_ocr=a.force_ocr)
    bad = [k for k, v in m.items() if v["status"] != "ok"]
    print(f"\n{len(m) - len(bad)} ok, {len(bad)} need attention: {bad}")
