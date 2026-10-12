"""Glue: Parser -> Chunker -> Embedder -> VectorStore, plus search for the Agent.

Usage (from Agent/ or api.py):
    from DB.pipeline import Pipeline
    db = Pipeline()
    db.ingest()                                  # once per document set
    hits = db.search("What is the hotel limit in Dubai?")
    context, hits = db.context("...")            # numbered sources for the LLM prompt
"""
import json
import time
from pathlib import Path

from .chroma import VectorStore
from .chunks import Chunker
from .embedding import Embedder
from .parse import Parser

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = {
    "raw_dir": ROOT / "data/raw",            # put documents here
    "parsed_dir": ROOT / "data/parsed",
    "db_dir": ROOT / "data/chroma",
    "embed_model": "bge-m3",
    "chunk_algorithm": "structure",          # or "fixed"
    "top_k_retrieve": 20,                    # candidates from hybrid search
    "top_k_final": 5,                        # chunks given to the LLM
    "use_reranker": False,                   # True = better ranking, ~2 GB more RAM
    "reranker_model": "BAAI/bge-reranker-v2-m3",
    "expand_parents": True,                  # give the LLM the whole section if small
    "parent_max_tokens": 700,
}


class Pipeline:
    def __init__(self, **overrides):
        self.cfg = {**DEFAULTS, **overrides}
        c = self.cfg
        self.parser = Parser(c["raw_dir"], c["parsed_dir"])
        self.chunker = Chunker(c["chunk_algorithm"])
        self.embedder = Embedder(c["embed_model"])
        self.store = VectorStore(c["db_dir"])
        self.meta_path = Path(c["db_dir"]) / "index_meta.json"
        self._reranker = None

    # ================================================================ ingest
    def _signature(self):
        files = {f.name: f.stat().st_mtime for f in sorted(Path(self.cfg["parsed_dir"]).glob("*.md"))}
        return {"files": files, "embed_model": self.cfg["embed_model"],
                "chunk_algorithm": self.cfg["chunk_algorithm"]}

    def ingest(self, force: bool = False) -> int:
        """Parse new/changed documents, then rebuild the index if anything changed."""
        t0 = time.time()
        self.parser.parse_all(force=force)
        sig = self._signature()
        if not sig["files"]:
            print("ingest: nothing to index")
            return 0
        if (not force and self.meta_path.exists() and self.store.count()
                and json.loads(self.meta_path.read_text()) == sig):
            print(f"ingest: index up to date ({self.store.count()} chunks)")
            return self.store.count()

        paths = sorted(Path(self.cfg["parsed_dir"]).glob("*.md"))
        chunks, parents = self.chunker.chunk_files(paths)
        print(f"chunks: {len(chunks)} from {len(paths)} files ({self.cfg['chunk_algorithm']})")
        vectors = self.embedder.embed_documents([c["embed_text"] for c in chunks])

        self.store.reset()
        self.store.add(chunks, vectors)
        self.store.save_parents(parents)
        self.meta_path.write_text(json.dumps(sig))
        print(f"ingest: {self.store.count()} chunks indexed in {time.time() - t0:.1f}s")
        return self.store.count()

    # ================================================================ search
    def _rerank(self, query, hits):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.cfg["reranker_model"], max_length=512)
        scores = self._reranker.predict([(query, h["text"]) for h in hits])
        for h, s in zip(hits, scores):
            h["rerank"] = round(float(s), 4)
        return sorted(hits, key=lambda h: -h["rerank"])

    def search(self, query: str, k: int = None, rerank: bool = None):
        """Hybrid search (+ optional rerank). Returns the best chunk dicts."""
        k = k or self.cfg["top_k_final"]
        rerank = self.cfg["use_reranker"] if rerank is None else rerank
        qvec = self.embedder.embed_query(query)
        hits = self.store.hybrid_search(query, qvec, k=self.cfg["top_k_retrieve"])
        if rerank and hits:
            try:
                hits = self._rerank(query, hits)
            except Exception as e:                  # missing package / no memory
                print(f"(reranker skipped: {e})")
        hits = hits[:k]
        if self.cfg["expand_parents"]:
            for h in hits:
                p = self.store.parent(h.get("parent_id"))
                if p and len(p["text"].split()) * 1.35 <= self.cfg["parent_max_tokens"]:
                    h["context"] = p["text"]        # whole (small) section
        return hits

    @staticmethod
    def cite(h) -> str:
        return f"{h['source']}, p.{h['page']}" if h.get("page") else h["source"]

    def context(self, query: str, k: int = None):
        """Numbered sources ready to paste into an LLM prompt -> (text, hits)."""
        hits = self.search(query, k)
        blocks, seen = [], set()
        for i, h in enumerate(hits, 1):
            body = h.get("context") or h["text"]
            key = h.get("parent_id") if h.get("context") else h["id"]
            if key in seen:                          # two chunks from the same section
                body = "(same section as above)"
            seen.add(key)
            blocks.append(f"[{i}] ({self.cite(h)} | {h['section'] or '-'})\n{body}")
        return "\n\n".join(blocks), hits

    def stats(self):
        return {"chunks": self.store.count(), "files": self.store.sources(),
                "embed_model": self.cfg["embed_model"],
                "chunk_algorithm": self.cfg["chunk_algorithm"]}