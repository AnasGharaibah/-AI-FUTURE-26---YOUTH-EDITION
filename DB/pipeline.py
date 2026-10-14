"""Glue: Parser -> Chunker -> Embedder -> VectorStore, plus search for the Agent.

Usage (from Agent/ or api.py):
    from DB.pipeline import Pipeline
    db = Pipeline()
    db.ingest()                                  # once per document set
    hits = db.search("What is the hotel limit in Dubai?")
    context, hits = db.context("...")            # numbered sources for the LLM prompt
"""
import json
import threading
import time
from pathlib import Path

from .arabic import is_arabic, normalize_display
from .chroma import VectorStore
from .chunks import Chunker, ntok
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
    "rewrite_query": True,                   # Arabic questions: also search an MSA rewrite (needs the LLM, cached)
    "rewriter": None,                        # callable(question) -> MSA question; None = Agent.llm.rewrite_to_msa
    "schema": 2,                             # bump when chunking/normalisation changes -> forces a re-index
}

_SHARED = None                               # see Pipeline.shared()
_SHARED_LOCK = threading.Lock()


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
                "chunk_algorithm": self.cfg["chunk_algorithm"], "schema": self.cfg["schema"]}

    def ingest(self, force: bool = False) -> int:
        """Parse new/changed documents, then rebuild the index if anything changed."""
        t0 = time.time()
        self.parser.parse_all(force=force)
        sig = self._signature()
        if not sig["files"]:
            print("ingest: nothing to index")
            return 0
        if (not force and self.meta_path.exists() and self.store.count()
                and json.loads(self.meta_path.read_text(encoding="utf-8")) == sig):
            print(f"ingest: index up to date ({self.store.count()} chunks)")
            return self.store.count()

        paths = sorted(Path(self.cfg["parsed_dir"]).glob("*.md"))
        chunks, parents = self.chunker.chunk_files(paths)
        print(f"chunks: {len(chunks)} from {len(paths)} files ({self.cfg['chunk_algorithm']})")
        vectors = self.embedder.embed_documents([c["embed_text"] for c in chunks])

        self.store.reset()
        self.store.add(chunks, vectors)
        self.store.save_parents(parents)
        self.meta_path.write_text(json.dumps(sig), encoding="utf-8")
        print(f"ingest: {self.store.count()} chunks indexed in {time.time() - t0:.1f}s")
        return self.store.count()

    # ================================================================ search
    def _rerank(self, query, hits):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            import torch
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            kw = {"dtype": torch.float16} if dev == "mps" else {}       # half the RAM (1.1 GB instead of 2.2)
            self._reranker = CrossEncoder(self.cfg["reranker_model"], max_length=512, device=dev, model_kwargs=kw)
        # same "[file > section]" context line the embedder saw - otherwise identical-looking chunks of different services tie
        scores = self._reranker.predict([(query, f"{h['source']} > {h['section'] or '-'}\n{h['text']}") for h in hits])
        for h, s in zip(hits, scores):
            h["rerank"] = round(float(s), 4)
        return sorted(hits, key=lambda h: -h["rerank"])

    def _queries(self, query: str, rewrite: bool):
        """[original] + [MSA rewrite] for Arabic questions (when enabled and the rewriter works)."""
        qs = [query]
        if rewrite and is_arabic(query):
            try:
                fn = self.cfg["rewriter"]
                if fn is None:
                    from Agent.llm import rewrite_to_msa as fn
                alt = normalize_display(fn(query))
                if alt and alt not in qs:
                    qs.append(alt)
            except Exception as e:                  # LLM down / DB used without Agent: plain search still works
                print(f"(query rewrite skipped: {e})")
        return qs

    def search(self, query: str, k: int = None, rerank: bool = None, rewrite: bool = None, sources=None):
        """Hybrid search over the question (+ its MSA rewrite), RRF-fused, optional rerank.
        sources: optional list of file names to limit the search to. Returns the best chunk dicts."""
        k = k or self.cfg["top_k_final"]
        rerank = self.cfg["use_reranker"] if rerank is None else rerank
        rewrite = self.cfg["rewrite_query"] if rewrite is None else rewrite
        query = normalize_display(query)
        queries = self._queries(query, rewrite)

        fused, hits_by_id, why = {}, {}, {}
        for qi, q in enumerate(queries):
            qvec = self.embedder.embed_query(q)
            for rank, h in enumerate(self.store.hybrid_search(q, qvec, k=self.cfg["top_k_retrieve"], sources=sources)):
                fused[h["id"]] = fused.get(h["id"], 0) + 1 / (60 + rank)
                best = hits_by_id.setdefault(h["id"], h)
                best["sim"] = max(best.get("sim", 0), h.get("sim", 0))
                why.setdefault(h["id"], set()).update(h["found_by"].split("+"))
                if qi:
                    why[h["id"]].add("rewrite")
        order = sorted(fused, key=fused.get, reverse=True)[:self.cfg["top_k_retrieve"]]
        hits = []
        for cid in order:
            h = hits_by_id[cid]
            h["score"] = round(fused[cid], 5)
            h["found_by"] = "+".join(sorted(why[cid]))
            hits.append(h)

        if rerank and hits:
            try:
                hits = self._rerank(query, hits)
            except Exception as e:                  # missing package / no memory
                print(f"(reranker skipped: {e})")
        hits = hits[:k]
        if self.cfg["expand_parents"]:
            for h in hits:
                p = self.store.parent(h.get("parent_id"))
                if p and ntok(p["text"]) <= self.cfg["parent_max_tokens"]:
                    h["context"] = p["text"]        # whole (small) section
        return hits

    @staticmethod
    def shared() -> "Pipeline":
        """The one Pipeline of this process (one Chroma client, one BM25 index), created on first use.
        Agent.rag, Agent.tools and api.py all use it."""
        global _SHARED
        with _SHARED_LOCK:
            if _SHARED is None:
                _SHARED = Pipeline()
            return _SHARED

    @staticmethod
    def cite(h) -> str:
        return f"{h['source']}, p.{h['page']}" if h.get("page") else h["source"]

    def context(self, query: str, k: int = None, **kw):
        """Numbered sources ready to paste into an LLM prompt -> (text, hits)."""
        hits = self.search(query, k, **kw)
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