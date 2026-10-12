"""Step 4 - store chunks and search them.

Chroma (on disk) holds vectors + text + metadata.
BM25 keyword index is rebuilt from Chroma in memory, so there is one source of truth.
hybrid_search() fuses vector + keyword results with reciprocal rank fusion.

Usage:
    from DB.chroma import VectorStore
    store = VectorStore()
    store.reset(); store.add(chunks, vectors); store.save_parents(parents)
    hits = store.hybrid_search(query_text, query_vector, k=20)
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _tok(text: str):
    return re.findall(r"\w+", text.lower())


class VectorStore:
    def __init__(self, path=ROOT / "data/chroma", collection: str = "docs"):
        import chromadb
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.name = collection
        self.col = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})
        self.parents_path = self.path / "parents.json"
        self._bm25 = None
        self._rows = None                # id -> chunk dict (lazy)
        self._parents = None

    # ------------------------------------------------------------- writing
    def reset(self):
        """Delete everything (used before a full rebuild)."""
        self.client.delete_collection(self.name)
        self.col = self.client.create_collection(self.name, metadata={"hnsw:space": "cosine"})
        self.parents_path.unlink(missing_ok=True)
        self._bm25 = self._rows = self._parents = None

    def add(self, chunks, vectors, batch: int = 500):
        for s in range(0, len(chunks), batch):
            part = chunks[s:s + batch]
            self.col.add(
                ids=[c["id"] for c in part],
                embeddings=vectors[s:s + batch],
                documents=[c["text"] for c in part],
                metadatas=[{"source": c["source"], "page": c["page"] or 0,
                            "section": c["section"] or "", "parent_id": c.get("parent_id") or "",
                            "profile": c.get("profile", ""), "tokens": c.get("tokens", 0)}
                           for c in part])
        self._bm25 = self._rows = None

    def save_parents(self, parents):
        self.parents_path.write_text(json.dumps({p["id"]: p for p in parents}, ensure_ascii=False))
        self._parents = None

    # ------------------------------------------------------------- reading
    def count(self) -> int:
        return self.col.count()

    def _load(self):
        if self._rows is None:
            got = self.col.get(include=["documents", "metadatas"])
            self._rows = {}
            for i, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
                self._rows[i] = {"id": i, "text": doc, **meta, "page": meta.get("page") or None}
            self._ids = list(self._rows)
            from rank_bm25 import BM25Okapi
            corpus = [_tok(f"{r['source']} {r['section']} {r['text']}") for r in self._rows.values()]
            self._bm25 = BM25Okapi(corpus) if corpus else None
        return self._rows

    def get(self, ids):
        rows = self._load()
        return [dict(rows[i]) for i in ids if i in rows]

    def parent(self, parent_id):
        if self._parents is None:
            self._parents = json.loads(self.parents_path.read_text()) if self.parents_path.exists() else {}
        return self._parents.get(parent_id)

    def sources(self):
        """{file name: number of chunks} - handy for a stats screen."""
        out = {}
        for r in self._load().values():
            out[r["source"]] = out.get(r["source"], 0) + 1
        return out

    # ------------------------------------------------------------ searching
    def vector_search(self, query_vector, k: int = 20):
        n = min(k, self.count())
        if n == 0:
            return []
        res = self.col.query(query_embeddings=[query_vector], n_results=n)
        return [(i, 1 - d) for i, d in zip(res["ids"][0], res["distances"][0])]   # (id, similarity)

    def keyword_search(self, query: str, k: int = 20):
        self._load()
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tok(query))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self._ids[i], float(scores[i])) for i in top if scores[i] > 0]

    def hybrid_search(self, query: str, query_vector, k: int = 20, rrf_k: int = 60):
        """Vector + keyword, fused by reciprocal rank. Returns chunk dicts with scores."""
        vec = self.vector_search(query_vector, k)
        kw = self.keyword_search(query, k)
        fused, why = {}, {}
        for name, ranked in (("vector", vec), ("keyword", kw)):
            for rank, (cid, _) in enumerate(ranked):
                fused[cid] = fused.get(cid, 0) + 1 / (rrf_k + rank)
                why.setdefault(cid, []).append(name)
        order = sorted(fused, key=fused.get, reverse=True)[:k]
        hits = self.get(order)
        for h in hits:
            h["score"] = round(fused[h["id"]], 5)
            h["found_by"] = "+".join(why[h["id"]])
        return hits