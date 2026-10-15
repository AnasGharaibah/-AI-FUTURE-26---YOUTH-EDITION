"""Step 3 - text -> vectors with a local Ollama embedding model.

Embeddings are cached on disk, so re-ingesting unchanged chunks is instant.

Usage:
    from DB.embedding import Embedder
    emb = Embedder("bge-m3")
    vectors = emb.embed_documents(["text one", "text two"])
    qvec = emb.embed_query("my question")
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Embedder:
    def __init__(self, model: str = "bge-m3", batch_size: int = 32,
                 cache_path=ROOT / "data/embed_cache.jsonl", query_on_cpu: bool = True):
        self.model = model
        self.batch_size = batch_size
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache = self._load_cache()
        self._client = None
        # questions are embedded on the CPU so the GPU stays free for the chat model, Whisper and OmniVoice during a
        # voice turn (bge-m3 on CPU: one short query is fast); ingest (many chunks) still uses the GPU.
        self.query_on_cpu = query_on_cpu

    # ----------------------------------------------------------------- cache
    def _key(self, text: str) -> str:
        return hashlib.md5(f"{self.model}\n{text}".encode()).hexdigest()

    def _load_cache(self) -> dict:
        cache = {}
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.open(encoding="utf-8"):
                k, v = json.loads(line)
                cache[k] = v
        return cache

    def _save(self, items):
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as fh:
            for k, v in items:
                fh.write(json.dumps([k, v]) + "\n")

    # ------------------------------------------------------------- embedding
    def _call(self, texts, cpu: bool = False):
        import ollama
        opts = {"options": {"num_gpu": 0}} if cpu else {}
        try:
            if self._client is None:
                self._client = ollama.Client(timeout=120)         # never hang forever on a stuck runner
            try:
                return self._client.embed(model=self.model, input=texts, **opts)["embeddings"]
            except Exception as e:
                if "timeout" not in type(e).__name__.lower():
                    raise
                return self._client.embed(model=self.model, input=texts, **opts)["embeddings"]   # one retry
        except Exception as e:
            msg = str(e).lower()
            if "not found" in msg:
                raise RuntimeError(f"Embedding model missing. Run: ollama pull {self.model}") from e
            if "connect" in msg or "refused" in msg:
                raise RuntimeError("Ollama is not running. Open the Ollama app.") from e
            raise

    def embed_documents(self, texts, show_progress: bool = True, cpu: bool = False):
        out = [None] * len(texts)
        todo = []
        for i, t in enumerate(texts):
            k = self._key(t)
            if k in self._cache:
                out[i] = self._cache[k]
            else:
                todo.append(i)
        if show_progress:
            print(f"embedding: {len(texts) - len(todo)} cached, {len(todo)} new ({self.model})")
        for s in range(0, len(todo), self.batch_size):
            idx = todo[s:s + self.batch_size]
            vecs = self._call([texts[i] for i in idx], cpu=cpu)
            new = []
            for i, v in zip(idx, vecs):
                out[i] = v
                k = self._key(texts[i])
                self._cache[k] = v
                new.append((k, v))
            self._save(new)
            if show_progress and len(todo) > self.batch_size:
                print(f"  {min(s + self.batch_size, len(todo))}/{len(todo)}")
        return out

    def embed_query(self, text: str):
        return self._call([text], cpu=self.query_on_cpu)[0]       # queries are never cached

    @property
    def dim(self) -> int:
        return len(self.embed_query("dimension check"))