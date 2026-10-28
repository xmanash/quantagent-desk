"""Simple hybrid RAG retriever using Chroma + keyword fallback.

Designed to work offline with local embeddings (sentence-transformers).
Falls back gracefully if Chroma is unavailable.

Research results are clipped to a token budget so methodology context
does not flood the desk answer.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from quantagent_desk.utils.tokens import clip_text, estimate_tokens

DOCS_DIR = Path(__file__).resolve().parents[3] / "data" / "docs"


class SimpleRetriever:
    """Lightweight retriever that indexes markdown files in data/docs."""

    def __init__(self, persist_dir: Optional[str] = None):
        self.docs: List[Dict[str, str]] = []
        self._index: Dict[str, List[str]] = {}  # simple inverted index
        self._load_docs()
        self._build_keyword_index()

        self.chroma_available = False
        self.collection = None
        self._model = None
        try:
            import chromadb
            from chromadb.config import Settings

            client = chromadb.Client(Settings(anonymized_telemetry=False))
            self.collection = client.get_or_create_collection(
                name="quantagent_docs",
                metadata={"hnsw:space": "cosine"},
            )
            if self.collection.count() == 0:
                self._ingest_to_chroma()
            self.chroma_available = True
        except Exception:
            # Chroma or embeddings not ready – keyword mode only
            pass

    def _embeddings_enabled(self) -> bool:
        flag = os.getenv("QUANTAGENT_SKIP_EMBEDDINGS", "").strip().lower()
        return flag not in {"1", "true", "yes"}

    def _get_model(self):
        if not self._embeddings_enabled():
            return None
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self._model = None
        return self._model

    def _load_docs(self) -> None:
        if not DOCS_DIR.exists():
            return
        for path in DOCS_DIR.glob("*.md"):
            text = path.read_text(encoding="utf-8")
            # split into rough chunks by heading
            chunks = self._chunk_markdown(text, source=path.name)
            self.docs.extend(chunks)

    def _chunk_markdown(self, text: str, source: str, max_chars: int = 800) -> List[Dict[str, str]]:
        chunks = []
        current = []
        current_len = 0
        for line in text.splitlines():
            if line.startswith("#") and current:
                chunk_text = "\n".join(current).strip()
                if chunk_text:
                    chunks.append({"text": chunk_text, "source": source, "id": hashlib.md5(chunk_text.encode()).hexdigest()[:12]})
                current = [line]
                current_len = len(line)
            else:
                current.append(line)
                current_len += len(line) + 1
                if current_len > max_chars:
                    chunk_text = "\n".join(current).strip()
                    if chunk_text:
                        chunks.append({"text": chunk_text, "source": source, "id": hashlib.md5(chunk_text.encode()).hexdigest()[:12]})
                    current = []
                    current_len = 0
        if current:
            chunk_text = "\n".join(current).strip()
            if chunk_text:
                chunks.append({"text": chunk_text, "source": source, "id": hashlib.md5(chunk_text.encode()).hexdigest()[:12]})
        return chunks

    def _build_keyword_index(self) -> None:
        import re
        for doc in self.docs:
            words = set(re.sub(r"[^a-z0-9]", "", w.lower()) for w in doc["text"].split())
            words = {w for w in words if len(w) > 3}
            for w in words:
                self._index.setdefault(w, []).append(doc["id"])

    def _ingest_to_chroma(self) -> None:
        if not self.collection or not self.docs:
            return
        try:
            model = self._get_model()
            if model is None:
                raise RuntimeError("Embedding model unavailable")
            texts = [d["text"] for d in self.docs]
            ids = [d["id"] for d in self.docs]
            metadatas = [{"source": d["source"]} for d in self.docs]
            embeddings = model.encode(texts).tolist()
            self.collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)
        except Exception:
            # fallback – store without embeddings
            self.collection.add(
                documents=[d["text"] for d in self.docs],
                ids=[d["id"] for d in self.docs],
                metadatas=[{"source": d["source"]} for d in self.docs],
            )

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        max_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return list of {text, source, score}, clipped to max_tokens when set."""
        results: List[Dict[str, Any]] = []

        # Try Chroma first
        if self.chroma_available and self.collection:
            try:
                model = self._get_model()
                if model is not None:
                    q_emb = model.encode([query]).tolist()
                    res = self.collection.query(query_embeddings=q_emb, n_results=top_k)
                    for i, doc in enumerate(res["documents"][0]):
                        results.append({
                            "text": doc,
                            "source": res["metadatas"][0][i].get("source", "unknown"),
                            "score": 1.0 - (res["distances"][0][i] if res.get("distances") else 0.5),
                        })
                    if results:
                        return self._fit_budget(results, max_tokens)
            except Exception:
                pass

        # Keyword fallback
        import re
        q_words = set(re.sub(r"[^a-z0-9]", "", w.lower()) for w in query.split())
        q_words = {w for w in q_words if len(w) > 3}
        scores: Dict[str, float] = {}
        for w in q_words:
            for doc_id in self._index.get(w, []):
                scores[doc_id] = scores.get(doc_id, 0) + 1.0
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        id_to_doc = {d["id"]: d for d in self.docs}
        for doc_id, score in ranked:
            d = id_to_doc.get(doc_id)
            if d:
                results.append({"text": d["text"], "source": d["source"], "score": score})
        return self._fit_budget(results, max_tokens)

    def _fit_budget(
        self,
        results: List[Dict[str, Any]],
        max_tokens: Optional[int],
    ) -> List[Dict[str, Any]]:
        if not max_tokens or max_tokens <= 0:
            return results
        fitted: List[Dict[str, Any]] = []
        remaining = max_tokens
        for item in results:
            if remaining <= 16:
                break
            text = clip_text(item.get("text", ""), remaining)
            used = estimate_tokens(text)
            if not text:
                continue
            fitted.append({**item, "text": text, "tokens": used})
            remaining -= used
        return fitted


# Singleton for convenience
_retriever: Optional[SimpleRetriever] = None


def get_retriever() -> SimpleRetriever:
    global _retriever
    if _retriever is None:
        _retriever = SimpleRetriever()
    return _retriever
