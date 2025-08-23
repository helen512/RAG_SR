#!/usr/bin/env python3
"""
Hybrid query pipeline over Qdrant (dense) + BM25 (sparse) with optional cross-encoder re-ranking.

- Connects to the SAME persistent Qdrant collection you indexed with haystack_pipeline.py
- Rebuilds the BM25 in-memory store from your JSONL cache (or you can point it at another persistent store)
- Does hybrid retrieval via Reciprocal Rank Fusion (RRF)
- Optionally re-ranks the fused candidates with a cross-encoder (SentenceTransformersReranker)

CLI EXAMPLE
-----------
python haystack_query_pip.py \
  --qdrant-collection papers \
  --qdrant-persist ./qdrant_papers \
  --bm25-cache ./bm25_cache.jsonl \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --dense-k 20 --sparse-k 40 --final-k 10 \
  --query "what is lyapunov stability?"

NOTES
-----
- embedding-dim is NOT required here; we only query vectors, not create collections. It must match what you indexed with.
- If you change the embedding model, ensure the Qdrant collection was built with the same embedding dimensionality.
- If you don't want re-ranking, pass --no-rerank.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any

# -----------------------------
# Haystack imports (with gentle fallbacks)
# -----------------------------

from haystack.dataclasses import Document


try:
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
    from haystack.components.retrievers import InMemoryBM25Retriever
except Exception:
    try:
        from haystack.document_stores import QdrantDocumentStore, InMemoryDocumentStore  # type: ignore
        from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
        from haystack.components.retrievers import InMemoryBM25Retriever
    except Exception as e:
        raise ImportError("Cannot import Haystack document stores and retrievers; install haystack and qdrant-client") from e

# Embedding (query side)
_query_embedder_fallback = False
try:
    from haystack.components.embedders import SentenceTransformersTextEmbedder
except Exception:
    _query_embedder_fallback = True
    from sentence_transformers import SentenceTransformer  # type: ignore

    class SentenceTransformersTextEmbedder:  # type: ignore
        def __init__(self, model: str):
            self._m = SentenceTransformer(model)
        def warm_up(self) -> None:
            _ = self._m.encode(["warmup"], show_progress_bar=False)
        def run(self, text: str) -> Dict[str, Any]:
            vec = self._m.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
            return {"embedding": vec.tolist()}

# BM25 retriever is now handled by InMemoryBM25Retriever (imported above)

# Reranker (cross-encoder)
_reranker_fallback = False
try:
    from haystack.components.rankers import SentenceTransformersReranker
except Exception:
    _reranker_fallback = True
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except Exception as e:
        CrossEncoder = None  # type: ignore


# -----------------------------
# Utilities
# -----------------------------

def load_documents_jsonl(path: Path) -> List[Document]:
    """Load Document(id, content, meta) triplets from a JSONL cache.
    Matches the format written by haystack_pipeline.py
    """
    out: List[Document] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.append(Document(id=rec.get("id"), content=rec.get("content"), meta=rec.get("meta", {})))
    return out


def pretty(doc: Document, rank: int | None = None) -> str:
    src = (doc.meta or {}).get("source", "?")
    btype = (doc.meta or {}).get("block_type")
    score = getattr(doc, "score", None)
    prefix = f"#{rank} " if rank is not None else ""
    head = doc.content[:160].replace("\n", " ") + ("…" if doc.content and len(doc.content) > 160 else "")
    return f"{prefix}[{score:.4f} | {src} | {btype}] {head}"


def rrf_fuse(dense: List[Document], sparse: List[Document], k: int = 60, weight_dense: float = 1.0, weight_sparse: float = 1.0) -> List[Document]:
    """Reciprocal Rank Fusion (RRF) with simple weighting.
    score = sum(w_i * 1/(k + rank_i)) across systems; lower rank is better (0-based).
    """
    def accumulate(acc: Dict[str, float], docs: List[Document], weight: float):
        for i, d in enumerate(docs):
            rid = d.id or f"anon-{i}"
            acc[rid] = acc.get(rid, 0.0) + weight * (1.0 / (k + i + 1))
    acc: Dict[str, float] = {}
    accumulate(acc, dense, weight_dense)
    accumulate(acc, sparse, weight_sparse)

    # Build a map id->doc preferring the best-scored version
    by_id: Dict[str, Document] = {}
    for lst in (dense, sparse):
        for d in lst:
            did = d.id or id(d)
            if did not in by_id:
                by_id[did] = d
    # attach fused score
    fused: List[Document] = []
    for did, s in acc.items():
        d = by_id.get(did)
        if d is None:
            continue
        d.score = s  # type: ignore[attr-defined]
        fused.append(d)
    fused.sort(key=lambda x: getattr(x, "score", 0.0), reverse=True)
    return fused


# -----------------------------
# Builders
# -----------------------------

def build_dense_store(collection: str, persist_path: str | None, embedding_dim: int = 384) -> QdrantDocumentStore:
    kwargs = {"index": collection, "embedding_dim": embedding_dim}
    if persist_path:
        kwargs.update({"path": persist_path, "on_disk": True})
    else:
        kwargs.update({"location": ":memory:"})  # will be empty unless you also indexed in-memory
    return QdrantDocumentStore(**kwargs)


def build_sparse_store(cache_jsonl: Path | None) -> InMemoryDocumentStore:
    store = InMemoryDocumentStore()
    if cache_jsonl and cache_jsonl.exists():
        docs = load_documents_jsonl(cache_jsonl)
        if docs:
            # Deduplicate documents by ID to avoid DuplicateDocumentError
            seen_ids = set()
            deduped_docs = []
            for doc in docs:
                if doc.id not in seen_ids:
                    deduped_docs.append(doc)
                    seen_ids.add(doc.id)
            
            store.write_documents(deduped_docs)
            logging.info("Loaded %d docs into BM25 in-memory store (deduped from %d)", len(deduped_docs), len(docs))
    else:
        logging.warning("No BM25 cache found; sparse retriever will return nothing.")
    return store


@dataclass
class QuerySettings:
    dense_k: int = 20
    sparse_k: int = 40
    final_k: int = 10
    weight_dense: float = 1.0
    weight_sparse: float = 1.0


class HybridQueryEngine:
    def __init__(
        self,
        qdrant_store: QdrantDocumentStore,
        sparse_store: InMemoryDocumentStore,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str | None = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        settings: QuerySettings = QuerySettings(),
    ) -> None:
        self.qdrant = qdrant_store
        self.sparse = sparse_store
        self.embedder = SentenceTransformersTextEmbedder(model=embedding_model)
        self.embedder.warm_up()
        self.settings = settings

        # Initialize retrievers for Haystack 2.x
        self.dense_retriever = QdrantEmbeddingRetriever(document_store=qdrant_store)
        self.sparse_retriever = InMemoryBM25Retriever(document_store=sparse_store)

        self.reranker_model = reranker_model
        self.reranker = None
        if reranker_model:
            if not _reranker_fallback:
                self.reranker = SentenceTransformersReranker(model=reranker_model)
            else:
                if CrossEncoder is None:
                    logging.warning("No reranker available; install sentence-transformers to enable cross-encoder re-ranking.")
                else:
                    self.reranker = CrossEncoder(reranker_model)

        # BM25 retrieval is handled by InMemoryBM25Retriever above

    def dense_retrieve(self, query: str, top_k: int) -> List[Document]:
        # Embed the query and ask Qdrant for nearest neighbors
        vec = self.embedder.run(text=query)["embedding"]
        # Use QdrantEmbeddingRetriever for Haystack 2.x
        result = self.dense_retriever.run(query_embedding=vec, top_k=top_k)
        # haystack v2 returns {"documents": [...]}
        docs = result.get("documents", [])
        return docs

    def sparse_retrieve(self, query: str, top_k: int) -> List[Document]:
        # Use InMemoryBM25Retriever for Haystack 2.x
        result = self.sparse_retriever.run(query=query, top_k=top_k)
        # haystack v2 returns {"documents": [...]}
        docs = result.get("documents", [])
        return docs

    def fuse(self, dense_docs: List[Document], sparse_docs: List[Document]) -> List[Document]:
        s = self.settings
        fused = rrf_fuse(dense_docs, sparse_docs, k=60, weight_dense=s.weight_dense, weight_sparse=s.weight_sparse)
        return fused

    def rerank(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        if not self.reranker:
            return docs[:top_k]
        # Two code paths depending on implementation
        try:
            # Haystack component API
            out = self.reranker.run(query=query, documents=docs)  # type: ignore[call-arg]
            reranked = out.get("documents", docs)
            return reranked[:top_k]
        except Exception:
            # Plain sentence-transformers CrossEncoder
            pairs = [(query, d.content or "") for d in docs]
            scores = self.reranker.predict(pairs)  # type: ignore[attr-defined]
            for d, s in zip(docs, scores):
                d.score = float(s)  # type: ignore[attr-defined]
            docs.sort(key=lambda x: getattr(x, "score", 0.0), reverse=True)
            return docs[:top_k]

    def query(self, query: str) -> List[Document]:
        s = self.settings
        dense_docs = self.dense_retrieve(query, s.dense_k)
        sparse_docs = self.sparse_retrieve(query, s.sparse_k)
        fused = self.fuse(dense_docs, sparse_docs)
        final_docs = self.rerank(query, fused, s.final_k)
        return final_docs


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid query over Qdrant (dense) + BM25 (sparse) with optional re-ranking")
    p.add_argument("--qdrant-collection", type=str, default="papers")
    p.add_argument("--qdrant-persist", type=str, default="./qdrant_papers")
    p.add_argument("--bm25-cache", type=Path, default=Path("./bm25_cache.jsonl"), help="JSONL cache to populate the in-memory BM25 store")

    p.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--reranker-model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder re-ranking")

    p.add_argument("--dense-k", type=int, default=20)
    p.add_argument("--sparse-k", type=int, default=40)
    p.add_argument("--final-k", type=int, default=10)
    p.add_argument("--wd", type=float, default=1.0, help="Weight for dense scores in RRF")
    p.add_argument("--ws", type=float, default=1.0, help="Weight for sparse scores in RRF")

    p.add_argument("--query", type=str, default=None, help="Single-shot query to run and print results")
    p.add_argument("--interactive", action="store_true", help="Interactive REPL mode")
    p.add_argument("--loglevel", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    # Connect to stores
    qdrant = build_dense_store(args.qdrant_collection, args.qdrant_persist)
    bm25 = build_sparse_store(args.bm25_cache)

    settings = QuerySettings(
        dense_k=args.dense_k, sparse_k=args.sparse_k, final_k=args.final_k,
        weight_dense=args.wd, weight_sparse=args.ws,
    )
    reranker_model = None if args.no_rerank else args.reranker_model

    engine = HybridQueryEngine(
        qdrant_store=qdrant,
        sparse_store=bm25,
        embedding_model=args.embedding_model,
        reranker_model=reranker_model,
        settings=settings,
    )

    def run_query(q: str) -> None:
        docs = engine.query(q)
        print("\n=== RESULTS (top {}): {} ===".format(len(docs), q))
        for i, d in enumerate(docs, 1):
            print(pretty(d, i))

    if args.query:
        run_query(args.query)
    if args.interactive:
        try:
            while True:
                q = input("\nquery> ").strip()
                if not q:
                    continue
                if q.lower() in {"exit", "quit", ":q", "q"}:
                    break
                run_query(q)
        except (EOFError, KeyboardInterrupt):
            pass


if __name__ == "__main__":
    main()
