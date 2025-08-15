#!/usr/bin/env python3
"""
load_chunks_to_chroma.py

Reads a RAG chunks JSONL (one record per line: {"id","text","metadata":{...}})
and stores it into a persistent Chroma collection with Sentence-Transformer embeddings.

Usage:
  python load_chunks_to_chroma.py \
    --jsonl /path/to/murray_lyapunov.chunks.jsonl \
    --persist ./chroma_db \
    --collection cartpole_rag \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --batch 256 \
    --normalize

Optional:
  --field text_with_math     # if your JSONL has an alternate field; defaults to "text"
  --test-query "Lyapunov decrease dot V < 0"
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List

# ---- Chroma setup ----
try:
    import chromadb
    from chromadb.utils import embedding_functions
except Exception as e:
    raise SystemExit(
        "chromadb is not installed. Run: pip install chromadb sentence-transformers\n"
        f"Import error: {e}"
    )

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # tolerate trailing commas, etc.
                continue

def normalize_text(s: str) -> str:
    # light-normalize whitespace; keep $$..$$ intact
    return " ".join(s.split()).replace(" $ $ ", "$$")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Path to chunks.jsonl")
    ap.add_argument("--persist", required=True, help="Chroma persist directory")
    ap.add_argument("--collection", required=True, help="Collection name")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Sentence-Transformers model for embeddings")
    ap.add_argument("--batch", type=int, default=256, help="Batch size for upserts")
    ap.add_argument("--field", default="text", help="JSONL field to index (default: text)")
    ap.add_argument("--normalize", action="store_true", help="Light whitespace normalization")
    ap.add_argument("--test-query", default=None, help="Optional query to sanity-check retrieval")
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl).expanduser().resolve()
    persist_dir = Path(args.persist).expanduser().resolve()
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Embedding function
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=args.model  # e.g., "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Persistent client & collection
    try:
        client = chromadb.PersistentClient(path=str(persist_dir))
    except TypeError:
        # older API fallback
        client = chromadb.PersistentClient(path=str(persist_dir))

    collection = client.get_or_create_collection(
        name=args.collection,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}  # cosine is typical for ST embeddings
    )

    # Load and prepare records
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    seen = set()
    total = 0

    for rec in iter_jsonl(jsonl_path):
        rid = str(rec.get("id") or "")
        text = rec.get(args.field) or rec.get("text") or ""
        meta = rec.get("metadata") or {}

        if not rid or not text:
            continue
        if rid in seen:
            continue
        seen.add(rid)

        total += 1
        if args.normalize:
            text = normalize_text(text)

        ids.append(rid)
        docs.append(text)
        # Ensure metadata is JSON-serializable & not too large
        metas.append(meta)

        # flush in batches
        if len(ids) >= args.batch:
            collection.upsert(ids=ids, documents=docs, metadatas=metas)
            ids, docs, metas = [], [], []

    # final flush
    if ids:
        collection.upsert(ids=ids, documents=docs, metadatas=metas)

    # Persist (older chroma auto-persists; this is just explicit)
    try:
        client.persist()
    except Exception:
        pass

    print(f"✅ Ingest complete into collection '{args.collection}' at {persist_dir}")
    print(f"   Records ingested: {total}")

    if args.test_query:
        print(f"\n🔎 test query: {args.test_query}")
        res = collection.query(
            query_texts=[args.test_query],
            n_results=5,
            include=["documents", "metadatas", "distances", "ids"],
        )
        for i, (rid, doc, meta, dist) in enumerate(zip(
            res.get("ids", [[]])[0],
            res.get("documents", [[]])[0],
            res.get("metadatas", [[]])[0],
            res.get("distances", [[]])[0],
        ), start=1):
            page = meta.get("page")
            print(f"\n[{i}] id={rid}  dist={dist:.4f}  page={page}")
            preview = (doc[:300] + "…") if len(doc) > 300 else doc
            print(preview)

if __name__ == "__main__":
    main()



