#!/usr/bin/env python3
"""
Haystack ingestion pipeline (updated):
- Handles MANY Marker exports (multiple folders each with blocks.json [+ optional content/meta]).
- NEW: Also ingests Markdown (.md) files with section-based chunking.
- Persists Qdrant locally (no re-ingest on every run) and skips duplicates via stable IDs.
- Optionally keeps a JSONL cache to quickly repopulate the BM25 (sparse) store.
- Supports batch and streaming indexing for large documents (100+ pages).

CLI example (both sources):
    python haystack_ingestion.py \
        --marker-root rag_seed/marker \
        --md-root rag_seed/md \
        --qdrant-collection papers \
        --qdrant-persist ./qdrant_papers \
        --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
        --embedding-dim 384 \
        --batch-size 128 \
        --bm25-cache ./bm25_cache.jsonl

Notes:
- embedding_dim defaults to 384 (all-MiniLM-L6-v2). If you switch models, set --embedding-dim accordingly.
- By default we DO NOT recreate the Qdrant index (safe). Pass --recreate-index to wipe it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# -----------------------------
# Haystack imports (support v2 first, fall back to v1-ish)
# -----------------------------
try:  # Haystack 2.x
    from haystack.dataclasses import Document
except Exception as e:
    print('haystack 2.x not found')  # pragma: no cover
    try:
        from haystack import Document  # type: ignore
    except Exception as e:
        raise ImportError("Cannot import Haystack Document. Please install haystack>=2.0.") from e

try:  # Haystack 2.x with qdrant-haystack integration
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
    from haystack.document_stores.in_memory import InMemoryDocumentStore
except Exception:
    print('haystack 2.x with qdrant-haystack integration not found')  # pragma: no cover
    try:
        from haystack.document_stores import QdrantDocumentStore, InMemoryDocumentStore  # type: ignore
    except Exception as e:
        raise ImportError("Cannot import Haystack document stores. Install haystack-ai and qdrant-haystack.") from e

# Embedder
_embedder_fallback = False
try:
    from haystack.components.embedders import SentenceTransformersDocumentEmbedder
except Exception:  # pragma: no cover
    # Fallback: use sentence-transformers directly if components API isn't available
    print('SentenceTransformersDocumentEmbedder not found')
    _embedder_fallback = True
    from sentence_transformers import SentenceTransformer  # type: ignore

    class SentenceTransformersDocumentEmbedder:  # type: ignore
        def __init__(self, model: str) -> None:
            self._model = SentenceTransformer(model)

        def warm_up(self) -> None:
            _ = self._model.encode(["warmup"], show_progress_bar=False)

        def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
            texts = [(d.content or "") for d in documents]
            embs = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            for d, e in zip(documents, embs):
                # Haystack expects list[float]
                d.embedding = e.tolist()
            return {"documents": documents}

# Duplicate policy
try:
    from haystack.document_stores.types import DuplicatePolicy
except Exception:  # pragma: no cover
    try:
        from haystack.document_stores.base import DuplicatePolicy  # type: ignore
    except Exception:
        class DuplicatePolicy:  # very small shim
            SKIP = "skip"


# =============================
# Markdown → Documents (new)
# =============================

MD_SECTION_RE = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")

def _split_markdown_sections(text: str) -> List[tuple[int, str, str]]:
    """Return list of (level, heading, body) for MD sections.
    Includes a pseudo-section for any preface text before the first heading.
    """
    parts = []
    # find heading spans
    matches = list(MD_SECTION_RE.finditer(text))
    if not matches:
        body = text.strip()
        if body:
            parts.append((0, "preface", body))
        return parts

    # preface
    pre_start = 0
    first = matches[0]
    preface = text[pre_start:first.start()].strip()
    if preface:
        parts.append((0, "preface", preface))

    # each section
    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            parts.append((level, heading, body))
    return parts


def load_md_as_documents(md_path: Path, source_name: Optional[str] = None) -> List[Document]:
    """Convert a Markdown file into section-chunked Documents.
    - Uses MD headings (# .. ######) as chunk boundaries.
    - Stores 'section' and 'level' in meta for filtering.
    """
    src = source_name or md_path.name
    text = md_path.read_text(encoding="utf-8")
    sections = _split_markdown_sections(text)
    docs: List[Document] = []
    for level, heading, body in sections:
        docs.append(
            Document(
                content=body,
                meta={
                    "source": src,
                    "section": heading,
                    "level": level,
                    "format": "markdown",
                },
            )
        )
    return docs


def discover_markdown_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*.md") if p.is_file())


# -----------------------------
# Marker JSON → Documents (existing)
# -----------------------------

def load_marker_as_documents(
    blocks_json: Path,
    content_json: Path,
    meta_json: Optional[Path] = None,
    source_name: str = "source.pdf",
) -> List[Document]:
    """Minimal loader for Marker JSONs.
    Prefers block-level HTML; falls back to flattened content HTML; last resort is raw content JSON.
    """
    docs: List[Document] = []

    # Load optional meta
    meta: Dict = {}
    if meta_json and meta_json.exists():
        try:
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
            logging.info("marker meta loaded from %s", meta_json)
        except Exception:
            meta = {}
            logging.warning("marker meta could not be parsed: %s", meta_json)

    # Load primary blocks.json
    blocks = json.loads(blocks_json.read_text(encoding="utf-8"))

    def collect_blocks(node) -> None:
        # node can be dict with 'block_type', 'html', 'children'
        if isinstance(node, dict):
            btype = node.get("block_type", "")
            html = node.get("html")
            if isinstance(html, str) and html.strip():
                docs.append(
                    Document(
                        content=html,
                        meta={
                            "source": source_name,
                            "block_type": btype,
                            "format": "marker",
                            **({"marker_meta": meta} if meta else {}),
                        },
                    )
                )
            for child in (node.get("children", []) or []):
                collect_blocks(child)
        elif isinstance(node, list):
            for child in node:
                collect_blocks(child)

    collect_blocks(blocks)

    # Fallback 1: content JSON flatten
    if not docs and content_json.exists():
        logging.info("blocks had no html; flattening content json: %s", content_json)
        content_obj = json.loads(content_json.read_text(encoding="utf-8"))

        def flatten_html(x):
            if isinstance(x, dict):
                if isinstance(x.get("html"), str):
                    return [x["html"]]
                out = []
                for v in x.values():
                    out.extend(flatten_html(v))
                return out
            if isinstance(x, list):
                out = []
                for v in x:
                    out.extend(flatten_html(v))
                return out
            return []

        html_chunks = flatten_html(content_obj)
        for h in html_chunks:
            if h and h.strip():
                docs.append(
                    Document(
                        content=h,
                        meta={"source": source_name, "format": "marker", **({"marker_meta": meta} if meta else {})},
                    )
                )

    # Fallback 2: raw content
    if not docs and content_json.exists():
        logging.warning("no html found anywhere; emitting raw content json: %s", content_json)
        docs.append(
            Document(
                content=content_json.read_text(encoding="utf-8"),
                meta={
                    "source": source_name,
                    "raw_marker_content": True,
                    "format": "marker",
                    **({"marker_meta": meta} if meta else {}),
                },
            )
        )

    return docs


# -----------------------------
# Multi-file discovery & loaders (Marker)
# -----------------------------

@dataclass
class MarkerTriplet:
    blocks: Path
    content: Optional[Path]
    meta: Optional[Path]
    source_name: str

def discover_marker_triplets(root: Path) -> List[MarkerTriplet]:
    """Recursively find Marker exports under 'root'.
    Heuristic: each paper lives in a directory containing 'blocks.json' + other jsons.
    The first non-meta json in that directory is considered the 'content' json.
    """
    triplets: List[MarkerTriplet] = []
    for blocks in root.rglob("blocks.json"):
        base_dir = blocks.parent
        json_candidates = sorted(p for p in base_dir.glob("*.json") if p.name != "blocks.json")
        meta = next((p for p in json_candidates if "meta" in p.stem.lower()), None)
        content = next((p for p in json_candidates if p != meta), None)
        # Use folder name as a readable source tag
        source_name = base_dir.name + ".pdf"
        triplets.append(MarkerTriplet(blocks=blocks, content=content, meta=meta, source_name=source_name))
    return triplets


def load_many_markers(root: Path) -> List[Document]:
    all_docs: List[Document] = []
    for t in discover_marker_triplets(root):
        docs = load_marker_as_documents(
            blocks_json=t.blocks,
            content_json=t.content if t.content else Path(""),
            meta_json=t.meta,
            source_name=t.source_name,
        )
        all_docs.extend(docs)
    return all_docs


def iter_many_markers(root: Path) -> Iterable[Document]:
    for t in discover_marker_triplets(root):
        for d in load_marker_as_documents(
            blocks_json=t.blocks,
            content_json=t.content if t.content else Path(""),
            meta_json=t.meta,
            source_name=t.source_name,
        ):
            yield d


# -----------------------------
# Stable IDs, persistence helpers, and indexing
# -----------------------------

def _stable_id(text: str, source: str = "") -> str:
    h = hashlib.sha1()
    h.update(source.encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    h.update(text.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def assign_ids(documents: List[Document]) -> List[Document]:
    for d in documents:
        src = (d.meta or {}).get("source", "")
        d.id = _stable_id(d.content or "", src)
    return documents


def save_documents_jsonl(path: Path, docs: List[Document]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            rec = {"id": d.id, "content": d.content, "meta": d.meta}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_documents_jsonl(path: Path) -> List[Document]:
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


# -----------------------------
# Stores
# -----------------------------

def build_stores(
    qdrant_collection: str = "papers",
    persist_path: Optional[str] = "./qdrant_papers",
    recreate_index: bool = False,
    embedding_dim: int = 384,
):
    """Create (or attach to) a Qdrant vector store + an in-memory BM25 store.
    - If persist_path is provided, use embedded Qdrant on disk.
    - Otherwise, use in-memory Qdrant (ephemeral).
    - Do NOT recreate index by default (keeps data).
    """
    qdrant_kwargs = {
        "index": qdrant_collection,
        "recreate_index": recreate_index,
        "embedding_dim": embedding_dim,
    }
    if persist_path:
        # Embedded/Local Qdrant DB on disk
        qdrant_kwargs.update({"path": persist_path, "on_disk": True})
    else:
        # Ephemeral in-memory instance
        qdrant_kwargs.update({"location": ":memory:"})

    qdrant_store = QdrantDocumentStore(**qdrant_kwargs)

    # Sparse store for BM25 (in-memory; repopulate from cache on startup if desired)
    sparse_store = InMemoryDocumentStore()

    return qdrant_store, sparse_store


def index_documents(
    qdrant_store: QdrantDocumentStore,
    sparse_store: InMemoryDocumentStore,
    documents: List[Document],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 128,
):
    # Ensure stable ids for idempotent ingestion
    documents = assign_ids(documents)

    # Write raw docs to sparse store (BM25). Skip duplicates on re-runs.
    sparse_store.write_documents(documents, policy=DuplicatePolicy.SKIP)

    # Embed + write to Qdrant in batches
    embedder = SentenceTransformersDocumentEmbedder(model=embedding_model)
    embedder.warm_up()

    for i in range(0, len(documents), batch_size):
        chunk = documents[i : i + batch_size]
        result = embedder.run(documents=chunk)
        embedded_docs = result["documents"]
        qdrant_store.write_documents(embedded_docs, policy=DuplicatePolicy.SKIP)


def index_streaming(
    qdrant_store: QdrantDocumentStore,
    sparse_store: InMemoryDocumentStore,
    doc_iter: Iterable[Document],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 128,
):
    buffer: List[Document] = []
    embedder = SentenceTransformersDocumentEmbedder(model=embedding_model)
    embedder.warm_up()

    def _flush(buf: List[Document]) -> None:
        if not buf:
            return
        buf = assign_ids(buf)
        sparse_store.write_documents(buf, policy=DuplicatePolicy.SKIP)
        embedded = embedder.run(documents=buf)["documents"]
        qdrant_store.write_documents(embedded, policy=DuplicatePolicy.SKIP)

    for d in doc_iter:
        buffer.append(d)
        if len(buffer) >= batch_size:
            _flush(buffer)
            buffer.clear()
    _flush(buffer)


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest Marker exports and/or Markdown into Qdrant (dense) + BM25 (sparse)")
    p.add_argument("--marker-root", type=Path, help="Root folder containing many papers with blocks.json")
    p.add_argument("--md-root", type=Path, help="Root folder containing Markdown .md files")
    p.add_argument("--qdrant-collection", type=str, default="papers")
    p.add_argument("--qdrant-persist", type=str, default="./qdrant_papers", help="Folder for embedded Qdrant persistence (set empty for in-memory)")
    p.add_argument("--recreate-index", action="store_true", help="Drop & recreate the Qdrant collection before indexing")
    p.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--embedding-dim", type=int, default=384, help="Vector size; must match embedding model output")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--bm25-cache", type=Path, default=Path("./bm25_cache.jsonl"), help="Optional JSONL cache of docs for BM25 warm start")
    p.add_argument("--streaming", action="store_true", help="Stream ingestion to save memory")
    p.add_argument("--loglevel", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    if not args.marker_root and not args.md_root:
        raise SystemExit("Provide at least one of --marker-root or --md-root")

    # Build or attach to stores
    qdrant_store, sparse_store = build_stores(
        qdrant_collection=args.qdrant_collection,
        persist_path=args.qdrant_persist if args.qdrant_persist else None,
        recreate_index=args.recreate_index,
        embedding_dim=args.embedding_dim,
    )

    # BM25 warm start (optional): load cached docs (ids+content+meta) into sparse store
    cached_docs = load_documents_jsonl(args.bm25_cache)
    if cached_docs:
        logging.info("Loaded %d cached docs for BM25 warm start", len(cached_docs))
        sparse_store.write_documents(assign_ids(cached_docs), policy=DuplicatePolicy.SKIP)

    documents: List[Document] = []

    # Ingest Marker
    if args.marker_root:
        root = args.marker_root
        assert root.exists(), f"Marker root does not exist: {root}"
        logging.info("Loading Marker exports from %s", root)
        documents.extend(load_many_markers(root))

    # Ingest Markdown
    if args.md_root:
        mdroot = args.md_root
        assert mdroot.exists(), f"Markdown root does not exist: {mdroot}"
        md_files = discover_markdown_files(mdroot)
        logging.info("Discovered %d Markdown files under %s", len(md_files), mdroot)
        for md in md_files:
            documents.extend(load_md_as_documents(md, source_name=md.name))

    logging.info("Total documents loaded: %d", len(documents))

    if args.streaming:
        logging.info("Streaming ingestion is only used with generator sources; falling back to batch for mixed inputs.")

    index_documents(
        qdrant_store=qdrant_store,
        sparse_store=sparse_store,
        documents=documents,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
    )

    # Update cache for future BM25 warm starts
    try:
        save_documents_jsonl(args.bm25_cache, assign_ids(documents))
        logging.info("Saved BM25 cache to %s", args.bm25_cache)
    except Exception as e:
        logging.warning("Could not save BM25 cache: %s", e)

    logging.info("Ingestion complete.")


if __name__ == "__main__":
    main()
