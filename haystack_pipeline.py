"""
Hybrid RAG with Haystack (store -> retrieve -> rerank)
- Vector store: QdrantDocumentStore (dense retrieval)
- Sparse store: InMemoryDocumentStore + InMemoryBM25Retriever (BM25)
- Fusion: simple union + score normalization
- Rerank: TransformersSimilarityRanker (e.g., BAAI/bge-reranker-base)

Install (Python 3.10+ recommended):
    pip install "haystack-ai>=2.1.0" qdrant-client qdrant-haystack
    pip install "sentence-transformers>=2.7.0" torch  # CPU ok
    # (optional) pip install unstructured-fileconverter-haystack  # if ingesting raw files directly via Unstructured

Refs:
- Hybrid retrieval tutorial & reranker: https://haystack.deepset.ai/tutorials/33_hybrid_retrieval
- TransformersSimilarityRanker: https://docs.haystack.deepset.ai/docs/transformerssimilarityranker
- QdrantDocumentStore: https://docs.haystack.deepset.ai/docs/qdrant-document-store
"""

from pathlib import Path
import json
from typing import List, Dict

from haystack import Document, Pipeline

# Dense side (Qdrant + embedding)
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder

# Sparse side (BM25)
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryBM25Retriever

# Reranker
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.document_stores.types import DuplicatePolicy


# ---------- 1) Load Marker outputs -> Haystack Documents ----------
def load_marker_as_documents(
    blocks_json: Path,
    content_json: Path,
    meta_json: Path | None = None,
    source_name: str = "murray_lyapunov.pdf"
) -> List[Document]:
    """
    Minimal loader for Marker JSONs:
      - blocks.json: layout blocks with text/html per block
      - content json: linearized text/html per page or grouped
      - meta json: optional metadata from Marker
    We create one Document per block that has text/html.
    """
    docs: List[Document] = []

    # Load meta (if present)
    meta: Dict = {}
    if meta_json and meta_json.exists():
        try:
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    # Prefer the layout 'blocks.json' to keep section/figure context;
    # fall back to content JSON if blocks are missing.
    with blocks_json.open("r", encoding="utf-8") as f:
        blocks = json.load(f)

    # Marker stores page objects that contain child blocks; we walk them.
    # Each child has "block_type" (e.g., Text, TextInlineMath, Caption, Equation) and possibly "html".
    def collect_blocks(node) -> None:
        # node can be dict with 'block_type', 'html', 'children'
        if isinstance(node, dict):
            btype = node.get("block_type", "")
            html = node.get("html")
            # Use html if present; otherwise skip (Marker sometimes omits plain text in this export)
            if html and isinstance(html, str) and html.strip():
                docs.append(
                    Document(
                        content=html,
                        meta={
                            "source": source_name,
                            "block_type": btype,
                            **({"marker_meta": meta} if meta else {}),
                        }
                    )
                )
            # Recurse into children
            for child in node.get("children", []) or []:
                collect_blocks(child)
        elif isinstance(node, list):
            for child in node:
                collect_blocks(child)

    collect_blocks(blocks)

    # If no blocks found, try content_json as a single big doc
    if not docs and content_json.exists():
        content_obj = json.loads(content_json.read_text(encoding="utf-8"))
        # Some Marker exports store HTML under content_obj["children"][...]["html"]
        def flatten_html(x):
            if isinstance(x, dict):
                if "html" in x and isinstance(x["html"], str):
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
            if h.strip():
                docs.append(Document(content=h, meta={"source": source_name, **({"marker_meta": meta} if meta else {})}))

    # As a final fallback, stitch the entire content JSON
    if not docs and content_json.exists():
        docs.append(
            Document(
                content=content_json.read_text(encoding="utf-8"),
                meta={"source": source_name, "raw_marker_content": True, **({"marker_meta": meta} if meta else {})}
            )
        )

    return docs


# ---------- 2) Build stores (Qdrant for vectors, InMemory for BM25) ----------
def build_stores(qdrant_collection: str = "papers", persist_path: str | None = None):
    # Vector store (Qdrant)
    qdrant_kwargs = {
        "index": qdrant_collection,
        "recreate_index": True,
        "embedding_dim": 384,  # matches default SentenceTransformers model below (all-MiniLM-L6-v2)
    }
    # Use embedded Qdrant by default to avoid needing an external server
    if persist_path:
        qdrant_kwargs.update({"path": persist_path, "on_disk": True})
    else:
        qdrant_kwargs.update({"location": ":memory:"})

    qdrant_store = QdrantDocumentStore(**qdrant_kwargs)

    # Sparse store (BM25)
    sparse_store = InMemoryDocumentStore()  # keeps raw text for BM25 keyword retrieval

    return qdrant_store, sparse_store


# ---------- 3) Index pipeline ----------
def index_documents(qdrant_store, sparse_store, documents: List[Document], embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    # Write raw docs to sparse store
    sparse_store.write_documents(documents, policy=DuplicatePolicy.OVERWRITE)

    # Embed documents for the vector store
    doc_embedder = SentenceTransformersDocumentEmbedder(model=embedding_model)
    doc_embedder.warm_up()
    # Important: run() on Document lists returns {"documents": [...]}
    embedded = doc_embedder.run(documents=documents)["documents"]
    qdrant_store.write_documents(embedded, policy=DuplicatePolicy.OVERWRITE)  # upsert with embeddings


# ---------- 4) Query (hybrid) + rerank ----------
def build_query_pipeline(qdrant_store, sparse_store,
                         retriever_top_k: int = 30,
                         ranker_top_k: int = 10,
                         text_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                         reranker_model: str = "BAAI/bge-reranker-base"):
    """
    We compose a small pipeline programmatically:
      - text embedder for the query (for dense retriever)
      - QdrantEmbeddingRetriever for dense
      - InMemoryBM25Retriever for sparse
      - simple fusion (concatenate + normalize)
      - TransformersSimilarityRanker for cross-encoder rerank
    """

    # Components
    query_embedder = SentenceTransformersTextEmbedder(model=text_embed_model)
    query_embedder.warm_up()
    dense_retriever = QdrantEmbeddingRetriever(document_store=qdrant_store, top_k=retriever_top_k)
    sparse_retriever = InMemoryBM25Retriever(document_store=sparse_store, top_k=retriever_top_k)
    reranker = TransformersSimilarityRanker(model=reranker_model, top_k=ranker_top_k)
    reranker.warm_up()

    # Wire the DAG
    pipe = Pipeline()

    # Nodes
    pipe.add_component("query_embedder", query_embedder)
    pipe.add_component("dense", dense_retriever)
    pipe.add_component("sparse", sparse_retriever)
    pipe.add_component("rerank", reranker)

    # Edges
    # Dense branch needs the query embedding
    pipe.connect("query_embedder.embedding", "dense.query_embedding")
    # Both sparse and dense also need the raw query string
    pipe.connect("query_embedder", "sparse")  # passes 'query' along implicitly

    # After we have two candidate lists, we fuse by concatenating then rerank by the cross-encoder
    # Haystack pipelines allow passing multiple inputs into a component by naming arguments.
    def fuse_and_rerank(pipeline: Pipeline, query: str):
        # 1) run embedder -> dense + sparse
        out = pipeline.run(
            data={"query_embedder": {"text": query},
                  "sparse": {"query": query}},
            include_outputs_from=["dense", "sparse"]
        )
        dense_docs = out["dense"]["documents"]
        sparse_docs = out["sparse"]["documents"]

        # 2) naive fusion (union by id while preserving highest score per doc);
        #    normalize scores to [0,1] per list to balance sparse/dense.
        def normalize(ds):
            if not ds:
                return ds
            scores = [d.score for d in ds if d.score is not None]
            if not scores:
                return ds
            mn, mx = min(scores), max(scores)
            rng = max(mx - mn, 1e-6)
            for d in ds:
                if d.score is not None:
                    d.score = (d.score - mn) / rng
            return ds

        dense_docs = normalize(dense_docs)
        sparse_docs = normalize(sparse_docs)
        pool: Dict[str, Document] = {}
        for d in dense_docs + sparse_docs:
            if d.id not in pool or (d.score or 0.0) > (pool[d.id].score or 0.0):
                pool[d.id] = d
        fused_docs = list(pool.values())

        # 3) rerank with cross-encoder
        ranked = pipeline.get_component("rerank").run(query=query, documents=fused_docs)["documents"]
        return ranked

    # attach helper
    pipe.fuse_and_rerank = fuse_and_rerank  # type: ignore[attr-defined]
    return pipe


# ---------- 5) Example main ----------
if __name__ == "__main__":
    # Paths to your Marker outputs
    blocks = Path("rag_seed/marker/murray_lyapunov/blocks.json")
    content = Path("rag_seed/marker/murray_lyapunov/murray_lyapunov.json")
    meta = Path("rag_seed/marker/murray_lyapunov/murray_lyapunov_meta.json")

    documents = load_marker_as_documents(blocks, content, meta, source_name="murray_lyapunov.pdf")
    assert documents, "No documents were created from Marker JSONs."

    qdrant_store, sparse_store = build_stores(qdrant_collection="rl_control_papers")

    # Index
    index_documents(qdrant_store, sparse_store, documents)

    # Query pipeline
    pipe = build_query_pipeline(qdrant_store, sparse_store)

    # Try a query
    user_query = "Lyapunov candidate function for pendulum stability with energy shaping"
    final_docs = pipe.fuse_and_rerank(pipe, user_query)

    print(f"\nTop results for: {user_query}\n" + "-" * 60)
    for i, d in enumerate(final_docs[:5], 1):
        print(f"[{i}] score={d.score:.3f}  id={d.id}")
        print(f"meta: { {k:v for k,v in d.meta.items() if k!='marker_meta'} }")
        print(f"content (truncated): {d.content[:280].replace('\\n',' ') }")
        print("-" * 60)
