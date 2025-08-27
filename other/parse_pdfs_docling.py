#!/usr/bin/env python3
# parse_pdfs_docling.py
# Convert PDFs to structured outputs with Docling, export images, and create RAG-ready chunks.

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Iterable, List, Union

# --- Docling imports (high-level API) ---
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
)

# --- Docling chunking (for RAG) ---
from docling.chunking import HybridChunker

# --- Types used when exporting images/figures/tables ---
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem


def gather_inputs(path_or_url: str) -> List[Union[str, Path]]:
    """
    f --input is a directory, it recursively collects all *.pdf files. 
    If it's a single file, it returns that. If the path doesn't exist, 
    it treats the string as a URL 
    """
    p = Path(path_or_url)
    if p.exists():
        if p.is_dir():
            return sorted([q for q in p.rglob("*.pdf")])
        return [p]
    # Not a local file—treat as URL
    return [path_or_url]


def build_converter(
    artifacts_path: str = None,
    enable_remote: bool = False,
    export_images: bool = False,
    #benchmark on your corpus; “accurate” isn’t universally better if your tables are simple—fast 
    # may be statistically indistinguishable yet 2–5× cheaper.
    table_mode: str = "fast",
    # if the source PDF has noisy text boxes (common in scanned+OCR’d docs), this step can
    # occasionally introduce errors. For scans, consider disabling or pairing with high-quality OCR
    match_pdf_cells: bool = True,
) -> DocumentConverter:
    """
    Configure Docling's PDF pipeline.
    """
    pipeline_options = PdfPipelineOptions(
        artifacts_path=artifacts_path,               # use local cached models if provided
        enable_remote_services=enable_remote,        # must be True to call any remote services
    )

    # Control table structure recognition behavior & model mode (fast vs accurate)
    pipeline_options.do_table_structure = True

    pipeline_options.table_structure_options.mode = (
        TableFormerMode.ACCURATE if table_mode == "accurate" else TableFormerMode.FAST
    )
    # Whether to match recognized structure back to PDF cells (can help merged columns issues)
    pipeline_options.table_structure_options.do_cell_matching = match_pdf_cells

    if export_images:
        # IMPORTANT: to keep page/figure/table images, we must enable & set a scale.
        # scale=1 corresponds to ~72 DPI; 2.0 gives nicer figures with modest memory.
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True  # figures/tables images

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def export_conv_result(
    conv_res: ConversionResult,
    out_dir: Path,
    export_images: bool = False,
) -> None:
    """
    Save multiple serializations (JSON/MD/HTML/TXT) and optionally images.
    """
    doc = conv_res.document
    stem = conv_res.input.file.stem if conv_res.input.file else "document"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Modern v2 exports (rich Docling document) ---
    doc.save_as_json(out_dir / f"{stem}.json", image_mode=ImageRefMode.PLACEHOLDER)
    doc.save_as_markdown(out_dir / f"{stem}.md", image_mode=ImageRefMode.PLACEHOLDER)
    doc.save_as_markdown(out_dir / f"{stem}.txt", image_mode=ImageRefMode.PLACEHOLDER, strict_text=True)
    doc.save_as_html(out_dir / f"{stem}.html", image_mode=ImageRefMode.REFERENCED)

    # Also dump the raw dict (handy for debugging)
    with (out_dir / f"{stem}.yaml").open("w", encoding="utf-8") as fp:
        # export_to_dict is stable and easier to diff than JSON sometimes
        import yaml  # lazy import to avoid dependency if not needed
        fp.write(yaml.safe_dump(doc.export_to_dict()))

    # --- Optional image exports ---
    if export_images:
        # Page images
        for page_no, page in doc.pages.items():
            if page.image and page.image.pil_image:
                page_img = out_dir / f"{stem}-page-{page_no}.png"
                page.image.pil_image.save(page_img, format="PNG")

        # Figure & table crops
        tbl_count = 0
        pic_count = 0
        for element, _lvl in doc.iterate_items():
            if isinstance(element, TableItem):
                tbl_count += 1
                element.get_image(doc).save(out_dir / f"{stem}-table-{tbl_count}.png", "PNG")
            if isinstance(element, PictureItem):
                pic_count += 1
                element.get_image(doc).save(out_dir / f"{stem}-figure-{pic_count}.png", "PNG")


def chunk_and_write_jsonl(
    conv_res: ConversionResult,
    out_dir: Path,
    tokenizer: str = "BAAI/bge-small-en-v1.5",
    max_chunks: int = None,
) -> int:
    """
    Produce JSONL chunks (text + metadata) suitable for vector DB ingestion (e.g., Chroma/Qdrant).
    Defaults to a BGE tokenizer so chunk sizes align with BGE reranker/embeddings.
    """
    doc = conv_res.document
    chunker = HybridChunker(tokenizer=tokenizer)  # hybrid page/semantic chunker
    chunks_iter = chunker.chunk(doc)

    # Output file (append mode so we can run per-doc)
    out_path = out_dir / "chunks.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("a", encoding="utf-8") as fp:
        for n, ch in enumerate(chunks_iter, start=1):
            record = {
                "text": ch["text"],
                "meta": ch.get("meta", {}),
                # Useful top-level metadata for filtering/grouping:
                "source": str(conv_res.input.file) if conv_res.input.file else conv_res.input.url or "unknown",
                "doc_stem": conv_res.input.file.stem if conv_res.input.file else "document",
            }
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            if max_chunks and n >= max_chunks:
                break
    return n


def convert_all(
    inputs: Iterable[Union[str, Path]],
    converter: DocumentConverter,
    out_dir: Path,
    export_images: bool,
    chunks: bool,
    tokenizer: str,
    max_pages: int,
    max_file_size: int,
    raises_on_error: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Batch convert (lets you keep going even if one file fails)
    conv_results = converter.convert_all(
        list(inputs),
        raises_on_error=raises_on_error,
        max_num_pages=max_pages if max_pages else None,
        max_file_size=max_file_size if max_file_size else None,
    )

    ok = 0
    partial = 0
    fail = 0

    for res in conv_results:
        if res.status == ConversionStatus.SUCCESS:
            ok += 1
            export_conv_result(res, out_dir, export_images=export_images)
            if chunks:
                n = chunk_and_write_jsonl(res, out_dir / "chunks", tokenizer=tokenizer)
                logging.info(f"[chunks] wrote {n} chunks for {res.input.file}")
        elif res.status == ConversionStatus.PARTIAL_SUCCESS:
            partial += 1
            logging.warning(f"[partial] {res.input.file} had errors: {[e.error_message for e in res.errors]}")
            export_conv_result(res, out_dir, export_images=export_images)
            if chunks:
                n = chunk_and_write_jsonl(res, out_dir / "chunks", tokenizer=tokenizer)
                logging.info(f"[chunks] wrote {n} chunks for {res.input.file}")
        else:
            fail += 1
            logging.error(f"[failed] {res.input.file} failed to convert.")

    logging.info(f"[done] success={ok} partial={partial} failed={fail}")


def main():
    ap = argparse.ArgumentParser(description="Parse PDFs with Docling and emit structured outputs.")
    ap.add_argument("--input", required=True, help="PDF file, folder, or URL")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--export-images", action="store_true", help="Export page/figure/table PNGs")
    ap.add_argument("--chunks", action="store_true", help="Emit JSONL chunks for vector DB ingestion")
    ap.add_argument("--tokenizer", default="BAAI/bge-small-en-v1.5", help="Tokenizer for HybridChunker")
    ap.add_argument("--artifacts-path", default=os.environ.get("DOCLING_ARTIFACTS_PATH"), help="Local model cache dir")
    ap.add_argument("--enable-remote", action="store_true", help="Allow remote services (requires explicit opt-in)")
    ap.add_argument("--table-mode", choices=["fast", "accurate"], default="accurate", help="TableFormer mode")
    ap.add_argument("--match-pdf-cells", action="store_true", help="Match table structure back to PDF text cells")
    ap.add_argument("--max-pages", type=int, default=20, help="Max pages per document")
    ap.add_argument("--max-file-size", type=int, default=52428800, help="Max file size (bytes) per document")
    ap.add_argument("--loglevel", default="INFO", help="Logging level")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    inputs = gather_inputs(args.input)
    if not inputs:
        raise SystemExit("No inputs found.")

    converter = build_converter(
        artifacts_path=args.artifacts_path,
        enable_remote=args.enable_remote,
        export_images=args.export_images,
        table_mode=args.table_mode,
        match_pdf_cells=args.match_pdf_cells,
    )

    convert_all(
        inputs=inputs,
        converter=converter,
        out_dir=Path(args.out),
        export_images=args.export_images,
        chunks=args.chunks,
        tokenizer=args.tokenizer,
        max_pages=args.max_pages,
        max_file_size=args.max_file_size,
    )


if __name__ == "__main__":
    main()
