#!/usr/bin/env python3
"""
rag_pdf_math_recover.py

Pipeline:
1) Run Marker (PDF -> Markdown/JSON) for general text.
2) For each page, check if it has any math markers; mark pages "missing math".
3) Render those pages; detect formula regions and recognize LaTeX with Pix2Text.
   Optionally crop each region and run pix2tex (LatexOCR) as a fallback.
4) Emit RAG-friendly JSONL chunks: per-page text (+ recovered $$...$$), with metadata.

Requirements:
  pip install marker-pdf pix2text pypdfium2 pdfminer.six latex-ocr tqdm

Notes:
- We don't rely on Marker to give per-page splits (schemas vary). We extract per-page text with pdfminer.
- For scanned PDFs, pdfminer may return empty text; formula OCR still works and you'll at least capture math.
- Formula detection/recognition is via Pix2Text (P2T). You can enable pix2tex fallback with --use-pix2tex.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from tqdm import tqdm

# PDF rendering (for images used by Pix2Text / pix2tex)
import pypdfium2 as pdfium

# Vector text extraction (fast for native PDFs; empty for scanned ones)
from pdfminer.high_level import extract_text

# Pix2Text (detect + recognize formulas; returns latex + boxes)
try:
    from pix2text import Pix2Text  # type: ignore
    HAVE_P2T = True
except Exception:
    HAVE_P2T = False

# pix2tex fallback
try:
    from pix2tex.cli import LatexOCR  # type: ignore
    HAVE_P2TEX = True
except Exception:
    HAVE_P2TEX = False

# ------------------------ helpers ------------------------

MATH_PATTERNS = [
    r"\$\$[\s\S]+?\$\$",            # $$ ... $$
    r"\\\[[\s\S]+?\\\]",            # \[ ... \]
    r"\\begin\{equation\}[\s\S]+?\\end\{equation\}",
    r"\\begin\{align\*?\}[\s\S]+?\\end\{align\*?\}",
    r"\\\([\s\S]+?\\\)",            # \( ... \)
    r"(?<!\\)\$[^$\n]{1,256}(?<!\\)\$",  # inline $...$ (not greedy, skip escaped \$)
]

MATH_RE = re.compile("|".join(MATH_PATTERNS), flags=re.MULTILINE)

def has_math_markers(text: str) -> bool:
    return bool(MATH_RE.search(text or ""))

def run(cmd: str, cwd: Optional[Path] = None) -> None:
    print(f"[cmd] {cmd}")
    res = subprocess.run(shlex.split(cmd), cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed: {cmd}")

def soft_run(cmd: str, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    print(f"[cmd] {cmd}")
    return subprocess.run(shlex.split(cmd), cwd=cwd, capture_output=True, text=True)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def render_page_as_pil(pdf: pdfium.PdfDocument, page_index: int, scale: float = 2.0):
    page = pdf[page_index]
    bitmap = page.render(scale=scale)
    return bitmap.to_pil()

def split_into_chunks(text: str, max_chars: int = 1800) -> List[str]:
    # paragraph-first split, then greedy packing
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                chunks.append(buf)
            if len(p) <= max_chars:
                buf = p
            else:
                # hard wrap long paragraph
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i+max_chars])
                buf = ""
    if buf:
        chunks.append(buf)
    return chunks

@dataclass
class Formula:
    page: int
    bbox: List[float]  # [x0, y0, x1, y1] in image coords
    latex: str
    source: str  # 'pix2text' or 'pix2tex'
    score: Optional[float] = None
    crop_path: Optional[str] = None

# ------------------------ core pipeline ------------------------

def run_marker(pdf_path: Path, out_dir: Path, force_ocr: bool) -> Dict[str, Any]:
    """Run Marker CLI. We save markdown and (if available) json.
    Return dict with discovered file paths."""
    marker_dir = out_dir / "marker"
    ensure_dir(marker_dir)

    # Prefer markdown output for readability; CLI is consistent
    md_out = marker_dir / (pdf_path.stem + ".md")
    json_out = marker_dir / (pdf_path.stem + ".json")

    # Try JSON first (schema may vary by version), then always MD.
    # JSON:
    r = soft_run(
        f"marker_single {shlex.quote(str(pdf_path))} "
        f"--output_format json --output_dir {shlex.quote(str(marker_dir))} "
        + ("--force_ocr" if force_ocr else "")
    )
    if r.returncode != 0:
        print("[warn] marker json failed, continuing with markdown only")
    # Markdown:
    run(
        f"marker_single {shlex.quote(str(pdf_path))} "
        f"--output_format markdown --output_dir {shlex.quote(str(marker_dir))} "
        + ("--force_ocr" if force_ocr else "")
    )

    files = {"marker_md": str(md_out)}
    if json_out.exists():
        files["marker_json"] = str(json_out)
    return files

def extract_text_per_page_with_pdfminer(pdf_path: Path) -> List[str]:
    # pdfminer page_numbers is 0-based
    # We need number of pages:
    doc = pdfium.PdfDocument(str(pdf_path))
    n_pages = len(doc)
    texts = []
    for i in range(n_pages):
        try:
            t = extract_text(str(pdf_path), page_numbers=[i]) or ""
        except Exception:
            t = ""
        texts.append(t)
    return texts

def detect_and_ocr_formulas(
    pdf_path: Path,
    pages_to_process: List[int],
    out_dir: Path,
    render_scale: float = 2.0,
    use_pix2tex: bool = False,
) -> List[Formula]:
    if not HAVE_P2T:
        raise RuntimeError("pix2text is not installed. `pip install pix2text`")

    ensure_dir(out_dir / "crops")
    pdf = pdfium.PdfDocument(str(pdf_path))
    p2t = Pix2Text()  # default config works well; customize if you like

    ocr = None
    if use_pix2tex:
        if not HAVE_P2TEX:
            print("[warn] --use-pix2tex passed but latex-ocr not installed; skipping fallback")
        else:
            ocr = LatexOCR()

    formulas: List[Formula] = []
    for pidx in tqdm(pages_to_process, desc="Formula OCR pages"):
        img = render_page_as_pil(pdf, pidx, scale=render_scale)
        # Pix2Text returns a list of blocks; we keep only formula types
        try:
            res = p2t.recognize(img)  # returns blocks with 'type', 'position', 'text', 'score'
        except Exception as e:
            print(f"[warn] Pix2Text failed on page {pidx+1}: {e}")
            res = []

        for obj in res if isinstance(res, list) else []:
            ttype = obj.get("type")
            if ttype not in ("latex", "formula", "equation", "latex_formula"):
                continue
            text = (obj.get("text") or "").strip()
            score = obj.get("score") if isinstance(obj.get("score"), (int, float)) else None
            # position may be dict or list
            bbox = None
            pos = obj.get("position") or obj.get("box") or obj.get("bbox")
            if isinstance(pos, dict):
                # normalize to [x0,y0,x1,y1]
                x0 = pos.get("x0") or pos.get("left") or 0
                y0 = pos.get("y0") or pos.get("top") or 0
                x1 = pos.get("x1") or pos.get("right") or 0
                y1 = pos.get("y1") or pos.get("bottom") or 0
                bbox = [float(x0), float(y0), float(x1), float(y1)]
            elif isinstance(pos, (list, tuple)) and len(pos) == 4:
                bbox = [float(v) for v in pos]
            else:
                bbox = [0.0, 0.0, img.width, img.height]

            crop_path = None
            if bbox:
                left, top, right, bottom = [int(round(v)) for v in bbox]
                left = max(0, min(left, img.width))
                right = max(left+1, min(right, img.width))
                top = max(0, min(top, img.height))
                bottom = max(top+1, min(bottom, img.height))
                crop = img.crop((left, top, right, bottom))
                crop_path = str(out_dir / "crops" / f"p{pidx+1}_{len(formulas)}.png")
                crop.save(crop_path)

            used = "pix2text"
            latex = text

            # If Pix2Text text is empty/low, try pix2tex fallback
            if use_pix2tex and ocr is not None:
                need_fallback = (not latex) or (score is not None and score < 0.45)
                if need_fallback and crop_path is not None:
                    try:
                        latex_fallback = ocr(crop)
                        if latex_fallback and latex_fallback.strip():
                            latex = latex_fallback.strip()
                            used = "pix2tex"
                    except Exception as e:
                        print(f"[warn] pix2tex fallback failed on page {pidx+1}: {e}")

            if latex:
                # normalize for RAG: wrap display math
                if not (latex.startswith("$$") and latex.endswith("$$")):
                    latex = f"$$ {latex} $$"
                formulas.append(Formula(
                    page=pidx+1, bbox=bbox, latex=latex, source=used, score=score, crop_path=crop_path
                ))

    return formulas

def build_page_payloads(
    pdf_path: Path,
    marker_files: Dict[str, Any],
    per_page_text: List[str],
    formulas: List[Formula],
) -> List[Dict[str, Any]]:
    # group formulas by page
    f_by_page: Dict[int, List[Formula]] = {}
    for f in formulas:
        f_by_page.setdefault(f.page, []).append(f)

    n_pages = len(per_page_text)
    payloads = []
    for p in range(1, n_pages + 1):
        text = per_page_text[p - 1]
        has_math = has_math_markers(text)
        ftex = "\n".join([f.latex for f in f_by_page.get(p, [])])
        merged = text.strip()
        if ftex and ftex not in merged:
            merged = (merged + "\n\n" + ftex).strip() if merged else ftex

        payloads.append({
            "page": p,
            "text": merged,
            "has_math_markers": has_math,
            "formulas_added": len(f_by_page.get(p, [])),
            "formula_sources": list({f.source for f in f_by_page.get(p, [])}),
            "marker_md": marker_files.get("marker_md"),
            "marker_json": marker_files.get("marker_json"),
            "source_pdf": str(pdf_path),
        })
    return payloads

def write_jsonl_chunks(
    payloads: List[Dict[str, Any]],
    out_jsonl: Path,
    max_chars: int = 1800,
    doc_id: Optional[str] = None,
) -> None:
    ensure_dir(out_jsonl.parent)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for p in payloads:
            page = p["page"]
            body = p["text"]
            chunks = split_into_chunks(body, max_chars=max_chars) if body else []
            if not chunks:
                # still emit an empty chunk (helps traceability)
                chunks = [""]

            for i, ch in enumerate(chunks):
                rec = {
                    "id": f"{(doc_id or 'doc')}_p{page}_c{i}",
                    "text": ch,
                    "metadata": {
                        "page": page,
                        "source_pdf": p["source_pdf"],
                        "marker_md": p["marker_md"],
                        "marker_json": p.get("marker_json"),
                        "has_math_markers": p["has_math_markers"],
                        "formulas_added": p["formulas_added"],
                        "engine": "marker+p2t"  # +pix2tex if used; see formula_sources
                    },
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ------------------------ main ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to input PDF")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--force-ocr", action="store_true", help="Pass --force_ocr to Marker")
    ap.add_argument("--use-pix2tex", action="store_true", help="Try pix2tex fallback if Pix2Text is low/empty")
    ap.add_argument("--render-scale", type=float, default=2.0, help="Page render scale for OCR (pypdfium2)")
    ap.add_argument("--max-chars", type=int, default=1800, help="Max characters per chunk")
    args = ap.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    ensure_dir(out_dir)

    # 1) Run Marker (for overall text)
    marker_files = run_marker(pdf_path, out_dir, force_ocr=args.force_ocr)

    # 2) Get per-page text (independent of Marker schema)
    print("[info] extracting per-page text with pdfminer …")
    per_page_text = extract_text_per_page_with_pdfminer(pdf_path)

    # Pages with no math markers (eligible for formula OCR)
    pages_missing_math = [i for i, t in enumerate(per_page_text) if not has_math_markers(t)]
    print(f"[info] pages missing math markers: {[p+1 for p in pages_missing_math]}")

    # 3) Detect & OCR formulas only on those pages
    print("[info] running Pix2Text (and optional pix2tex fallback) …")
    formulas = detect_and_ocr_formulas(
        pdf_path=pdf_path,
        pages_to_process=pages_missing_math if pages_missing_math else list(range(len(per_page_text))),
        out_dir=out_dir,
        render_scale=args.render_scale,
        use_pix2tex=args.use_pix2tex,
    )
    print(f"[info] recovered {len(formulas)} formulas")

    # 4) Build payloads & write JSONL
    payloads = build_page_payloads(pdf_path, marker_files, per_page_text, formulas)
    out_jsonl = out_dir / f"{pdf_path.stem}.chunks.jsonl"
    write_jsonl_chunks(payloads, out_jsonl, max_chars=args.max_chars, doc_id=pdf_path.stem)

    print(f"[done] wrote chunks: {out_jsonl}")

if __name__ == "__main__":
    main()
