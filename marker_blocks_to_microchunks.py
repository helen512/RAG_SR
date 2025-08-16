#!/usr/bin/env python3
"""
marker_blocks_to_microchunks.py

Turn Marker `blocks.json` into micro-chunks for RAG:
- 1 chunk per paragraph/equation/list/caption (optionally split long paragraphs into sentences/windows)
- Preserve page, block_id, bbox, type
- Normalize whitespace, keep TeX in $$ ... $$ (from <math>...</math>)

Usage:
  python marker_blocks_to_microchunks.py \
    --blocks /path/to/blocks.json \
    --out /path/to/doc.microchunks.jsonl \
    --doc-id murray_lyapunov \
    --max-chars 600 \
    --split-sentences

Notes:
- This works even if Marker didn't emit the high-level JSON; `blocks.json` alone is enough.
- If your Marker schema has string `block_type` names (e.g., "Equation") instead of numeric codes,
  this script handles both.
"""

from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Fallback-friendly mapping. Marker variants may use numeric codes or names.
TYPE_EQN = {"14", "Equation", "MathBlock", "BlockMath"}
TYPE_INLINE = {"16", "TextInlineMath", "InlineMath"}
TYPE_PARA = {"23", "Paragraph", "Text", "ParagraphBlock"}
TYPE_LINE = {"1", "Line"}
TYPE_SPAN = {"2", "Span", "Token"}
TYPE_CAPTION = {"Caption", "20"}
TYPE_LIST = {"ListItem", "22", "List", "Bullet"}
TYPE_HEADER = {"SectionHeader", "Heading", "21"}

def bt(val: Any) -> str:
    """Normalize block_type to string token for matching."""
    if val is None:
        return ""
    if isinstance(val, int):
        return str(val)
    return str(val)

def get_text_from_span(span: Dict[str, Any]) -> str:
    # Prefer explicit 'text'; some schemas put text in 'content'
    return (span.get("text") or span.get("content") or "").rstrip("\n")

def math_from_html(html: str) -> str:
    """Extract TeX payload from <math ...> TEX </math> (Marker often emits TeX inside <math>)."""
    if not html:
        return ""
    m = re.search(r"<math[^>]*>([\s\S]*?)</math>", html)
    if not m:
        return ""
    return m.group(1).strip()

def normalize_ws(s: str) -> str:
    s = s.replace("\r", "")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()

def dehyphenate(s: str) -> str:
    # join soft hyphens at line breaks: "Lyapunov func-\ntion" -> "Lyapunov function"
    return re.sub(r"(\w)-\n(\w)", r"\1-\2", s)

def sentence_split(p: str) -> List[str]:
    # Lightweight sentence split that leaves $$...$$ intact.
    # Split on . ! ? ; when followed by space+capital or end of line.
    cuts = re.split(r'(?<=[\.\!\?;])\s+(?=[A-Z(\\])', p.strip())
    return [c.strip() for c in cuts if c.strip()]

def window_chunks(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    # fall back to hard wrap
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def rebuild_paragraph(blk: Dict[str, Any], by_id: Dict[str, Dict[str, Any]]) -> str:
    """Reconstruct paragraph text by walking lines -> spans."""
    out_lines: List[str] = []
    for item in blk.get("structure") or []:
        if bt(item.get("block_type")) != "1" and item.get("block_type") not in TYPE_LINE:
            continue
        line = by_id.get(item.get("block_id"))
        if not line:
            continue
        spans = []
        for s in (line.get("structure") or []):
            if bt(s.get("block_type")) not in ({"2"} | TYPE_SPAN):
                continue
            span = by_id.get(s.get("block_id"))
            if not span:
                continue
            spans.append(get_text_from_span(span))
        if spans:
            out_lines.append("".join(spans))
    para = "\n".join(out_lines)
    return normalize_ws(dehyphenate(para))

def collect_eqn_tex(blk: Dict[str, Any]) -> Optional[str]:
    html = blk.get("html") or ""
    tex = math_from_html(html)
    if tex:
        # Standardize to display math for RAG
        if not (tex.startswith("$$") and tex.endswith("$$")):
            tex = f"$$ {tex} $$"
        return tex
    return None

def bbox_of(blk: Dict[str, Any]) -> Optional[Dict[str, float]]:
    b = blk.get("bbox") or blk.get("box") or blk.get("position")
    if isinstance(b, dict):
        # normalize keys
        keys = {k.lower(): v for k,v in b.items()}
        x0 = keys.get("x0", keys.get("left", 0.0))
        y0 = keys.get("y0", keys.get("top", 0.0))
        x1 = keys.get("x1", keys.get("right", 0.0))
        y1 = keys.get("y1", keys.get("bottom", 0.0))
        return {"x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1)}
    return None

def make_microchunks(page: Dict[str, Any], doc_id: str, max_chars: int, split_sentences: bool) -> List[Dict[str, Any]]:
    blocks = page.get("children") or []
    # map for quick lookup
    by_id = {b.get("block_id"): b for b in blocks if b.get("block_id")}
    page_idx = int(page.get("page_id", 0)) + 1

    # find nearest header to tag section context (best-effort)
    headers = [b for b in blocks if bt(b.get("block_type")) in (TYPE_HEADER | {"21"})]
    section_text = None
    if headers:
        # pick the first header on the page
        section_text = normalize_ws((headers[0].get("text") or headers[0].get("content") or "").strip())

    micro: List[Dict[str, Any]] = []
    order = 0

    for blk in blocks:
        btype = bt(blk.get("block_type"))

        if btype in (TYPE_PARA | TYPE_INLINE | {"23","16","Paragraph","Text","TextInlineMath"}):
            para = rebuild_paragraph(blk, by_id)
            if not para:
                continue
            units: List[str]
            if split_sentences:
                # sentence windows, then wrap to max_chars
                pieces = sentence_split(para)
                units = []
                acc = ""
                for s in pieces:
                    if len(acc) + len(s) + 1 <= max_chars:
                        acc = s if not acc else acc + " " + s
                    else:
                        if acc:
                            units.append(acc)
                        acc = s
                if acc:
                    units.append(acc)
            else:
                units = window_chunks(para, max_chars)

            for k, txt in enumerate(units):
                order += 1
                micro.append({
                    "id": f"{doc_id}_p{page_idx}_b{blk.get('block_id')}_s{k}",
                    "text": txt,
                    "metadata": {
                        "type": "paragraph",
                        "page": page_idx,
                        "block_id": blk.get("block_id"),
                        "bbox": bbox_of(blk),
                        "order_on_page": order,
                        "section": section_text
                    }
                })

        elif btype in (TYPE_EQN | {"14","Equation"}):
            tex = collect_eqn_tex(blk)
            if not tex:
                continue
            order += 1
            micro.append({
                "id": f"{doc_id}_p{page_idx}_eq{blk.get('block_id')}",
                "text": tex,
                "metadata": {
                    "type": "equation",
                    "page": page_idx,
                    "block_id": blk.get("block_id"),
                    "bbox": bbox_of(blk),
                    "order_on_page": order,
                    "section": section_text
                }
            })

        elif btype in (TYPE_CAPTION | {"Caption","20"}):
            cap = normalize_ws(blk.get("text") or blk.get("content") or "")
            if not cap:
                # try reconstruct like a paragraph
                cap = rebuild_paragraph(blk, by_id)
            if cap:
                for k, txt in enumerate(window_chunks(cap, max_chars)):
                    order += 1
                    micro.append({
                        "id": f"{doc_id}_p{page_idx}_cap{blk.get('block_id')}_s{k}",
                        "text": txt,
                        "metadata": {
                            "type": "caption",
                            "page": page_idx,
                            "block_id": blk.get("block_id"),
                            "bbox": bbox_of(blk),
                            "order_on_page": order,
                            "section": section_text
                        }
                    })

        elif btype in (TYPE_LIST | {"ListItem","List","22"}):
            txt = rebuild_paragraph(blk, by_id)
            if txt:
                for k, piece in enumerate(window_chunks(txt, max_chars)):
                    order += 1
                    micro.append({
                        "id": f"{doc_id}_p{page_idx}_li{blk.get('block_id')}_s{k}",
                        "text": piece,
                        "metadata": {
                            "type": "list_item",
                            "page": page_idx,
                            "block_id": blk.get("block_id"),
                            "bbox": bbox_of(blk),
                            "order_on_page": order,
                            "section": section_text
                        }
                    })
        else:
            # ignore other types (tables/figures) for now or add handling if needed
            continue

    return micro

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", required=True, help="Marker blocks.json path")
    ap.add_argument("--out", required=True, help="Output micro-chunks JSONL")
    ap.add_argument("--doc-id", required=True, help="Logical document id (prefix for chunk ids)")
    ap.add_argument("--max-chars", type=int, default=600, help="Max characters per micro-chunk")
    ap.add_argument("--split-sentences", action="store_true", help="Split long paragraphs by sentences first")
    args = ap.parse_args()

    pages = json.loads(Path(args.blocks).read_text(encoding="utf-8"))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out_path.open("w", encoding="utf-8") as f:
        for page in pages:
            micro = make_microchunks(page, args.doc_id, args.max_chars, args.split_sentences)
            for m in micro:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
                total += 1

    print(f"✅ wrote {total} micro-chunks → {out_path}")

if __name__ == "__main__":
    main()
