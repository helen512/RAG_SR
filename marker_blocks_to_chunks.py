#!/usr/bin/env python3
import json, re, argparse
from pathlib import Path

# types we care about inside blocks.json
T_PARAGRAPH = "23"   # paragraph/line
T_LINE      = "1"    # line
T_SPAN      = "2"    # span
T_EQN       = "14"   # block equation
T_INLINE    = "16"   # paragraph that contains inline math (still uses lines+spans)

def dehyphenate(s: str) -> str:
    # join hyphenated line-breaks: "so-\ncalled" -> "so-called"
    return re.sub(r"(\w)-\n(\w)", r"\1-\2", s)

def normalize_ws(s: str) -> str:
    s = s.replace("\r", "")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()

def extract_tex_from_math_html(html: str) -> str:
    # <math ...> TEX </math>  -> return TEX only
    m = re.search(r"<math[^>]*>([\s\S]*?)</math>", html or "")
    return (m.group(1).strip() if m else "").strip()

def page_text_from_children(children):
    """Rebuild prose by following paragraph -> line -> span.text."""
    # Group children by block_id for quick lookup
    by_id = {c.get("block_id"): c for c in children}
    texts = []

    for blk in children:
        bt = blk.get("block_type")
        if bt in (T_PARAGRAPH, T_INLINE):
            # Walk lines
            line_ids = [x["block_id"] for x in (blk.get("structure") or []) if x.get("block_type")==T_LINE]
            for lid in line_ids:
                line = by_id.get(lid)
                if not line: continue
                span_ids = [x["block_id"] for x in (line.get("structure") or []) if x.get("block_type")==T_SPAN]
                line_txt = ""
                for sid in span_ids:
                    span = by_id.get(sid)
                    if span and span.get("text"):
                        line_txt += span["text"]
                if line_txt.strip():
                    texts.append(line_txt.rstrip("\n"))
    prose = "\n".join(texts)
    return normalize_ws(dehyphenate(prose))

def collect_display_equations(children):
    eqs = []
    for blk in children:
        if blk.get("block_type") == T_EQN:
            tex = extract_tex_from_math_html(blk.get("html") or "")
            if tex:
                # wrap as display math for LLMs/retrievers
                eqs.append(f"$$ {tex} $$")
    return eqs

def chunk(text, max_chars=1800):
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = p if not buf else f"{buf}\n\n{p}"
        else:
            if buf: out.append(buf)
            if len(p) <= max_chars:
                buf = p
            else:
                # hard cut long para
                for i in range(0, len(p), max_chars):
                    out.append(p[i:i+max_chars])
                buf = ""
    if buf: out.append(buf)
    return out or [""]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", required=True, help="Path to blocks.json from Marker")
    ap.add_argument("--out", required=True, help="Output JSONL file")
    ap.add_argument("--doc-id", default=None)
    ap.add_argument("--max-chars", type=int, default=1800)
    args = ap.parse_args()

    pages = json.loads(Path(args.blocks).read_text(encoding="utf-8"))
    with open(args.out, "w", encoding="utf-8") as fout:
        for page in pages:
            page_idx = int(page.get("page_id", 0)) + 1
            children = page.get("children") or []
            prose = page_text_from_children(children)
            eqns  = collect_display_equations(children)
            body  = "\n\n".join([s for s in [prose, "\n".join(eqns)] if s.strip()])

            for i, ch in enumerate(chunk(body, args.max_chars)):
                rec = {
                    "id": f"{(args.doc_id or 'doc')}_p{page_idx}_c{i}",
                    "text": ch,
                    "metadata": {
                        "page": page_idx,
                        "source": "marker/blocks.json",
                        "equation_count_on_page": len(eqns),
                    },
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
