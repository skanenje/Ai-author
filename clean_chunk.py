"""
clean_chunk.py - Turn ANY copy-pasted AI chat text into clean,
semantically-sized chunks, without depending on the source platform's
role markers or copy format.

Design principle: raw copy-paste from a chat UI (Claude, ChatGPT,
Gemini, whatever) is never reliably structured - different platforms
leak different UI chrome into a page copy, and turn boundaries are
often invisible in the text itself. Rather than betting on detecting
user/assistant boundaries correctly, this treats the whole file as one
corpus: strip known noise, collapse copy-paste duplication artifacts,
then chunk by paragraph. That's what actually matters for building a
book - the ideas, not who said which sentence.

Usage:
    python clean_chunk.py --input input.txt
"""

import argparse
import re

# UI chrome strings commonly leaked into a raw page copy across
# Claude / ChatGPT / Gemini interfaces.
NOISE_LINE_PATTERNS = [
    r"^\s*Show more\s*$",
    r"^\s*Show less\s*$",
    r"^\s*Copy\s*$",
    r"^\s*Copy code\s*$",
    r"^\s*Retry\s*$",
    r"^\s*Regenerate( response)?\s*$",
    r"^\s*Edit\s*$",
    r"^\s*\d+\s*/\s*\d+\s*$",  # pagination under regenerated replies, e.g. "1 / 2"
]
NOISE_RE = re.compile("|".join(NOISE_LINE_PATTERNS), re.IGNORECASE | re.MULTILINE)

# Role-prefix labels to STRIP if present - never required for this to work.
ROLE_PREFIX_RE = re.compile(
    r"^\s*\**\s*(Human|User|You said|You|Assistant|Claude|ChatGPT said|ChatGPT|Gemini|AI)\s*:\**\s*",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def clean_text(raw: str) -> str:
    """
    Strip known UI chrome and collapse the collapsed-heading paste
    artifact (a line repeated back-to-back). Deliberately does NOT try
    to detect duplicated passages fused onto new content - that pattern
    has no clean regex signature (see clean_chunk near-duplicate notes
    in cluster_themes.py). Near-duplicate *chunks* get caught later via
    embedding similarity, which handles it far more robustly than
    string matching on raw, unpunctuated, run-on text ever could.
    """
    text = NOISE_RE.sub("", raw)

    lines = text.split("\n")
    lines = [ROLE_PREFIX_RE.sub("", ln) for ln in lines]

    deduped_lines = []
    for ln in lines:
        if deduped_lines and ln.strip() and ln.strip() == deduped_lines[-1].strip():
            continue
        deduped_lines.append(ln)
    text = "\n".join(deduped_lines)

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return "\n\n".join(paragraphs)


def merge_and_split(paragraphs, target_chars=1200, max_chars=2000):
    """Combine short consecutive paragraphs up toward target_chars;
    split any single paragraph that exceeds max_chars on sentence
    boundaries. Produces evenly-sized chunks regardless of how the
    source text was originally broken up."""
    chunks = []
    buf = ""
    for para in paragraphs:
        if len(para) > max_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            sentences = re.split(r"(?<=[.!?])\s+", para)
            piece = ""
            for s in sentences:
                if len(piece) + len(s) + 1 <= max_chars:
                    piece = f"{piece} {s}".strip()
                else:
                    if piece:
                        chunks.append(piece)
                    piece = s
            if piece:
                chunks.append(piece)
            continue

        if len(buf) + len(para) + 2 <= target_chars:
            buf = f"{buf}\n\n{para}".strip()
        else:
            if buf:
                chunks.append(buf)
            buf = para
    if buf:
        chunks.append(buf)
    return chunks


def chunk_file(input_path, target_chars=1200, max_chars=2000):
    with open(input_path, "r", encoding="utf-8") as f:
        raw = f.read()
    cleaned = clean_text(raw)
    paragraphs = [p for p in cleaned.split("\n\n") if p.strip()]
    pieces = merge_and_split(paragraphs, target_chars, max_chars)
    return [{"chunk_id": str(i), "text": piece} for i, piece in enumerate(pieces)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw pasted chat text")
    parser.add_argument("--target-chars", type=int, default=1200)
    parser.add_argument("--max-chars", type=int, default=2000)
    args = parser.parse_args()

    chunks = chunk_file(args.input, args.target_chars, args.max_chars)
    print(f"Produced {len(chunks)} chunks\n")
    for c in chunks[:8]:
        print(f"--- chunk {c['chunk_id']} ({len(c['text'])} chars) ---")
        print(c["text"][:200].replace("\n", " "), "...")
        print()