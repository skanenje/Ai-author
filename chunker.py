"""
chunker.py - Group turns into exchanges and split into overlapping chunks.

Two chunking modes:
  exchange mode  (default): sliding window across exchanges.
  paragraph mode (auto when any turn > LONG_TURN_CHARS chars):
      Turns split on blank lines first, then windowed across paragraphs.
      Handles very long assistant responses in copy-pasted conversations.
"""

from typing import List, Dict
import re

LONG_TURN_CHARS = 4000
PARA_SPLIT_RE   = re.compile(r"\n{2,}")


def build_exchanges(turns: List[Dict]) -> List[Dict]:
    """Pair consecutive user/assistant turns into exchange dicts."""
    exchanges: List[Dict] = []
    i = 0
    while i < len(turns):
        turn    = turns[i]
        role    = turn.get("role", "")
        content = turn.get("content", "").strip()
        if role == "user":
            if i + 1 < len(turns) and turns[i + 1].get("role") == "assistant":
                exchanges.append({
                    "user":      content,
                    "assistant": turns[i + 1].get("content", "").strip(),
                    "index":     len(exchanges),
                })
                i += 2
            else:
                exchanges.append({"user": content, "assistant": "", "index": len(exchanges)})
                i += 1
        elif role == "assistant":
            exchanges.append({"user": "", "assistant": content, "index": len(exchanges)})
            i += 1
        else:
            i += 1
    return exchanges


def _split_paragraphs(text: str, min_chars: int = 120) -> List[str]:
    """Split on blank lines, merging short fragments into neighbours."""
    raw    = [p.strip() for p in PARA_SPLIT_RE.split(text) if p.strip()]
    merged: List[str] = []
    buf    = ""
    for para in raw:
        buf = (buf + chr(10)*2 + para).strip() if buf else para
        if len(buf) >= min_chars:
            merged.append(buf)
            buf = ""
    if buf:
        if merged:
            merged[-1] += chr(10)*2 + buf
        else:
            merged.append(buf)
    return merged


def _exchange_to_text(ex: Dict) -> str:
    parts = []
    if ex.get("user"):      parts.append("User: "      + ex["user"])
    if ex.get("assistant"): parts.append("Assistant: " + ex["assistant"])
    return chr(10).join(parts)


def _needs_paragraph_mode(exchanges: List[Dict], threshold: int = LONG_TURN_CHARS) -> bool:
    for ex in exchanges:
        if len(ex.get("user", "")) > threshold or len(ex.get("assistant", "")) > threshold:
            return True
    return False


def _paragraph_chunks(exchanges, window, stride, min_chars):
    paras: List[Dict] = []
    for ex in exchanges:
        idx = ex["index"]
        if ex.get("user"):
            for p in _split_paragraphs(ex["user"]):
                paras.append({"text": "User: " + p, "exchange_index": idx})
        if ex.get("assistant"):
            for p in _split_paragraphs(ex["assistant"]):
                paras.append({"text": "Assistant: " + p, "exchange_index": idx})
    chunks: List[Dict] = []
    n = len(paras)
    for start in range(0, n, stride):
        end  = min(start + window, n)
        win  = paras[start:end]
        text = (chr(10)*2).join(p["text"] for p in win).strip()
        if len(text) >= min_chars:
            chunks.append({
                "text":        text,
                "start_index": win[0]["exchange_index"],
                "end_index":   win[-1]["exchange_index"],
            })
        if end == n:
            break
    return chunks


def chunks_from_exchanges(
    exchanges  : List[Dict],
    window     : int = 3,
    stride     : int = 1,
    min_chars  : int = 50,
    para_window: int = 5,
    para_stride: int = 2,
) -> List[Dict]:
    """
    Produce overlapping text chunks. Auto-switches to paragraph mode
    when any turn exceeds LONG_TURN_CHARS characters.
    Each chunk: {"text": str, "start_index": int, "end_index": int}
    """
    if not exchanges:
        return []
    if _needs_paragraph_mode(exchanges):
        return _paragraph_chunks(exchanges, para_window, para_stride, min_chars)
    chunks: List[Dict] = []
    n = len(exchanges)
    for start in range(0, n, stride):
        end  = min(start + window, n)
        win  = exchanges[start:end]
        text = (chr(10)*2).join(_exchange_to_text(e) for e in win).strip()
        if len(text) >= min_chars:
            chunks.append({
                "text":        text,
                "start_index": win[0]["index"],
                "end_index":   win[-1]["index"],
            })
        if end == n:
            break
    return chunks
