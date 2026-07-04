"""
ingest.py - Parse chat exports into a normalized list of turns.

Supports three input formats:

  1. JSON: a list of {"role": "user"|"assistant", "content": "..."} objects

  2. Standard markdown/plain text: lines prefixed with a role marker, e.g.
       Human: ...
       Assistant: ...
     or
       **User:** ...
       **Claude:** ...

  3. Claude.ai copy-paste format:
       [user message — no prefix]

       Show more

       [summary line (repeated twice)]
       [assistant response]
       [next user message — directly appended, no separator]

       Show more
       ...

     The key insight is that "Show more" is the only unambiguous boundary.
     After each "Show more" block, the content is:
       - 2 duplicate summary lines (skip them)
       - assistant response paragraphs
       - final paragraph(s) = next user message (if another block follows)
"""

import json
import re
from pathlib import Path
from typing import List, Dict

ROLE_MARKERS = {
    "human": "user",
    "user": "user",
    "you": "user",
    "assistant": "assistant",
    "claude": "assistant",
    "ai": "assistant",
}

TURN_SPLIT_RE = re.compile(
    r"^\s*\**\s*(Human|User|You|Assistant|Claude|AI)\s*:\**\s*",
    re.IGNORECASE | re.MULTILINE,
)

# Matches the "Show more" separator used in Claude.ai copy-paste exports
SHOW_MORE_RE = re.compile(r"^\s*Show more\s*$", re.IGNORECASE | re.MULTILINE)

# A paragraph boundary: two or more newlines
PARA_SPLIT_RE = re.compile(r"\n{2,}")


def load_json_export(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    turns = []
    for item in data:
        role = str(item.get("role", "")).lower()
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        norm_role = ROLE_MARKERS.get(role, role)
        turns.append({"role": norm_role, "content": content})
    return turns


def _strip_summary_lines(lines: List[str]) -> List[str]:
    """
    Remove the two leading duplicate summary lines that Claude.ai inserts
    before each assistant response in the copy-paste export.

    The raw block starts with blank lines, then two identical summary lines.
    We skip leading blank lines first, then check for the duplicate pair.
    """
    # Drop leading blank lines
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    lines = lines[i:]

    # Remove the two duplicate summary lines
    if len(lines) >= 2 and lines[0].strip() == lines[1].strip() and lines[0].strip():
        lines = lines[2:]

    return lines


def load_claude_copy_paste(text: str) -> List[Dict]:
    """
    Parse Claude.ai's copy-paste export format.

    Structure after splitting on "Show more":
      parts[0]  = first user message
      parts[1]  = [summary x2] + assistant_1_body + user_2_message
      parts[2]  = [summary x2] + assistant_2_body + user_3_message
      ...
      parts[-1] = [summary x2] + last_assistant_body
                  (no trailing user message if conversation ended with assistant)

    For each parts[i] where i >= 1:
      1. Strip leading blank lines, then the two duplicate summary lines.
      2. Split remaining content into paragraphs.
      3. If followed by another "Show more" block (i < len(parts)-1),
         the last paragraph block is the next user message.
      4. If this is the last part, everything is the assistant response.
    """
    parts = SHOW_MORE_RE.split(text)
    turns: List[Dict] = []

    # parts[0]: content before the very first "Show more" → first user message
    first_user = parts[0].strip()
    if first_user:
        turns.append({"role": "user", "content": first_user})

    for i in range(1, len(parts)):
        block = parts[i]
        lines = _strip_summary_lines(block.split("\n"))
        remaining = "\n".join(lines).strip()

        is_last = (i == len(parts) - 1)

        if is_last:
            # Everything is the assistant response
            if remaining:
                turns.append({"role": "assistant", "content": remaining})
        else:
            # Split into paragraphs; last paragraph belongs to the next user turn
            paragraphs = [p.strip() for p in PARA_SPLIT_RE.split(remaining) if p.strip()]
            if not paragraphs:
                continue
            if len(paragraphs) == 1:
                # Only one paragraph — treat as assistant (edge case)
                turns.append({"role": "assistant", "content": paragraphs[0]})
            else:
                # Last paragraph = next user message
                user_msg = paragraphs[-1]
                assistant_body = "\n\n".join(paragraphs[:-1])
                if assistant_body:
                    turns.append({"role": "assistant", "content": assistant_body})
                if user_msg:
                    turns.append({"role": "user", "content": user_msg})

    return turns


def load_markdown_export(text: str) -> List[Dict]:
    """Parse the standard Human:/Assistant: role-marker format."""
    matches = list(TURN_SPLIT_RE.finditer(text))
    turns = []
    for i, m in enumerate(matches):
        role_raw = m.group(1).lower()
        role = ROLE_MARKERS.get(role_raw, role_raw)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            turns.append({"role": role, "content": content})
    return turns


def load_export(path: str) -> List[Dict]:
    p = Path(path)
    if p.suffix.lower() == ".json":
        return load_json_export(p)

    text = p.read_text(encoding="utf-8")
    # Normalise Windows-style CRLF so that ^ anchors work correctly
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Prefer the Claude.ai copy-paste parser whenever "Show more" markers are
    # present — these are unambiguous UI artefacts that don't appear in
    # natural conversation text.
    if SHOW_MORE_RE.search(text):
        return load_claude_copy_paste(text)

    # Fall back to standard role-marker format
    return load_markdown_export(text)
