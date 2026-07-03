"""
ingest.py - Parse chat exports into a normalized list of turns.

Supports two input formats:
  - JSON: a list of {"role": "user"|"assistant", "content": "..."} objects
  - Markdown/plain text: lines prefixed with a role marker, e.g.
      Human: ...
      Assistant: ...
    or
      **User:** ...
      **Claude:** ...
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


def load_markdown_export(path: Path) -> List[Dict]:
    text = path.read_text(encoding="utf-8")
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
    return load_markdown_export(p)