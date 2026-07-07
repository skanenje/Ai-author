#!/usr/bin/env python3
"""
parse_export.py - Browse and extract conversations from a Claude.ai JSON export.

The Claude.ai data export (conversations.json) contains all your conversations
but is not easy to navigate raw. This tool lets you:

  List all conversations:
    python parse_export.py list --export conversations.json

  Search by name:
    python parse_export.py list --export conversations.json --search "Islamic"

  Extract one conversation as clean JSON (ready for cluster_themes.py):
    python parse_export.py extract --export conversations.json --index 3 --out out.json

  Extract by name keyword:
    python parse_export.py extract --export conversations.json --search "Buraq" --out out.json

  Extract ALL conversations merged into one file (for whole-corpus analysis):
    python parse_export.py extract --export conversations.json --all --out corpus.json

  Preview a conversation (first N turns, printed to stdout):
    python parse_export.py preview --export conversations.json --index 3 --turns 4
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_export(export_path: str) -> list:
    p = Path(export_path)
    if not p.exists():
        sys.exit(f"Error: export file not found: {export_path}")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        sys.exit("Error: expected a JSON array at the top level of the export file.")
    return data


def fmt_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except Exception:
        return iso[:10]


def extract_text(msg: dict) -> str:
    """Return the plain text of a message, preferring top-level 'text'."""
    text = msg.get("text", "").strip()
    if text:
        return text
    # Fall back to content array
    for block in msg.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            t = block.get("text", "").strip()
            if t:
                return t
    return ""


def conversation_to_turns(conv: dict) -> list:
    """Convert a conversation dict to a list of {role, content} turns."""
    turns = []
    for msg in conv.get("chat_messages", []):
        sender = msg.get("sender", "").lower()
        role = "user" if sender == "human" else "assistant"
        text = extract_text(msg)
        if text:
            turns.append({"role": role, "content": text})
    return turns


def match_conversations(data: list, search: str = None, index: int = None) -> list:
    """Return matching conversations as (original_index, conv) pairs."""
    if index is not None:
        if index < 0 or index >= len(data):
            sys.exit(f"Error: index {index} out of range (0–{len(data)-1}).")
        return [(index, data[index])]
    if search:
        q = search.lower()
        matches = [
            (i, c) for i, c in enumerate(data)
            if q in c.get("name", "").lower() or q in c.get("summary", "").lower()
        ]
        if not matches:
            sys.exit(f"No conversations found matching '{search}'.")
        return matches
    return list(enumerate(data))


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_list(args):
    data = load_export(args.export)
    matches = match_conversations(data, search=args.search)

    print(f"{'#':>5}  {'Date':10}  {'Msgs':>4}  Name")
    print("-" * 80)
    for i, conv in matches:
        date = fmt_date(conv.get("created_at", ""))
        n_msgs = len(conv.get("chat_messages", []))
        name = conv.get("name", "(unnamed)")[:60]
        print(f"{i:>5}  {date}  {n_msgs:>4}  {name}")
    print(f"\n{len(matches)} conversation(s) shown.")


def cmd_extract(args):
    data = load_export(args.export)

    if args.all:
        # Merge all conversations into one flat turn list
        all_turns = []
        for conv in data:
            all_turns.extend(conversation_to_turns(conv))
        out_path = Path(args.out)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_turns, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(all_turns)} turns from {len(data)} conversations → {out_path}")
        return

    matches = match_conversations(data, search=args.search, index=args.index)

    if len(matches) > 1 and not args.all:
        print(f"Found {len(matches)} conversations matching '{args.search}':")
        for i, conv in matches:
            print(f"  [{i:>4}] {conv.get('name', '')[:70]}")
        print("\nUse --index N to extract a specific one, or --all to merge all matches.")
        sys.exit(0)

    idx, conv = matches[0]
    turns = conversation_to_turns(conv)
    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(turns, f, ensure_ascii=False, indent=2)
    print(f"[{idx}] '{conv.get('name', '')}' → {len(turns)} turns → {out_path}")


def cmd_preview(args):
    data = load_export(args.export)
    matches = match_conversations(data, search=args.search, index=args.index)

    if len(matches) > 1:
        print(f"Found {len(matches)} matches. Showing first. Use --index N to be specific.")
    idx, conv = matches[0]

    print(f"\n=== [{idx}] {conv.get('name', '(unnamed)')} ===")
    print(f"    Created: {fmt_date(conv.get('created_at',''))}  |  "
          f"{len(conv.get('chat_messages',[]))} messages\n")

    turns = conversation_to_turns(conv)
    for t in turns[:args.turns]:
        label = "USER" if t["role"] == "user" else "CLAUDE"
        excerpt = t["content"][:400].replace("\n", " ")
        print(f"[{label}] {excerpt}")
        if len(t["content"]) > 400:
            print("       ...")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Browse and extract conversations from a Claude.ai JSON export."
    )
    parser.add_argument("--export", required=True,
                        help="Path to conversations.json from Claude.ai data export")

    sub = parser.add_subparsers(dest="command")

    # list
    p_list = sub.add_parser("list", help="List conversations")
    p_list.add_argument("--search", help="Filter by name keyword (case-insensitive)")

    # extract
    p_ext = sub.add_parser("extract", help="Extract conversation(s) to a JSON file")
    p_ext.add_argument("--index", type=int, help="Conversation index (from 'list')")
    p_ext.add_argument("--search", help="Match by name keyword")
    p_ext.add_argument("--all", action="store_true",
                       help="Merge ALL conversations into one file")
    p_ext.add_argument("--out", required=True, help="Output JSON file path")

    # preview
    p_pre = sub.add_parser("preview", help="Print a conversation to stdout")
    p_pre.add_argument("--index", type=int, help="Conversation index")
    p_pre.add_argument("--search", help="Match by name keyword")
    p_pre.add_argument("--turns", type=int, default=6,
                       help="Number of turns to show (default: 6)")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    {"list": cmd_list, "extract": cmd_extract, "preview": cmd_preview}[args.command](args)


if __name__ == "__main__":
    main()
