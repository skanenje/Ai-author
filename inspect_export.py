"""
inspect_export.py - Peek at the structure of an official Claude data
export (conversations.json from Settings > Privacy > Export data)
before trusting an adapter to convert it.

Usage:
    python inspect_export.py --path conversations.json --title "Islamic"
"""

import argparse
import json


def main(path, title_filter):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Top-level type: {type(data).__name__}")
    if isinstance(data, list):
        print(f"Number of conversations: {len(data)}\n")
        if not data:
            return
        sample = data[0]
        print("Keys in a conversation object:", list(sample.keys()))

        if title_filter:
            matches = [
                c for c in data
                if title_filter.lower() in str(c.get("name", "")).lower()
            ]
            print(f"\nConversations matching '{title_filter}': {len(matches)}")
            for m in matches[:5]:
                print(f"  - {m.get('name', 'Untitled')!r} (uuid: {m.get('uuid', m.get('id', 'n/a'))})")
            sample = matches[0] if matches else sample

        # Try to find the message list under common key names
        for key in ("chat_messages", "messages", "mapping", "turns"):
            if key in sample:
                msgs = sample[key]
                print(f"\nFound message list under key '{key}', type: {type(msgs).__name__}")
                if isinstance(msgs, list) and msgs:
                    print("Keys in a single message object:", list(msgs[0].keys()))
                    print("\nFirst message (raw):")
                    print(json.dumps(msgs[0], indent=2)[:800])
                break
        else:
            print("\nNo obvious message-list key found among:", list(sample.keys()))
            print("Full sample (truncated):")
            print(json.dumps(sample, indent=2)[:1200])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to conversations.json from the export")
    parser.add_argument("--title", default=None, help="Substring to filter conversation by title")
    args = parser.parse_args()
    main(args.path, args.title)