"""
generate_book.py — Phase 3

Reads outline.json, generates each chapter using the LLM, and writes
the full book to a .txt file.

Usage:
    python generate_book.py
    python generate_book.py --outline outline.json --out book.txt
    python generate_book.py --dry-run  # print prompts only, no API calls
"""

import os
import sys
import json
import argparse
from dotenv import load_dotenv
from llm import generate_text

load_dotenv()


CHAPTER_SYSTEM_PROMPT = """You are an expert author and educator. Write in a clear, engaging, and authoritative style appropriate for your audience. Each chapter should:
- Open with a compelling hook or question
- Build arguments or ideas logically
- Use concrete examples and analogies where helpful
- End with a brief summary and transition to the next topic
Do NOT include any meta-commentary about yourself or the writing process. Write the chapter directly."""


def build_chapter_prompt(chapter, book_meta):
    return f"""Write a full, detailed chapter for the following book.

Book Title: {book_meta.get('title', 'Untitled')}
Book Thesis: {book_meta.get('thesis', '')}
Book Type: {os.getenv('BOOK_TYPE', 'educational nonfiction')}
Target Audience: {os.getenv('BOOK_AUDIENCE', 'general reader')}

---

Chapter {chapter['chapter_number']}: {chapter['title']}

Chapter Description: {chapter.get('description', '')}

Source Themes: {', '.join(chapter.get('source_themes', []))}

---

Write the full chapter now. Aim for a thorough, well-developed chapter (at least 600-1000 words). Do not include a chapter heading — the heading will be added separately."""


def main():
    parser = argparse.ArgumentParser(description="Generate the full book from outline.json.")
    parser.add_argument("--outline", default="outline.json", help="Path to outline.json")
    parser.add_argument("--out", default="book.txt", help="Output file for the book")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts only, no API calls")
    args = parser.parse_args()

    # Load outline
    try:
        with open(args.outline, "r", encoding="utf-8") as f:
            outline = json.load(f)
    except FileNotFoundError:
        print(f"Error: {args.outline} not found. Run generate_outline.py first.")
        sys.exit(1)

    title = outline.get("title", "Untitled Book")
    thesis = outline.get("thesis", "")
    chapters = outline.get("chapters", [])

    if not chapters:
        print("Error: No chapters found in outline.json.")
        sys.exit(1)

    print(f"\n📖 Generating: \"{title}\"")
    print(f"   {len(chapters)} chapters to write\n")

    book_parts = []

    # Book header
    book_parts.append(f"{title.upper()}\n")
    book_parts.append("=" * len(title) + "\n\n")
    if thesis:
        book_parts.append(f"{thesis}\n\n")
    book_parts.append("---\n\n")

    # Generate each chapter
    for chapter in chapters:
        chapter_num = chapter.get("chapter_number", "?")
        chapter_title = chapter.get("title", "Untitled Chapter")
        heading = f"CHAPTER {chapter_num}: {chapter_title.upper()}"

        print(f"  Writing Chapter {chapter_num}: {chapter_title}...")

        prompt = build_chapter_prompt(chapter, outline)

        if args.dry_run:
            print(f"\n--- PROMPT for Chapter {chapter_num} ---")
            print(prompt)
            print("--- END PROMPT ---\n")
            chapter_text = f"[DRY RUN — Chapter {chapter_num} content would appear here]"
        else:
            try:
                chapter_text = generate_text(prompt, system_prompt=CHAPTER_SYSTEM_PROMPT)
                print(f"  ✓ Chapter {chapter_num} done ({len(chapter_text)} chars)")
            except Exception as e:
                print(f"  ✗ Chapter {chapter_num} FAILED: {e}")
                chapter_text = f"[ERROR generating Chapter {chapter_num}: {e}]"

        book_parts.append(f"{heading}\n")
        book_parts.append("-" * len(heading) + "\n\n")
        book_parts.append(chapter_text.strip() + "\n\n\n")

    # Write book
    full_book = "\n".join(book_parts)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(full_book)

    if args.dry_run:
        print(f"\nDry run complete.")
        print(f"Output template saved to: {args.out}")
    else:
        print(f"\n✅ Book written to: {args.out}")
        print(f"   Total length: {len(full_book):,} characters")


if __name__ == "__main__":
    main()
