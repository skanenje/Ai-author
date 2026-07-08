"""
generate_outline.py - Stage 4 of the chat-to-book pipeline.

Takes the structured cluster report (from cluster_themes.py --save-report)
plus your stated objective, and asks Claude to propose a book structure:
title, thesis, ordered chapters (which may merge, split, or drop clusters),
and a one-line description per chapter.

This is a HUMAN CHECKPOINT stage. Nothing gets drafted from this outline
automatically - review it, edit it, re-run with a refined objective if
needed, before moving to chapter drafting. Regenerating an outline is
cheap; regenerating 12 drafted chapters because the ordering was wrong
is not.

Requires: pip install anthropic --break-system-packages
          export ANTHROPIC_API_KEY=your-key-here

Usage:
    python generate_outline.py \\
        --report cluster_report.json \\
        --objective "Argue that classical Islamic cosmology and modern \\
                     dimensional physics ask the same questions about \\
                     reality's structure, for an educated lay audience" \\
        --book-type "educational/research nonfiction" \\
        --out outline.json

    Add --dry-run to print the constructed prompt without calling the API
    (useful for checking the prompt before spending a real call on it).
"""

import argparse
import json
import os
import re

from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_MODEL = "claude-sonnet-5"
GEMINI_MODEL = "gemini-2.5-flash"
OPENROUTER_MODEL = "openrouter/free"

SYSTEM_PROMPT = """You are an expert book editor and structural writing \
consultant. You take scattered thematic material from a long conversation \
and turn it into a coherent, well-ordered nonfiction book outline.

Your job is NOT to include everything. A stated objective is a filter: \
material that doesn't serve the thesis should be left out or noted as a \
possible appendix/footnote, not forced into a chapter. Small clusters that \
are tangential asides (illustrative thought experiments, side questions) \
should usually NOT become their own chapter - fold them into a relevant \
chapter as an example, or drop them, unless the objective specifically \
calls for them.

Respond with ONLY valid JSON, no preamble, no markdown fences, matching \
this exact schema:

{
  "title": "string",
  "subtitle": "string or null",
  "thesis": "one paragraph restating the book's core argument",
  "audience": "string describing the target reader",
  "chapters": [
    {
      "chapter_number": 1,
      "title": "string",
      "one_line_description": "string",
      "source_cluster_ids": [0, 2],
      "rationale": "why these clusters belong together here, and how this \
chapter advances the thesis"
    }
  ],
  "excluded_clusters": [
    {"cluster_id": 7, "reason": "string explaining why this was left out"}
  ]
}
"""


def build_user_prompt(report, objective, book_type, audience_hint):
    cluster_summaries = []
    for c in report:
        cluster_summaries.append(
            f"Cluster {c['cluster_id']} ({c['size']} chunks)\n"
            f"Keywords: {', '.join(c['keywords'])}\n"
            f"Representative excerpt: {c['representative_excerpt']}\n"
        )
    clusters_block = "\n".join(cluster_summaries)

    audience_line = f"\nIntended audience: {audience_hint}" if audience_hint else ""

    return f"""Here is the stated objective for this book:

Objective: {objective}
Book type: {book_type}{audience_line}

Here are the theme clusters extracted from the source conversation, each \
representing a candidate chapter topic:

{clusters_block}

Propose a book outline that serves the stated objective. You may merge \
clusters into a single chapter, split a large cluster into multiple \
chapters, reorder them for argumentative flow, or exclude clusters that \
don't serve the thesis. Explain your reasoning for each chapter and for \
any exclusions."""


def extract_json(raw_text):
    """
    Free models are less reliable than Sonnet 5 at obeying a strict
    'JSON only, no preamble' instruction - they often wrap the JSON in
    a sentence or two of explanation despite being told not to. Try a
    direct parse first, then fall back to locating the outermost {...}
    block in the response.
    """
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw_text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Could not parse JSON from the model's response. Raw response was:\n\n"
        + raw_text[:2000]
    )


def call_anthropic(system_prompt, user_prompt, model):
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=4000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def call_gemini(system_prompt, user_prompt, model):
    import os
    from google import genai
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in your environment first.")
        
    client = genai.Client(api_key=api_key)
    
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
        )
    )
    return response.text

def call_openrouter(system_prompt, user_prompt, model):
    import os
    import requests

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY in your environment first.")

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def generate_outline(report_path, objective, book_type, audience_hint, out_path, dry_run, provider, model):
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    user_prompt = build_user_prompt(report, objective, book_type, audience_hint)

    if dry_run:
        print("=== SYSTEM PROMPT ===\n")
        print(SYSTEM_PROMPT)
        print("\n=== USER PROMPT ===\n")
        print(user_prompt)
        print(f"\n(--dry-run: no API call made, would use provider={provider}, model={model})")
        return

    if provider == "gemini":
        raw_text = call_gemini(SYSTEM_PROMPT, user_prompt, model)
    elif provider == "openrouter":
        raw_text = call_openrouter(SYSTEM_PROMPT, user_prompt, model)
    else:
        raw_text = call_anthropic(SYSTEM_PROMPT, user_prompt, model)

    outline = extract_json(raw_text)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(outline, f, indent=2, ensure_ascii=False)

    print(f"Title: {outline['title']}")
    if outline.get("subtitle"):
        print(f"Subtitle: {outline['subtitle']}")
    print(f"\nThesis: {outline['thesis']}\n")
    print(f"Audience: {outline['audience']}\n")
    print(f"Chapters ({len(outline['chapters'])}):")
    for ch in outline["chapters"]:
        print(f"  {ch['chapter_number']}. {ch['title']}")
        print(f"     {ch['one_line_description']}")
        print(f"     (from clusters: {ch['source_cluster_ids']})")
    if outline.get("excluded_clusters"):
        print(f"\nExcluded:")
        for ex in outline["excluded_clusters"]:
            print(f"  Cluster {ex['cluster_id']}: {ex['reason']}")

    print(f"\nSaved full outline to {out_path}")
    print("Review this before moving to chapter drafting - re-run with a")
    print("refined objective if the structure isn't right yet.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, help="Path to cluster_themes.py --save-report JSON output")
    parser.add_argument("--objective", default=None, help="One-paragraph thesis/goal statement (falls back to BOOK_OBJECTIVE in .env)")
    parser.add_argument("--book-type", default=None, help="e.g. 'educational', 'research monograph' (falls back to BOOK_TYPE in .env)")
    parser.add_argument("--audience", default=None, help="Describe the intended reader (falls back to BOOK_AUDIENCE in .env)")
    parser.add_argument("--out", default="outline.json", help="Where to save the generated outline")
    parser.add_argument("--dry-run", action="store_true", help="Print the prompt without calling the API")
    parser.add_argument("--provider", choices=["gemini", "openrouter", "anthropic"], default="gemini")
    parser.add_argument("--model", default=None, help="Override the default model for the chosen provider")
    args = parser.parse_args()

    # Fall back to .env values if not provided on CLI
    objective = args.objective or os.getenv("BOOK_OBJECTIVE", "")
    book_type = args.book_type or os.getenv("BOOK_TYPE", "educational nonfiction")
    audience = args.audience or os.getenv("BOOK_AUDIENCE", None)

    if not objective:
        print("Error: No objective provided. Set BOOK_OBJECTIVE in .env or pass --objective.")
        raise SystemExit(1)

    # Select model based on provider if not overridden
    if not args.model:
        if args.provider == "gemini":
            model = GEMINI_MODEL
        elif args.provider == "openrouter":
            model = OPENROUTER_MODEL
        else:
            model = ANTHROPIC_MODEL
    else:
        model = args.model

    generate_outline(args.report, objective, book_type, audience, args.out, args.dry_run, args.provider, model)