# chat2book — Phase 1 (ingest → chunk → embed → cluster)

Zero-LLM-cost pipeline stage. Goal: prove the corpus splits into
human-recognizable themes before spending any generation budget on
outlines or chapters.

## Setup (your machine, conda env `ex00`)

```bash
conda activate ex00
cd ~/chat2book        # wherever you place this folder
pip install -r requirements.txt
```

If `conda activate ex00` doesn't take inside a script/non-interactive
shell (the issue you hit on Orca), either run `conda init bash` once
and restart the shell, or use `conda run -n ex00 python cluster_themes.py ...`
directly instead of activating first.

The first run of `sentence-transformers` will download the
`all-MiniLM-L6-v2` model (~90MB) from huggingface.co — this needs
real internet access, which is why it can't run in the sandboxed
environment I tested in.

## Getting your real corpus

1. In claude.ai: Settings → Privacy → Export data. You'll get an
   emailed download link to a ZIP containing `conversations.json`
   (all your conversations, not just one).
2. Unzip it, then inspect the structure before trusting any adapter:

   ```bash
   python inspect_export.py --path conversations.json --title "Islamic"
   ```

   This prints the actual key names Anthropic uses for role/content
   in your export so we can write `convert_claude_export.py` to match
   reality instead of guessing. Send me that output (or just the
   printed keys) and I'll write the adapter to match.

3. Once converted to the simple format `ingest.py` expects —
   `[{"role": "user"|"assistant", "content": "..."}, ...]` — you're
   ready to run the real pipeline.

## Running with real semantic embeddings

```bash
python cluster_themes.py --input your_theology_chat.json --n-clusters 8 --embedder sbert
```

Start with `--n-clusters` roughly equal to how many chapters you'd
guess the book needs, then adjust after reading the keyword/excerpt
report — too few clusters merges distinct arguments together, too
many fragments a single argument across clusters.

## What "good" looks like

Each cluster's keyword list and representative excerpt should read as
one coherent sub-topic. If two clusters look like they're circling the
same idea, lower `--n-clusters`. If one cluster's excerpt reads like
it's straddling two unrelated arguments, raise it.

## Files

- `ingest.py` — parses chat export into normalized turns
- `chunker.py` — merges turns into exchange-level chunks
- `cluster_themes.py` — embed + cluster + report (this is the one you run)
- `inspect_export.py` — peek at official export structure before adapting it
- `sample_export.json` — synthetic 4-topic test corpus (already validated
  the plumbing works end-to-end; ML/IT topics only separated cleanly once
  real embeddings replaced the TF-IDF fallback — that's the whole point
  of running this with `--embedder sbert`)