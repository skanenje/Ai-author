# chat2book

Convert long AI conversations into structured nonfiction book material.

> **Phase 1 (current):** Ingest → chunk → embed → cluster  
> Zero LLM cost. Proves the corpus splits into human-recognizable themes
> before spending any generation budget on outlines or chapter drafts.

---

## Setup

```bash
git clone https://github.com/skanenje/Ai-author
cd /Ai-author
pip install -r requirements.txt
```

## Workflow

### 1. Prepare your input

**Option A — copy-paste (simplest)**

Copy a conversation from Claude.ai and paste it into a `.txt` file.
The parser auto-detects the format (the `Show more` separators Claude.ai inserts).

**Option B — official JSON export**

Download from *Settings → Privacy → Export data*, unzip, then:

```bash
EXPORT=~/Downloads/.../conversations.json

# List all conversations
python parse_export.py --export $EXPORT list

# Search by name keyword
python parse_export.py --export $EXPORT list --search Islamic

# Preview a conversation (first 6 turns)
python parse_export.py --export $EXPORT preview --index 465

# Extract one conversation -> ready for cluster_themes.py
python parse_export.py --export $EXPORT extract --index 465 --out conv.json

# Merge ALL conversations into one corpus file
python parse_export.py --export $EXPORT extract --all --out corpus.json
```

### 2. Run the clustering pipeline

```bash
# Copy-paste .txt file
python cluster_themes.py --input input.txt --n-clusters 8

# Extracted .json file
python cluster_themes.py --input conv.json --n-clusters 6
```

### 3. Tune the cluster count

Each cluster prints a keyword list and a representative excerpt.

| Symptom | Fix |
|---|---|
| Two clusters look like the same topic | Lower `--n-clusters` |
| One cluster mixes unrelated ideas | Raise `--n-clusters` |

Start near your expected chapter count and adjust from there.

---

## Embedder options

| Flag | Model | Notes |
|---|---|---|
| `--embedder tfidf` | TF-IDF + SVD | **Default.** Fully offline. |
| `--embedder sbert` | all-MiniLM-L6-v2 | Better clusters. Downloads ~90 MB on first run. |

Use `tfidf` for a fast sanity check; switch to `sbert` when you want final chapter assignments.

---

## Project files

| File | Purpose |
|---|---|
| `ingest.py` | Parse any supported format into normalised turns |
| `chunker.py` | Group turns into overlapping text chunks (auto paragraph-splits long responses) |
| `cluster_themes.py` | Embed + cluster + print theme report ← **run this** |
| `parse_export.py` | Browse and extract from `conversations.json` export |
| `llm.py` | LLM client stub (Phase 2 — chapter drafting, not yet used) |
| `requirements.txt` | Python dependencies |
| `.env` | API keys (gitignored) |

---

## Supported input formats

`ingest.py` auto-detects all three:

| Format | Detection trigger | How to produce |
|---|---|---|
| Claude.ai copy-paste `.txt` | `Show more` separator lines | Copy conversation from browser, paste to file |
| Role-marker `.txt` | `Human:` / `Assistant:` prefixes | Any chat log with role labels |
| Extracted `.json` | File extension | Output of `parse_export.py extract` |
