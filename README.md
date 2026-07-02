# AI Author: Chat to Book Pipeline

This project is a standalone tool designed to transform long, unstructured conversational transcripts (e.g., from ChatGPT or Claude) into a coherent, structured book. Unlike basic summary tools, this pipeline focuses on structure extraction, theme clustering, and maintaining authorial voice across generated chapters.

## Project Structure
- **`cluster_themes.py`**: Phase 1 script. Ingests a chat log, chunks the exchanges, runs embeddings, and clusters them into distinct themes.
- **`ingest.py`**: Parses exported chat logs (JSON or Markdown) into turn-by-turn conversational chunks.
- **`chunker.py`**: Groups logical back-and-forths together into semantic chunks.
- **`llm.py`**: Phase 2 module. Handles the connection to the OpenRouter inference engine for generating chapters.
- **`sample_export.json`**: A synthetic test corpus featuring four distinct topics for validating the clustering logic.

## Setup Instructions

### 1. Create a Virtual Environment (Recommended)
To avoid system-wide package conflicts (PEP 668), it is highly recommended to use a virtual environment.
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
Install the required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
The project uses `.env` to store your API keys and model choices safely.
1. Make sure you have the `.env` file in the root of the project.
2. Edit `.env` and replace `"your_openrouter_api_key_here"` with your actual OpenRouter API Key.

---

## How to Run

### Phase 1: Validating Theme Clustering
To test if the pipeline can successfully ingest a chat and separate it into logical themes *without* spending money on LLM calls, run:
```bash
python3 cluster_themes.py --input input.txt --n-clusters 8

```
This will output a summary of the themes it discovered within the synthetic test corpus.

### Phase 2: Testing LLM Inference
To verify that your OpenRouter connection is working and ready to draft chapters, run the LLM tester directly:
```bash
python llm.py
```
If properly configured, this will print a generated response using the free Llama-3 8B model.
