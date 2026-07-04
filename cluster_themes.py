"""
cluster_themes.py - Phase 1 of the chat-to-book pipeline.

Ingest -> chunk -> embed -> cluster -> print a theme report.

No LLM calls in this phase. The goal is purely to validate that
clustering produces human-recognizable themes before spending any
generation budget on outlines or chapter drafts.

Usage:
    python cluster_themes.py --input sample_export.json --n-clusters 4
"""

import argparse
import sys
from collections import defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ingest import load_export
from chunker import build_exchanges, chunks_from_exchanges


def embed_chunks_tfidf(chunks, n_components=50):
    """
    Offline fallback embedder: TF-IDF + truncated SVD (LSA).
    No model download required - useful for environments without
    access to huggingface.co, or as a zero-cost sanity check before
    committing to a heavier embedding model.
    """
    texts = [c["text"] for c in chunks]
    # n_components must be >= 1 and strictly < n_samples
    n_components = max(1, min(n_components, len(texts) - 1, 100))
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf = vectorizer.fit_transform(texts)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(tfidf)
    return normalize(reduced)


def embed_chunks_sbert(chunks, model_name="all-MiniLM-L6-v2"):
    """
    Real semantic embedder via sentence-transformers. Requires internet
    access to huggingface.co to download the model on first run - use
    this on your own machine, not in a network-restricted sandbox.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.array(embeddings)


def embed_chunks(chunks, embedder="tfidf", model_name="all-MiniLM-L6-v2"):
    if embedder == "sbert":
        return embed_chunks_sbert(chunks, model_name)
    return embed_chunks_tfidf(chunks)


def cluster_embeddings(embeddings, n_clusters):
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average",
    )
    return clustering.fit_predict(embeddings)


def top_keywords(texts, top_n=6):
    if len(texts) < 2:
        return []
    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    tfidf = vectorizer.fit_transform(texts)
    scores = np.asarray(tfidf.sum(axis=0)).ravel()
    terms = np.array(vectorizer.get_feature_names_out())
    top_idx = scores.argsort()[::-1][:top_n]
    return terms[top_idx].tolist()


def representative_chunk(chunks, embeddings, indices):
    group_embeddings = embeddings[indices]
    centroid = group_embeddings.mean(axis=0)
    sims = group_embeddings @ centroid / (
        np.linalg.norm(group_embeddings, axis=1) * np.linalg.norm(centroid) + 1e-9
    )
    best = indices[int(np.argmax(sims))]
    return chunks[best]


def run(input_path, n_clusters, embedder, model_name):
    turns = load_export(input_path)
    exchanges = build_exchanges(turns)
    chunks = chunks_from_exchanges(exchanges)

    print(f"Parsed {len(turns)} turns -> {len(exchanges)} exchanges -> {len(chunks)} chunks")
    print(f"Embedder: {embedder}\n")

    if not chunks:
        sys.exit(
            "Error: no chunks produced.\n"
            "Check that your input file contains recognisable turn markers\n"
            "(e.g. 'Human:' / 'Assistant:' or 'User:' / 'Claude:') or is\n"
            "valid JSON with 'role' and 'content' fields."
        )

    # AgglomerativeClustering requires at least 2 samples
    if len(chunks) < 2:
        sys.exit(
            f"Error: only {len(chunks)} chunk(s) found - need at least 2 to cluster.\n"
            "Your input file may not be using a recognised turn-marker format.\n"
            "Supported plain-text prefixes: Human/User/You (user) and\n"
            "Assistant/Claude/AI (assistant), optionally wrapped in ** **.\n"
            "Alternatively convert your export to JSON format."
        )

    if n_clusters > len(chunks):
        print(
            f"Warning: --n-clusters {n_clusters} > number of chunks ({len(chunks)}). "
            f"Reducing to {len(chunks)}.\n"
        )
        n_clusters = len(chunks)

    embeddings = embed_chunks(chunks, embedder=embedder, model_name=model_name)
    labels = cluster_embeddings(embeddings, n_clusters)

    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    print(f"Formed {len(clusters)} theme clusters:\n")
    for label in sorted(clusters):
        indices = np.array(clusters[label])
        cluster_texts = [chunks[i]["text"] for i in indices]
        keywords = top_keywords(cluster_texts)
        rep = representative_chunk(chunks, embeddings, indices)

        print(f"--- Cluster {label} ({len(indices)} chunks) ---")
        print(f"Keywords: {', '.join(keywords) if keywords else 'n/a'}")
        excerpt = rep["text"][:220].replace("\n", " ")
        print(f"Representative excerpt: {excerpt}...")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to chat export (.json or .md/.txt)")
    parser.add_argument("--n-clusters", type=int, default=6, help="Number of theme clusters to form")
    parser.add_argument("--embedder", choices=["tfidf", "sbert"], default="tfidf",
                         help="tfidf = offline fallback, sbert = real semantic embeddings (needs internet)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="sentence-transformers model name (sbert only)")
    args = parser.parse_args()
    run(args.input, args.n_clusters, args.embedder, args.model)
