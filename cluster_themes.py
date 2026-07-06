"""
cluster_themes.py - Phase 1 of the chat-to-book pipeline.

Clean -> chunk -> embed -> dedup -> cluster -> print a theme report.

No LLM calls in this phase. The goal is purely to validate that
clustering produces human-recognizable themes before spending any
generation budget on outlines or chapter drafts.

Two input modes:
  --mode raw  (default) - any pasted chat text, no role markers required.
              Uses clean_chunk.py: strip UI noise, chunk by paragraph.
  --mode qa   - structured JSON export with clean role/content per turn.
              Uses ingest.py + chunker.py: pairs user/assistant exchanges.

Usage:
    python cluster_themes.py --input input.txt --n-clusters 8
    python cluster_themes.py --input theology_chat.json --mode qa --n-clusters 8
"""

import argparse
from collections import defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from clean_chunk import chunk_file
from ingest import load_export
from chunker import build_exchanges, chunks_from_exchanges


def get_chunks(input_path, mode):
    if mode == "qa":
        turns = load_export(input_path)
        exchanges = build_exchanges(turns)
        chunks = chunks_from_exchanges(exchanges)
        print(f"Parsed {len(turns)} turns -> {len(exchanges)} exchanges -> {len(chunks)} chunks")
        return chunks
    chunks = chunk_file(input_path)
    print(f"Cleaned and chunked into {len(chunks)} paragraphs-based chunks")
    return chunks


def embed_chunks_tfidf(chunks, n_components=50):
    """
    Offline fallback embedder: TF-IDF + truncated SVD (LSA).
    No model download required - useful for environments without
    access to huggingface.co, or as a zero-cost sanity check before
    committing to a heavier embedding model.
    """
    texts = [c["text"] for c in chunks]
    n_components = min(n_components, len(texts) - 1, 100)
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


def drop_near_duplicates(chunks, embeddings, threshold=0.85):
    """
    Catch near-duplicate chunks via embedding similarity rather than
    string matching. Handles the case where a pasted transcript repeats
    a passage fused onto different surrounding text - no clean regex
    signature exists for that, but two chunks that are 90%+ the same
    content will have near-identical embeddings regardless of what's
    glued onto either end.
    """
    keep_idx = []
    kept_embeddings = []
    for i, emb in enumerate(embeddings):
        is_dup = False
        for kept in kept_embeddings:
            sim = float(emb @ kept) / (np.linalg.norm(emb) * np.linalg.norm(kept) + 1e-9)
            if sim >= threshold:
                is_dup = True
                break
        if not is_dup:
            keep_idx.append(i)
            kept_embeddings.append(emb)

    dropped = len(chunks) - len(keep_idx)
    if dropped:
        print(f"Dropped {dropped} near-duplicate chunk(s) (similarity >= {threshold})")

    return [chunks[i] for i in keep_idx], embeddings[keep_idx]


def cluster_embeddings(embeddings, n_clusters):
    n_clusters = min(n_clusters, len(embeddings))
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


def run(input_path, n_clusters, embedder, model_name, mode):
    chunks = get_chunks(input_path, mode)
    print(f"Embedder: {embedder}\n")

    embeddings = embed_chunks(chunks, embedder=embedder, model_name=model_name)
    chunks, embeddings = drop_near_duplicates(chunks, embeddings)

    labels = cluster_embeddings(embeddings, n_clusters)

    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    print(f"\nFormed {len(clusters)} theme clusters:\n")
    for label in sorted(clusters):
        indices = clusters[label]
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
    parser.add_argument("--input", required=True, help="Path to pasted chat text (.txt) or JSON export")
    parser.add_argument("--n-clusters", type=int, default=6, help="Number of theme clusters to form")
    parser.add_argument("--embedder", choices=["tfidf", "sbert"], default="tfidf",
                         help="tfidf = offline fallback, sbert = real semantic embeddings (needs internet)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="sentence-transformers model name (sbert only)")
    parser.add_argument("--mode", choices=["raw", "qa"], default="raw",
                         help="raw = any pasted text, no markers needed (default). qa = structured JSON export")
    args = parser.parse_args()
    run(args.input, args.n_clusters, args.embedder, args.model, args.mode)