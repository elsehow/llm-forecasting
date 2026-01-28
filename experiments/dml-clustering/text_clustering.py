#!/usr/bin/env python3
"""
Text-based clustering of prediction market questions.

Tests whether semantic similarity (embeddings) produces interpretable clusters,
as an alternative to price comovement which failed (6.8% coverage).

Usage:
    python text_clustering.py --k 10 --output results/text_clusters_k10.json
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity


def load_questions(path: str) -> list[dict]:
    """Load market questions from JSON."""
    with open(path) as f:
        data = json.load(f)

    questions = []
    for m in data:
        # Handle both formats
        if isinstance(m, dict):
            q_id = m.get("condition_id", m.get("id", ""))
            q_text = m.get("question", m.get("title", ""))
            questions.append({"id": q_id, "question": q_text})

    return questions


def embed_questions(questions: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Embed questions using sentence transformer."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(questions, show_progress_bar=True)


def cluster_embeddings(embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Cluster embeddings using hierarchical clustering."""
    similarity = cosine_similarity(embeddings)
    distance = 1 - similarity
    np.fill_diagonal(distance, 0)
    # Clip small negative values from floating point errors
    distance = np.clip(distance, 0, 2)
    condensed = squareform(distance)
    Z = linkage(condensed, method="ward")
    labels = fcluster(Z, k, criterion="maxclust")
    return labels, similarity


def extract_keywords(questions: list[str], top_n: int = 10) -> list[tuple[str, int]]:
    """Extract top keywords from questions."""
    stopwords = {
        "will", "the", "a", "an", "in", "on", "at", "by", "to", "of", "for",
        "be", "is", "are", "was", "were", "win", "and", "or", "not", "have",
        "has", "been", "being", "before", "after", "than", "that", "this",
        "next", "first", "end", "day", "one", "new", "get", "between"
    }
    words = []
    for q in questions:
        tokens = re.findall(r'\b[a-zA-Z]+\b', q.lower())
        words.extend([t for t in tokens if t not in stopwords and len(t) > 2])
    return Counter(words).most_common(top_n)


def compute_cluster_coherence(
    embeddings: np.ndarray,
    labels: np.ndarray,
    similarity: np.ndarray
) -> dict[int, float]:
    """Compute mean intra-cluster similarity for each cluster."""
    coherence = {}
    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        if sum(mask) < 2:
            coherence[int(cluster_id)] = 1.0
            continue
        cluster_sim = similarity[mask][:, mask]
        # Mean of upper triangle (excluding diagonal)
        n = cluster_sim.shape[0]
        upper_tri = cluster_sim[np.triu_indices(n, k=1)]
        coherence[int(cluster_id)] = float(np.mean(upper_tri))
    return coherence


def main():
    parser = argparse.ArgumentParser(description="Text-based clustering of market questions")
    parser.add_argument("--markets", default="../conditional-forecasting/data/markets.json",
                        help="Path to markets.json")
    parser.add_argument("--k", type=int, default=10, help="Number of clusters")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence transformer model")
    parser.add_argument("--output", default="results/text_clusters.json", help="Output path")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    markets_path = script_dir / args.markets
    output_path = script_dir / args.output

    # Load questions
    print(f"Loading questions from {markets_path}...")
    questions = load_questions(markets_path)
    texts = [q["question"] for q in questions]
    print(f"  Loaded {len(texts)} questions")

    # Embed
    print(f"\nEmbedding questions with {args.model}...")
    embeddings = embed_questions(texts, args.model)
    print(f"  Embedding shape: {embeddings.shape}")

    # Cluster
    print(f"\nClustering with k={args.k}...")
    labels, similarity = cluster_embeddings(embeddings, args.k)

    # Compute coherence
    coherence = compute_cluster_coherence(embeddings, labels, similarity)

    # Analyze clusters
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[int(label)].append({
            "id": questions[i]["id"],
            "question": questions[i]["question"]
        })

    # Print results
    print(f"\n{'='*70}")
    print("CLUSTER LABELING WORKSHEET")
    print(f"{'='*70}")
    print("\nFor each cluster, review questions and assign a label.")
    print("Success criterion: 5+ clusters labelable, >80% of questions fit label.\n")

    results = {
        "k": args.k,
        "model": args.model,
        "n_questions": len(texts),
        "clusters": {}
    }

    for cluster_id in sorted(clusters.keys()):
        members = clusters[cluster_id]
        cluster_texts = [m["question"] for m in members]
        keywords = extract_keywords(cluster_texts)
        coh = coherence[cluster_id]

        print(f"\n--- Cluster {cluster_id} ({len(members)} questions, coherence={coh:.3f}) ---")
        print(f"Top keywords: {', '.join([f'{w}({c})' for w, c in keywords[:7]])}")
        print("\nSample questions:")
        for m in members[:8]:
            q = m["question"]
            print(f"  - {q[:75]}{'...' if len(q) > 75 else ''}")

        print(f"\n  LABEL: _________________")
        print(f"  FIT SCORE (0-100%): ___")

        results["clusters"][str(cluster_id)] = {
            "n_members": len(members),
            "coherence": coh,
            "keywords": [{"word": w, "count": c} for w, c in keywords],
            "members": members,
        }

    # Overall stats
    mean_coherence = np.mean(list(coherence.values()))
    print(f"\n{'='*70}")
    print(f"OVERALL STATS")
    print(f"{'='*70}")
    print(f"Mean cluster coherence: {mean_coherence:.3f}")
    print(f"Cluster sizes: {[len(clusters[i]) for i in sorted(clusters.keys())]}")

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results["mean_coherence"] = mean_coherence
    results["cluster_sizes"] = [len(clusters[i]) for i in sorted(clusters.keys())]

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
