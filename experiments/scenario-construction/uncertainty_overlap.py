#!/usr/bin/env python3
"""
Uncertainty axis overlap experiment.
Tests whether uncertainty axes cluster across targets.
"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def main():
    results_dir = Path(__file__).parent / "results"
    targets = [
        ("democrat_whitehouse_2028", "dual_v7_20260116_172017.json"),
        ("us_gdp_2029", "dual_v7_20260116_171305.json"),
        ("carney_pm_2027", "dual_v7_20260116_172007.json"),
    ]

    # 1. Load uncertainties
    all_axes = []
    for target, filename in targets:
        with open(results_dir / target / filename) as f:
            data = json.load(f)
        for u in data.get("uncertainties", []):
            all_axes.append({
                "target": target,
                "name": u["name"],
                "description": u["description"],
                "text": f"{u['name']}: {u['description']}"
            })

    print(f"Loaded {len(all_axes)} uncertainty axes from {len(targets)} targets\n")

    # 2. Embed
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [a["text"] for a in all_axes]
    embeddings = model.encode(texts)

    # 3. Compute similarity matrix
    sim = cosine_similarity(embeddings)

    print("=" * 60)
    print("SIMILARITY MATRIX")
    print("=" * 60)

    # Print axis names for reference
    for i, a in enumerate(all_axes):
        print(f"{i}: [{a['target'][:8]}] {a['name']}")

    print("\nPairwise similarities (top cross-target pairs):")
    pairs = []
    for i in range(len(all_axes)):
        for j in range(i+1, len(all_axes)):
            if all_axes[i]["target"] != all_axes[j]["target"]:
                pairs.append((i, j, sim[i,j]))

    pairs.sort(key=lambda x: -x[2])
    for i, j, s in pairs[:10]:
        print(f"  {s:.3f}: [{all_axes[i]['target'][:8]}] {all_axes[i]['name']}")
        print(f"         [{all_axes[j]['target'][:8]}] {all_axes[j]['name']}")
        print()

    # 4. Check for natural clusters
    print("=" * 60)
    print("CLUSTER ANALYSIS")
    print("=" * 60)

    # Try multiple thresholds
    for threshold in [0.6, 0.5, 0.4]:
        groups = []
        used = set()

        for i in range(len(all_axes)):
            if i in used:
                continue
            group = [i]
            used.add(i)
            for j in range(i+1, len(all_axes)):
                if j not in used and sim[i,j] >= threshold:
                    group.append(j)
                    used.add(j)
            groups.append(group)

        cross_target_groups = [g for g in groups if len(set(all_axes[i]["target"] for i in g)) >= 2]

        print(f"\n--- Threshold {threshold} ---")
        print(f"Groups: {len(groups)}, Cross-target: {len(cross_target_groups)}")

        if cross_target_groups:
            for g, group in enumerate(cross_target_groups):
                targets_in_group = set(all_axes[i]["target"] for i in group)
                print(f"  Cross-target group ({len(targets_in_group)} targets):")
                for i in group:
                    print(f"    - [{all_axes[i]['target'][:8]}] {all_axes[i]['name']}")

    # Check success criteria
    high_sim_pairs = [p for p in pairs if p[2] > 0.6]
    print(f"\nSUCCESS CRITERIA:")
    print(f"  Cross-target pairs with sim > 0.6: {len(high_sim_pairs)} (threshold: ≥2)")
    print(f"  Cross-target groups: {len(cross_target_groups)} (threshold: ≥1)")

if __name__ == "__main__":
    main()
