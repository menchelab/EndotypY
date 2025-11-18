#!/usr/bin/env python3
"""
Step 3: Aggregate term embeddings per node using three methods:
1. Mean pooling
2. Weighted mean pooling (IDF-weighted)
3. Max pooling
"""

import pickle
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    print("=" * 60)
    print("Step 3: Aggregate Embeddings Per Node")
    print("=" * 60)

    # 1. Load term embeddings
    print("\n1. Loading term embeddings...")
    with open('output/term_to_embedding.pkl', 'rb') as f:
        term_to_embedding = pickle.load(f)
    print(f"   ✓ Loaded embeddings for {len(term_to_embedding):,} terms")
    embedding_dim = next(iter(term_to_embedding.values())).shape[0]
    print(f"   Embedding dimension: {embedding_dim}")

    # 2. Load term-node mappings
    print("\n2. Loading term-node mappings...")
    mapping_df = pd.read_csv('input/term_node_mapping.tsv', sep='\t')
    print(f"   Total mappings: {len(mapping_df):,}")

    # Optionally filter to only nodes in edges
    edges_df = pd.read_csv('input/edges.tsv', sep='\t')
    edge_nodeids = set(edges_df['fromnodeid'].unique()) | set(edges_df['tonodeid'].unique())
    mapping_df = mapping_df[mapping_df['nodeid'].isin(edge_nodeids)].copy()
    print(f"   Filtered to nodes in edges: {len(mapping_df):,} mappings")

    # 3. Group terms by node
    print("\n3. Grouping terms by node...")
    node_terms = mapping_df.groupby('nodeid')['Term'].apply(list).to_dict()
    print(f"   Nodes with terms: {len(node_terms):,}")

    # Calculate term frequencies for IDF weighting
    term_counts = Counter(mapping_df['Term'])
    total_nodes = len(node_terms)

    # 4. Aggregate embeddings using three methods
    print("\n4. Aggregating embeddings...")

    mean_embeddings = {}
    weighted_mean_embeddings = {}
    max_embeddings = {}
    node_stats = {}

    for nodeid, terms in tqdm(node_terms.items(), desc="   Processing nodes"):
        # Filter to terms that have embeddings
        valid_terms = [t for t in terms if t in term_to_embedding]

        if len(valid_terms) == 0:
            # Node has no valid terms - use zero vector
            zero_vec = np.zeros(embedding_dim)
            mean_embeddings[nodeid] = zero_vec
            weighted_mean_embeddings[nodeid] = zero_vec
            max_embeddings[nodeid] = zero_vec
            node_stats[nodeid] = {'total_terms': len(terms), 'valid_terms': 0}
            continue

        # Get embeddings for valid terms
        term_embeddings = np.array([term_to_embedding[t] for t in valid_terms])

        # Method 1: Mean Pooling
        mean_embeddings[nodeid] = np.mean(term_embeddings, axis=0)

        # Method 2: Weighted Mean Pooling (IDF-weighted)
        # IDF weight: log(total_nodes / term_frequency)
        weights = []
        for term in valid_terms:
            term_freq = term_counts[term]
            idf_weight = np.log(total_nodes / term_freq) if term_freq > 0 else 0
            weights.append(idf_weight)

        weights = np.array(weights)
        weights = weights / weights.sum() if weights.sum() > 0 else weights  # Normalize
        weighted_mean_embeddings[nodeid] = np.average(term_embeddings, axis=0, weights=weights)

        # Method 3: Max Pooling
        max_embeddings[nodeid] = np.max(term_embeddings, axis=0)

        # Store statistics
        node_stats[nodeid] = {
            'total_terms': len(terms),
            'valid_terms': len(valid_terms),
            'mean_weight': np.mean(weights) if len(weights) > 0 else 0
        }

    # 5. Save results
    print("\n5. Saving aggregated embeddings...")

    # Save mean pooling
    with open('output/node_embeddings_mean.pkl', 'wb') as f:
        pickle.dump(mean_embeddings, f)
    print("   ✓ Saved node_embeddings_mean.pkl")

    # Save weighted mean pooling
    with open('output/node_embeddings_weighted_mean.pkl', 'wb') as f:
        pickle.dump(weighted_mean_embeddings, f)
    print("   ✓ Saved node_embeddings_weighted_mean.pkl")

    # Save max pooling
    with open('output/node_embeddings_max.pkl', 'wb') as f:
        pickle.dump(max_embeddings, f)
    print("   ✓ Saved node_embeddings_max.pkl")

    # Save statistics
    stats_df = pd.DataFrame.from_dict(node_stats, orient='index')
    stats_df.index.name = 'nodeid'
    stats_df.to_csv('output/node_aggregation_stats.csv')
    print("   ✓ Saved node_aggregation_stats.csv")

    # Print summary statistics
    print("\n6. Summary Statistics:")
    print("-" * 60)
    print(f"   Total nodes processed: {len(node_terms):,}")
    print(f"   Nodes with valid embeddings: {sum(1 for s in node_stats.values() if s['valid_terms'] > 0):,}")
    print(f"   Average terms per node: {stats_df['total_terms'].mean():.2f}")
    print(f"   Average valid terms per node: {stats_df['valid_terms'].mean():.2f}")
    print(f"   Max terms per node: {stats_df['total_terms'].max()}")

    print("\n" + "=" * 60)
    print("Step 3 complete!")
    print("=" * 60)
    print("\nFiles created:")
    print("  - output/node_embeddings_mean.pkl: Mean pooling embeddings")
    print("  - output/node_embeddings_weighted_mean.pkl: IDF-weighted mean pooling embeddings")
    print("  - output/node_embeddings_max.pkl: Max pooling embeddings")
    print("  - output/node_aggregation_stats.csv: Statistics per node")

if __name__ == "__main__":
    main()
