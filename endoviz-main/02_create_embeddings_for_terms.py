#!/usr/bin/env python3
"""
Step 1: Embed each term from terms_filtered.tsv using sentence-transformers.
Uses all-mpnet-base-v2 model for high-quality embeddings.
"""

import pickle

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def main():
    print("=" * 60)
    print("Step 1: Embedding Terms")
    print("=" * 60)

    # Read terms
    print("\n1. Reading terms_filtered.tsv...")
    terms_df = pd.read_csv('output/terms_filtered.tsv', sep='\t')
    terms = terms_df['Term'].tolist()
    print(f"   Total unique terms: {len(terms):,}")

    # Load the model
    print("\n2. Loading sentence-transformers/all-mpnet-base-v2 model...")
    print("   (This may take a moment on first run - model will be downloaded)")
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    print(f"   ✓ Model loaded (embedding dimension: {model.get_sentence_embedding_dimension()})")

    # Embed all terms
    print("\n3. Embedding terms...")
    print("   This may take a few minutes depending on the number of terms...")
    embeddings = model.encode(
        terms,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )

    print(f"\n   ✓ Embedded {len(embeddings):,} terms")
    print(f"   Embedding shape: {embeddings.shape}")
    print(f"   Embedding dimension: {embeddings.shape[1]}")

    # Create mapping dictionary
    term_to_embedding = dict(zip(terms, embeddings))

    # Save embeddings
    print("\n4. Saving embeddings...")

    # Save as numpy array with term list
    np.save('output/term_embeddings.npy', embeddings)
    print("   ✓ Saved term_embeddings.npy")

    # Save term list (to maintain order)
    with open('output/terms_list.pkl', 'wb') as f:
        pickle.dump(terms, f)
    print("   ✓ Saved terms_list.pkl")

    # Save mapping dictionary (for easy lookup)
    with open('output/term_to_embedding.pkl', 'wb') as f:
        pickle.dump(term_to_embedding, f)
    print("   ✓ Saved term_to_embedding.pkl")

    # Also save as CSV with first few dimensions for inspection
    # (full embeddings are too large for CSV)
    print("\n5. Creating summary CSV (first 5 dimensions for inspection)...")
    summary_df = pd.DataFrame({
        'Term': terms,
        'dim_0': embeddings[:, 0],
        'dim_1': embeddings[:, 1],
        'dim_2': embeddings[:, 2],
        'dim_3': embeddings[:, 3],
        'dim_4': embeddings[:, 4],
    })
    summary_df.to_csv('output/term_embeddings_summary.csv', index=False)
    print("   ✓ Saved term_embeddings_summary.csv (first 5 dimensions)")

    print("\n" + "=" * 60)
    print("Step 1 complete!")
    print("=" * 60)
    print("\nFiles created:")
    print("  - output/term_embeddings.npy: Full embedding matrix (numpy array)")
    print("  - output/terms_list.pkl: List of terms in same order as embeddings")
    print("  - output/term_to_embedding.pkl: Dictionary mapping term -> embedding")
    print("  - output/term_embeddings_summary.csv: First 5 dimensions for inspection")

if __name__ == "__main__":
    main()
