#!/usr/bin/env python3
"""
Analyze and visualize the distribution of terms per node.
Creates a histogram chart and lists the top 10 nodes with the most terms.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("=" * 60)
    print("Analyzing Term Distribution per Node")
    print("=" * 60)
    
    # 1. Load term-node mappings
    print("\n1. Loading term_node_mapping.tsv...")
    mapping_df = pd.read_csv('input/term_node_mapping.tsv', sep='\t')
    print(f"   Total mappings: {len(mapping_df):,}")
    print(f"   Unique nodes: {mapping_df['nodeid'].nunique():,}")
    print(f"   Unique terms: {mapping_df['Term'].nunique():,}")
    
    # 2. Count terms per node
    print("\n2. Counting terms per node...")
    terms_per_node = mapping_df.groupby('nodeid')['Term'].nunique().sort_values(ascending=False)
    
    print(f"   Nodes with terms: {len(terms_per_node):,}")
    print(f"   Average terms per node: {terms_per_node.mean():.2f}")
    print(f"   Median terms per node: {terms_per_node.median():.2f}")
    print(f"   Min terms per node: {terms_per_node.min()}")
    print(f"   Max terms per node: {terms_per_node.max()}")
    print(f"   Std dev: {terms_per_node.std():.2f}")
    
    # 3. Top 10 nodes with most terms
    print("\n3. Top 10 nodes with most terms:")
    print("-" * 60)
    top_10 = terms_per_node.head(10)
    for i, (nodeid, count) in enumerate(top_10.items(), 1):
        print(f"   {i:2d}. Node {nodeid}: {count:,} terms")
    
    # 4. Create distribution chart
    print("\n4. Creating distribution chart...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(terms_per_node.values, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Number of Terms per Node', fontsize=12)
    ax1.set_ylabel('Number of Nodes', fontsize=12)
    ax1.set_title('Distribution of Terms per Node (Histogram)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(terms_per_node.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {terms_per_node.mean():.1f}')
    ax1.axvline(terms_per_node.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {terms_per_node.median():.1f}')
    ax1.legend()
    
    # Log scale histogram (for better visualization if distribution is skewed)
    ax2.hist(terms_per_node.values, bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax2.set_xlabel('Number of Terms per Node', fontsize=12)
    ax2.set_ylabel('Number of Nodes (log scale)', fontsize=12)
    ax2.set_title('Distribution of Terms per Node (Log Scale)', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(terms_per_node.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {terms_per_node.mean():.1f}')
    ax2.axvline(terms_per_node.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {terms_per_node.median():.1f}')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save chart
    output_file = 'output/term_distribution_chart.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved chart to {output_file}")
    
    # 5. Create summary statistics table
    print("\n5. Creating summary statistics...")
    
    # Percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("\n   Percentiles:")
    for p in percentiles:
        value = np.percentile(terms_per_node.values, p)
        print(f"     {p:2d}th percentile: {value:.1f} terms")
    
    # Distribution bins
    print("\n   Distribution by bins:")
    bins = [0, 1, 5, 10, 20, 50, 100, float('inf')]
    bin_labels = ['1', '2-5', '6-10', '11-20', '21-50', '51-100', '100+']
    for i in range(len(bins) - 1):
        count = ((terms_per_node >= bins[i]) & (terms_per_node < bins[i+1])).sum()
        pct = (count / len(terms_per_node)) * 100
        print(f"     {bin_labels[i]:>6s} terms: {count:6,} nodes ({pct:5.2f}%)")
    
    # Save top 10 to CSV
    top_10_df = pd.DataFrame({
        'nodeid': top_10.index,
        'num_terms': top_10.values
    })
    top_10_file = 'output/top_10_nodes_by_terms.csv'
    top_10_df.to_csv(top_10_file, index=False)
    print(f"\n   ✓ Saved top 10 nodes to {top_10_file}")
    
    # Save full distribution to CSV
    distribution_df = pd.DataFrame({
        'nodeid': terms_per_node.index,
        'num_terms': terms_per_node.values
    }).sort_values('num_terms', ascending=False)
    distribution_file = 'output/terms_per_node_distribution.csv'
    distribution_df.to_csv(distribution_file, index=False)
    print(f"   ✓ Saved full distribution to {distribution_file}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {output_file}: Distribution chart")
    print(f"  - {top_10_file}: Top 10 nodes with most terms")
    print(f"  - {distribution_file}: Full distribution (all nodes)")

if __name__ == "__main__":
    main()

