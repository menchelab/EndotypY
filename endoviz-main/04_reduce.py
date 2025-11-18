#!/usr/bin/env python3
"""
Step 4: Reduce node embeddings to 2D or 3D coordinates using UMAP and PaCMAP.
Applies dimensionality reduction to each aggregation method (mean, weighted_mean, max).
"""

import argparse
import pickle

import numpy as np
import pandas as pd
import pacmap
from umap import UMAP


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Reduce node embeddings to 2D or 3D coordinates using UMAP and PaCMAP')
    parser.add_argument('--dimensions', type=int, default=2, choices=[2, 3],
                        help='Number of dimensions for reduction (2 or 3, default: 2)')
    args = parser.parse_args()

    n_dimensions = args.dimensions
    dim_str = f"{n_dimensions}D"

    print("=" * 60)
    print(f"Step 4: Reduce Embeddings to {dim_str}")
    print("=" * 60)

    # 1. Load aggregated embeddings
    print("\n1. Loading aggregated node embeddings...")

    with open('output/node_embeddings_mean.pkl', 'rb') as f:
        mean_embeddings = pickle.load(f)
    print(f"   ✓ Loaded mean pooling embeddings: {len(mean_embeddings):,} nodes")

    with open('output/node_embeddings_weighted_mean.pkl', 'rb') as f:
        weighted_mean_embeddings = pickle.load(f)
    print(f"   ✓ Loaded weighted mean pooling embeddings: {len(weighted_mean_embeddings):,} nodes")

    with open('output/node_embeddings_max.pkl', 'rb') as f:
        max_embeddings = pickle.load(f)
    print(f"   ✓ Loaded max pooling embeddings: {len(max_embeddings):,} nodes")

    # Get embedding dimension
    embedding_dim = next(iter(mean_embeddings.values())).shape[0]
    print(f"   Embedding dimension: {embedding_dim}")

    # Ensure all dictionaries have the same nodes
    all_nodeids = set(mean_embeddings.keys()) | set(weighted_mean_embeddings.keys()) | set(max_embeddings.keys())
    nodeids_sorted = sorted(all_nodeids)
    print(f"   Total unique nodes: {len(nodeids_sorted):,}")

    # 2. Convert to matrices (maintaining nodeid order)
    print("\n2. Converting embeddings to matrices...")

    mean_matrix = np.array([mean_embeddings.get(nid, np.zeros(embedding_dim)) for nid in nodeids_sorted])
    wmean_matrix = np.array([weighted_mean_embeddings.get(nid, np.zeros(embedding_dim)) for nid in nodeids_sorted])
    max_matrix = np.array([max_embeddings.get(nid, np.zeros(embedding_dim)) for nid in nodeids_sorted])

    print(f"   Mean matrix shape: {mean_matrix.shape}")
    print(f"   Weighted mean matrix shape: {wmean_matrix.shape}")
    print(f"   Max matrix shape: {max_matrix.shape}")

    # 3. Apply UMAP dimensionality reduction
    print(f"\n3. Applying UMAP dimensionality reduction to {n_dimensions}D...")
    print("   (This may take a few minutes)")
    
    # UMAP parameters
    n_neighbors = min(15, len(nodeids_sorted) - 1)  # Adjust if too few nodes
    min_dist = 0.1
    
    print("\n   Reducing mean pooling embeddings with UMAP...")
    umap_mean = UMAP(n_components=n_dimensions, random_state=42, n_neighbors=n_neighbors, min_dist=min_dist)
    coords_u_mean = umap_mean.fit_transform(mean_matrix)
    
    print("   Reducing weighted mean pooling embeddings with UMAP...")
    umap_wmean = UMAP(n_components=n_dimensions, random_state=42, n_neighbors=n_neighbors, min_dist=min_dist)
    coords_u_wmean = umap_wmean.fit_transform(wmean_matrix)
    
    print("   Reducing max pooling embeddings with UMAP...")
    umap_max = UMAP(n_components=n_dimensions, random_state=42, n_neighbors=n_neighbors, min_dist=min_dist)
    coords_u_max = umap_max.fit_transform(max_matrix)
    
    print("   ✓ UMAP reduction complete")
    
    # 4. Apply PaCMAP dimensionality reduction
    print(f"\n4. Applying PaCMAP dimensionality reduction to {n_dimensions}D...")
    print("   (This may take a few minutes)")
    
    print("\n   Reducing mean pooling embeddings with PaCMAP...")
    pacmap_mean = pacmap.PaCMAP(n_components=n_dimensions, verbose=True, n_neighbors=None)
    coords_p_mean = pacmap_mean.fit_transform(mean_matrix)
    
    print("   Reducing weighted mean pooling embeddings with PaCMAP...")
    pacmap_wmean = pacmap.PaCMAP(n_components=n_dimensions, verbose=True, n_neighbors=None)
    coords_p_wmean = pacmap_wmean.fit_transform(wmean_matrix)
    
    print("   Reducing max pooling embeddings with PaCMAP...")
    pacmap_max = pacmap.PaCMAP(n_components=n_dimensions, verbose=True, n_neighbors=None)
    coords_p_max = pacmap_max.fit_transform(max_matrix)
    
    print("   ✓ PaCMAP reduction complete")

    # 5. Create output dataframe
    print("\n5. Creating output dataframe...")
    
    # Build dataframe dynamically based on dimensions
    df_dict = {'nodeid': nodeids_sorted}
    
    # Add UMAP coordinates for each method (u = umap)
    for method, coords in [('mean', coords_u_mean), ('wmean', coords_u_wmean), ('max', coords_u_max)]:
        df_dict[f'x_u_{method}'] = coords[:, 0]
        df_dict[f'y_u_{method}'] = coords[:, 1]
        if n_dimensions == 3:
            df_dict[f'z_u_{method}'] = coords[:, 2]
    
    # Add PaCMAP coordinates for each method (p = pacmap)
    for method, coords in [('mean', coords_p_mean), ('wmean', coords_p_wmean), ('max', coords_p_max)]:
        df_dict[f'x_p_{method}'] = coords[:, 0]
        df_dict[f'y_p_{method}'] = coords[:, 1]
        if n_dimensions == 3:
            df_dict[f'z_p_{method}'] = coords[:, 2]
    
    coords_df = pd.DataFrame(df_dict)
    
    print(f"   ✓ Created dataframe with {len(coords_df):,} nodes")
    print(f"   Columns: {list(coords_df.columns)}")

    # 6. Save results
    print(f"\n6. Saving {dim_str} coordinates...")

    output_file = f'output/node_coordinates_{dim_str.lower()}.tsv'
    coords_df.to_csv(output_file, sep='\t', index=False)
    print(f"   ✓ Saved to {output_file}")

    # Also save as CSV for easy viewing
    output_file_csv = f'output/node_coordinates_{dim_str.lower()}.csv'
    coords_df.to_csv(output_file_csv, index=False)
    print(f"   ✓ Saved to {output_file_csv}")

    # 7. Print summary statistics
    print("\n7. Summary Statistics:")
    print("-" * 60)
    
    # UMAP statistics
    print("\n   UMAP coordinates:")
    for method in ['mean', 'wmean', 'max']:
        x_col = f'x_u_{method}'
        y_col = f'y_u_{method}'
        x_range = coords_df[x_col].max() - coords_df[x_col].min()
        y_range = coords_df[y_col].max() - coords_df[y_col].min()
        
        print(f"\n     {method.upper()} pooling:")
        print(f"       X range: [{coords_df[x_col].min():.4f}, {coords_df[x_col].max():.4f}] (span: {x_range:.4f})")
        print(f"       Y range: [{coords_df[y_col].min():.4f}, {coords_df[y_col].max():.4f}] (span: {y_range:.4f})")
        print(f"       X std: {coords_df[x_col].std():.4f}")
        print(f"       Y std: {coords_df[y_col].std():.4f}")
        
        if n_dimensions == 3:
            z_col = f'z_u_{method}'
            z_range = coords_df[z_col].max() - coords_df[z_col].min()
            print(f"       Z range: [{coords_df[z_col].min():.4f}, {coords_df[z_col].max():.4f}] (span: {z_range:.4f})")
            print(f"       Z std: {coords_df[z_col].std():.4f}")
    
    # PaCMAP statistics
    print("\n   PaCMAP coordinates:")
    for method in ['mean', 'wmean', 'max']:
        x_col = f'x_p_{method}'
        y_col = f'y_p_{method}'
        x_range = coords_df[x_col].max() - coords_df[x_col].min()
        y_range = coords_df[y_col].max() - coords_df[y_col].min()
        
        print(f"\n     {method.upper()} pooling:")
        print(f"       X range: [{coords_df[x_col].min():.4f}, {coords_df[x_col].max():.4f}] (span: {x_range:.4f})")
        print(f"       Y range: [{coords_df[y_col].min():.4f}, {coords_df[y_col].max():.4f}] (span: {y_range:.4f})")
        print(f"       X std: {coords_df[x_col].std():.4f}")
        print(f"       Y std: {coords_df[y_col].std():.4f}")
        
        if n_dimensions == 3:
            z_col = f'z_p_{method}'
            z_range = coords_df[z_col].max() - coords_df[z_col].min()
            print(f"       Z range: [{coords_df[z_col].min():.4f}, {coords_df[z_col].max():.4f}] (span: {z_range:.4f})")
            print(f"       Z std: {coords_df[z_col].std():.4f}")

    print("\n" + "=" * 60)
    print("Step 4 complete!")
    print("=" * 60)
    print(f"\nOutput file: {output_file}")
    print("Columns:")
    print("  - nodeid: Node identifier")
    coord_desc = f"{n_dimensions}D coordinates"
    
    print(f"\n  UMAP coordinates (u = umap):")
    if n_dimensions == 2:
        print(f"    - x_u_mean, y_u_mean: {coord_desc} from mean pooling")
        print(f"    - x_u_wmean, y_u_wmean: {coord_desc} from weighted mean pooling")
        print(f"    - x_u_max, y_u_max: {coord_desc} from max pooling")
    else:
        print(f"    - x_u_mean, y_u_mean, z_u_mean: {coord_desc} from mean pooling")
        print(f"    - x_u_wmean, y_u_wmean, z_u_wmean: {coord_desc} from weighted mean pooling")
        print(f"    - x_u_max, y_u_max, z_u_max: {coord_desc} from max pooling")
    
    print(f"\n  PaCMAP coordinates (p = pacmap):")
    if n_dimensions == 2:
        print(f"    - x_p_mean, y_p_mean: {coord_desc} from mean pooling")
        print(f"    - x_p_wmean, y_p_wmean: {coord_desc} from weighted mean pooling")
        print(f"    - x_p_max, y_p_max: {coord_desc} from max pooling")
    else:
        print(f"    - x_p_mean, y_p_mean, z_p_mean: {coord_desc} from mean pooling")
        print(f"    - x_p_wmean, y_p_wmean, z_p_wmean: {coord_desc} from weighted mean pooling")
        print(f"    - x_p_max, y_p_max, z_p_max: {coord_desc} from max pooling")

if __name__ == "__main__":
    main()
