# Endoviz: Node Embedding and Visualization Pipeline

Pipeline for embedding Reactome pathway terms and generating 2D coordinates for network nodes using semantic embeddings and dimensionality reduction.

## Overview

This pipeline processes network data (nodes and edges) with Reactome pathway annotations, creates semantic embeddings of pathway terms, aggregates them per node, reduces to 2D coordinates, and enriches with network statistics.

## Directory Structure

- **`input/`**: Input data files
  - `edges.tsv`: Network edges with `fromnodeid` and `tonodeid` columns
  - `nodes.tsv`: Node metadata with `nodeid` and `Endotype` columns
  - `term_node_mapping.tsv`: Mapping of Reactome pathway terms to node IDs (`Term` and `nodeid` columns)

- **`output/`**: All generated files (intermediate and final outputs)

## Input Files

Place the following files in the `input/` directory:

- **`edges.tsv`**: Network edges with `fromnodeid` and `tonodeid` columns
- **`nodes.tsv`**: Node metadata with `nodeid` and `Endotype` columns
- **`term_node_mapping.tsv`**: Mapping of Reactome pathway terms to node IDs (`Term` and `nodeid` columns)

## Python Scripts

### Data Validation & Preparation

- **`sanity.py`**: Validates data consistency across files
  - Checks for missing nodes, orphaned nodes, and coverage statistics
  - Compares node IDs across edges, nodes, and term mappings

- **`01_get_terms.py`**: Filters term mappings to only include nodes present in edges
  - Outputs `terms_filtered.tsv` with unique terms

### Embedding Pipeline

- **`02_create_embeddings_for_terms.py`**: Embeds pathway terms using sentence transformers
  - Uses `sentence-transformers/all-mpnet-base-v2` model
  - Outputs: `term_embeddings.npy`, `term_to_embedding.pkl`, `terms_list.pkl`

- **`03_aggregate_embeddings.py`**: Aggregates term embeddings per node using three methods
  - Mean pooling: Simple average of term embeddings
  - Weighted mean pooling: IDF-weighted average (rare terms weighted higher)
  - Max pooling: Element-wise maximum across term embeddings
  - Outputs: `node_embeddings_mean.pkl`, `node_embeddings_weighted_mean.pkl`, `node_embeddings_max.pkl`

- **`04_reduce.py`**: Reduces node embeddings to 2D or 3D coordinates using UMAP
  - Accepts `--dimensions` parameter (2 or 3, default: 2)
  - Normalizes embeddings before reduction
  - Applies UMAP to each aggregation method
  - **2D output**: `node_coordinates_2d.tsv`/`.csv` with columns: `x_mean`, `y_mean`, `x_wmean`, `y_wmean`, `x_max`, `y_max`
  - **3D output**: `node_coordinates_3d.tsv`/`.csv` with columns: `x_mean`, `y_mean`, `z_mean`, `x_wmean`, `y_wmean`, `z_wmean`, `x_max`, `y_max`, `z_max`

- **`05_merge_and_add_degree_concat_terms_parquet.py`**: Final enrichment step
  - Accepts `--dimensions` parameter (2 or 3, default: 2) to load corresponding coordinate file
  - Calculates node degrees (in-degree, out-degree, total)
  - Merges Endotype from `nodes.tsv`
  - Adds comma-separated list of all terms per node
  - **2D output**: `nodes_out_2d.parquet` with 2D coordinates
  - **3D output**: `nodes_out_3d.parquet` with 3D coordinates

- **`05b_merge_and_degree.py`**: Variant with term overlap-based nearest neighbors
  - Accepts `--dimensions` parameter (2 or 3, default: 2) to load corresponding coordinate file
  - Calculates node degrees (in-degree, out-degree, total)
  - Merges Endotype from `nodes.tsv`
  - Adds comma-separated list of all terms per node
  - Computes 10 nearest neighbors based on Jaccard similarity of pathway terms
  - Neighbors stored as `{"ids": [indices], "distances": [distances]}` format
  - Neighbor IDs are numerical indices (0-based) into the node list
  - **2D output**: `nodes_out_2d.parquet` with neighbors column and 2D coordinates
  - **3D output**: `nodes_out_3d.parquet` with neighbors column and 3D coordinates

### Graph-Based Embeddings (Alternative Approach)

- **`graph_based_embeddings.py`**: Complete pipeline using node2vec for graph-based embeddings
  - Builds network graph from edges
  - Generates embeddings using node2vec (captures network structure directly)
  - Reduces to 2D coordinates using UMAP
  - Calculates degrees, merges Endotype and terms
  - All-in-one script: single file that does everything
  - Outputs: `nodes_graph_embeddings.parquet`
  - **Advantage**: Captures network topology directly, not dependent on pathway terms

### Data Export

- **`export_embeddings.py`**: Export embeddings and metadata for transfer to other systems
  - Loads weighted mean embeddings from `node_embeddings_weighted_mean.pkl`
  - Merges with Endotype and concatenated terms
  - Exports to portable parquet format
  - Outputs: `export_embeddings.parquet` (main file) and `export_embeddings_array.npz` (numpy format)
  - **Use case**: Transfer data to another computer/app for downstream analysis

## Output Files

### Intermediate Files (in `output/` directory)
- `terms_filtered.tsv`: Filtered unique terms
- `term_embeddings.npy`: Full embedding matrix for all terms
- `term_to_embedding.pkl`: Dictionary mapping term → embedding vector
- `node_embeddings_*.pkl`: Aggregated embeddings per node (3 methods)
- `node_coordinates_2d.tsv`/`.csv`: 2D coordinates for visualization
- `node_coordinates_3d.tsv`/`.csv`: 3D coordinates for visualization (when using `--dimensions 3`)

### Final Output (in `output/` directory)

- **`nodes_out_2d.parquet`**: Complete node dataset (term-based embeddings, 2D) with:
  - `nodeid`: Node identifier
  - `Endotype`: Node classification (from nodes.tsv)
  - `degree`, `in_degree`, `out_degree`: Network connectivity metrics
  - `terms`: Comma-separated list of all Reactome pathway terms
  - `x_mean`, `y_mean`: 2D coordinates (mean pooling)
  - `x_wmean`, `y_wmean`: 2D coordinates (weighted mean pooling)
  - `x_max`, `y_max`: 2D coordinates (max pooling)
  - **Note**: If generated by `05b_merge_and_degree.py`, also includes `neighbors` column:
    - `neighbors`: Dictionary with `{"ids": [indices], "distances": [distances]}`
    - Neighbor IDs are numerical indices (0-based) into the node list
    - Distances are based on term overlap (1 - Jaccard similarity)

- **`nodes_out_3d.parquet`**: Complete node dataset (term-based embeddings, 3D) with:
  - Same columns as `nodes_out_2d.parquet` but with 3D coordinates:
  - `x_mean`, `y_mean`, `z_mean`: 3D coordinates (mean pooling)
  - `x_wmean`, `y_wmean`, `z_wmean`: 3D coordinates (weighted mean pooling)
  - `x_max`, `y_max`, `z_max`: 3D coordinates (max pooling)

- **`nodes_graph_embeddings.parquet`**: Complete node dataset (graph-based embeddings) with:
  - `nodeid`: Node identifier
  - `Endotype`: Node classification (from nodes.tsv)
  - `degree`, `in_degree`, `out_degree`: Network connectivity metrics
  - `terms`: Comma-separated list of all Reactome pathway terms
  - `x`, `y`: 2D coordinates from graph-based embeddings

- **`export_embeddings.parquet`**: Portable export file with:
  - `nodeid`: Node identifier
  - `Endotype`: Node classification
  - `terms`: Comma-separated Reactome pathway terms
  - `embedding`: Embedding vector (as list, convert to numpy array when reading)

## Workflow

```
1. sanity.py                    → Validate data consistency
2. get_terms.py                 → Filter terms to edge nodes
3. 02_create_embeddings_for_terms.py → Embed pathway terms
4. 03_aggregate_embeddings.py  → Aggregate embeddings per node
5. 04_reduce.py                 → Reduce to 2D coordinates
6. 05_merge_and_add_degree_concat_terms_parquet.py → Final enrichment
```

## Dependencies

### Using uv (Recommended)

**Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
pip install uv
```

**Create virtual environment and install dependencies:**
```bash
# Method 1: Using uv sync (recommended - creates venv and installs automatically)
uv sync                    # Core dependencies only
uv sync --extra graph      # Include graph-based dependencies
uv sync --extra graph --extra dev  # Include dev dependencies

# Method 2: Manual venv creation
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .                    # Core dependencies
uv pip install -e ".[graph]"           # With graph dependencies
uv pip install -e ".[graph,dev]"       # With dev dependencies
```

**Using the environment:**
```bash
# After uv sync, activate the environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or use uv run to run commands in the environment automatically
uv run python sanity.py
uv run make all
```

### Using pip

**Term-Based Pipeline:**
```bash
pip install pandas numpy scikit-learn sentence-transformers umap-learn pyarrow tqdm
```

**Graph-Based Pipeline:**
```bash
pip install pandas numpy scikit-learn sentence-transformers umap-learn pyarrow tqdm networkx node2vec
```

**Note**: Requires NumPy < 2.2 for UMAP compatibility:
```bash
pip install "numpy<2.2"
```

### Dependencies List

**Core dependencies** (required):
- `pandas>=1.5.0`
- `numpy>=1.26.0,<2.2` (UMAP compatibility)
- `scikit-learn>=1.3.0`
- `sentence-transformers>=2.2.0`
- `umap-learn>=0.5.0`
- `pyarrow>=10.0.0`
- `tqdm>=4.65.0`

**Graph-based dependencies** (optional):
- `networkx>=3.0`
- `node2vec>=0.5.0`

## Usage

### Quick Start with Makefile

**Run full pipeline (2D - default):**
```bash
make all
# or
make pipeline
```

**Run full pipeline (3D):**
```bash
make all3d
# or
make pipeline3d
```

**Run graph-based embeddings:**
```bash
make graph
```

**Export embeddings:**
```bash
make export
```

**Clean output directory:**
```bash
make clean
```

**Run individual steps (2D):**
```bash
make sanity    # Data validation
make terms     # Filter terms
make embed     # Create embeddings
make aggregate # Aggregate per node
make reduce    # Reduce to 2D
make merge     # Final merge (2D)
```

**Run individual steps (3D):**
```bash
make reduce3d  # Reduce to 3D
make merge3d   # Final merge (3D)
```