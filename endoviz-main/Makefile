.PHONY: clean all pipeline all3d pipeline3d neighbors2d neighbors3d graph export analyze help

# Default target
help:
	@echo "Available targets:"
	@echo "  make all       - Run full term-based pipeline (2D)"
	@echo "  make pipeline  - Run full term-based pipeline (2D, same as 'all')"
	@echo "  make all3d     - Run full term-based pipeline (3D)"
	@echo "  make pipeline3d - Run full term-based pipeline (3D, same as 'all3d')"
	@echo "  make neighbors2d - Run full pipeline with term-overlap neighbors (2D)"
	@echo "  make neighbors3d - Run full pipeline with term-overlap neighbors (3D)"
	@echo "  make graph     - Run graph-based embeddings pipeline"
	@echo "  make export    - Export embeddings for transfer"
	@echo "  make analyze   - Analyze term distribution per node"
	@echo "  make clean     - Remove all output files"
	@echo ""
	@echo "Individual steps (2D):"
	@echo "  make sanity    - Run data validation"
	@echo "  make terms     - Filter terms"
	@echo "  make embed     - Create term embeddings"
	@echo "  make aggregate - Aggregate embeddings per node"
	@echo "  make reduce    - Reduce to 2D coordinates"
	@echo "  make merge     - Merge and add degrees (2D)"
	@echo ""
	@echo "Individual steps (3D):"
	@echo "  make reduce3d  - Reduce to 3D coordinates"
	@echo "  make merge3d   - Merge and add degrees (3D)"
	@echo ""
	@echo "Individual steps (with neighbors):"
	@echo "  make mergenb   - Merge with neighbors (2D, uses 05b)"
	@echo "  make mergenb3d - Merge with neighbors (3D, uses 05b)"

# Ensure directories exist
output:
	mkdir -p output

input:
	mkdir -p input

# Full term-based pipeline (2D)
all: pipeline

pipeline: output input sanity terms embed aggregate reduce merge
	@echo ""
	@echo "============================================================"
	@echo "2D Pipeline complete! Output files in output/ directory"
	@echo "============================================================"

# Full term-based pipeline (3D)
all3d: pipeline3d

pipeline3d: output input sanity terms embed aggregate reduce3d merge3d
	@echo ""
	@echo "============================================================"
	@echo "3D Pipeline complete! Output files in output/ directory"
	@echo "============================================================"

# Full pipeline with neighbors (2D)
neighbors2d: output input sanity terms embed aggregate reduce mergenb
	@echo ""
	@echo "============================================================"
	@echo "2D Pipeline with neighbors complete! Output files in output/ directory"
	@echo "============================================================"

# Full pipeline with neighbors (3D)
neighbors3d: output input sanity terms embed aggregate reduce3d mergenb3d
	@echo ""
	@echo "============================================================"
	@echo "3D Pipeline with neighbors complete! Output files in output/ directory"
	@echo "============================================================"

# Graph-based pipeline
graph: output input
	@echo "Running graph-based embeddings pipeline..."
	python graph_based_embeddings.py

# Export for transfer
export: output
	@echo "Exporting embeddings..."
	python export_embeddings.py

# Analyze term distribution
analyze: output input
	@echo "Analyzing term distribution..."
	python analyze_term_distribution.py

# Individual pipeline steps
sanity: input
	@echo "Step 1: Running sanity checks..."
	python sanity.py

terms: output input
	@echo "Step 2: Filtering terms..."
	python 01_get_terms.py

embed: output
	@echo "Step 3: Creating term embeddings..."
	python 02_create_embeddings_for_terms.py

aggregate: output input
	@echo "Step 4: Aggregating embeddings..."
	python 03_aggregate_embeddings.py

reduce: output
	@echo "Step 5: Reducing to 2D coordinates..."
	python 04_reduce.py --dimensions 2

reduce3d: output
	@echo "Step 5: Reducing to 3D coordinates..."
	python 04_reduce.py --dimensions 3

merge: output input
	@echo "Step 6: Merging and adding degrees (2D)..."
	python 05_merge_and_add_degree_concat_terms_parquet.py --dimensions 2

merge3d: output input
	@echo "Step 6: Merging and adding degrees (3D)..."
	python 05_merge_and_add_degree_concat_terms_parquet.py --dimensions 3

# Merge with neighbors (using 05b)
mergenb: output input
	@echo "Step 6: Merging and adding degrees with neighbors (2D)..."
	python 05b_merge_and_degree.py --dimensions 2

mergenb3d: output input
	@echo "Step 6: Merging and adding degrees with neighbors (3D)..."
	python 05b_merge_and_degree.py --dimensions 3

# Clean output directory
clean:
	@echo "Cleaning output directory..."
	rm -rf output/*
	@echo "Output directory cleaned!"

