import numpy as np
import pandas as pd

from EndotypY.kl_clustering import compute_feature_matrix, kl_clustering_endotypes


def two_group_feature_matrix():
    """30 genes in two groups of 15; each group is enriched in its own half of 40 terms."""
    rng = np.random.default_rng(0)
    a = rng.random((15, 40)) < 0.15
    a[:, :20] |= rng.random((15, 20)) < 0.6
    b = rng.random((15, 40)) < 0.15
    b[:, 20:] |= rng.random((15, 20)) < 0.6
    return pd.DataFrame(np.vstack([a, b]).astype(int), index=[f"G{i}" for i in range(30)])


def test_compute_feature_matrix():
    go_terms = {"A": ["t1", "t2"], "B": ["t2"], "C": []}
    fm = compute_feature_matrix(go_terms)

    assert list(fm.index) == ["A", "B"]  # C has no terms and is dropped
    assert fm.loc["A", "t1"] == 1 and fm.loc["A", "t2"] == 1
    assert fm.loc["B", "t1"] == 0 and fm.loc["B", "t2"] == 1


def test_kl_clustering_recovers_two_groups():
    data = two_group_feature_matrix()
    predicted_labels, cluster_assignments, cluster_dict = kl_clustering_endotypes(data)

    assert len(cluster_dict) == 2
    found = sorted(sorted(genes) for genes in cluster_dict.values())
    expected = sorted(sorted(f"G{i}" for i in group) for group in (range(15), range(15, 30)))
    assert found == expected
    assert set(predicted_labels) == set(data.index)
