"""Top-level entrypoint for assembling the bio-sae ground-truth bundle.

Thin re-export so scripts/build_protein_data.py has a single import line
that mirrors econ-sae's `from econsae.ground_truth import build_feature_matrix`.
"""

from biosae.labels.feature_matrix import FeatureMatrices, build_feature_matrices

__all__ = ["FeatureMatrices", "build_feature_matrices"]
