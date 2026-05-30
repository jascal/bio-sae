"""Modular experts for bio-sae ensembles (ESM-2 baseline + JEPA world models).

See :mod:`biosae.experts.base` for the :class:`Expert` contract and
:mod:`biosae.experts.jepa_expert` for the protein-native JEPA and the
Hugging Face V-JEPA 2 / LeWorldModel adapter.
"""

from biosae.experts.base import (
    Expert,
    ExpertEnsemble,
    IdentityExpert,
    Router,
)
from biosae.experts.extract import CoExtraction, coextract, offsets_from_lengths
from biosae.experts.supervised_encoder import (
    SupervisedEncoder,
    SupervisedEncoderConfig,
    train_supervised_encoder,
)
from biosae.experts.jepa_expert import (
    FlatJepaScorer,
    HFJepaBackbone,
    JepaBackendUnavailable,
    JepaConfig,
    JepaExpert,
    ProteinJEPA,
    SupervisedJepaConfig,
    mutation_action,
    train_label_jepa,
    train_protein_jepa,
)

__all__ = [
    # base
    "Expert", "IdentityExpert", "Router", "ExpertEnsemble",
    # jepa
    "JepaConfig", "ProteinJEPA", "train_protein_jepa", "JepaExpert",
    "SupervisedJepaConfig", "train_label_jepa",
    "HFJepaBackbone", "JepaBackendUnavailable", "FlatJepaScorer",
    "mutation_action",
    # supervised encoder (P1-on-ESM)
    "SupervisedEncoder", "SupervisedEncoderConfig", "train_supervised_encoder",
    # extraction
    "CoExtraction", "coextract", "offsets_from_lengths",
]
