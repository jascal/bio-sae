"""Structural annotation helpers: DSSP-style SS, contact maps, pLDDT.

These wrappers are thin enough to swap implementations (BioPython DSSP
vs. PyMol vs. mdtraj). Returned types are plain Python / numpy so the
rest of the package stays decoupled from any one structural toolkit.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


SS3 = ("H", "E", "C")           # helix, strand, coil (DSSP collapsed)
SS8 = ("H", "G", "I", "E", "B", "T", "S", "C")


def secondary_structure_from_pdb(pdb_path: str) -> list[str]:
    """Return per-residue DSSP codes (SS8). Raises if PDB unreadable."""
    raise NotImplementedError("DSSP wrapper not yet implemented")


def contact_map_from_pdb(
    pdb_path: str,
    distance_threshold_a: float = 8.0,
) -> np.ndarray:
    """Return an (L, L) bool contact map between Cα atoms."""
    raise NotImplementedError("contact map extractor not yet implemented")


def plddt_from_esmfold(
    sequence: str,
    esmfold_model: Optional[object] = None,
) -> np.ndarray:
    """Return a (L,) per-residue pLDDT confidence array in [0, 100]."""
    raise NotImplementedError("ESMFold pLDDT path not yet implemented")
