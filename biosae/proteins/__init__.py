"""Protein data loading + ESM-2 activation extraction."""

from biosae.proteins.datasets import ProteinRecord, load_dataset_mix


def __getattr__(name: str):
    # Lazy import: keeps torch off the critical path for synthetic-only flows.
    if name == "EsmExtractor":
        from biosae.proteins.esm_extract import EsmExtractor
        return EsmExtractor
    raise AttributeError(name)


__all__ = ["ProteinRecord", "load_dataset_mix", "EsmExtractor"]
