"""USearch Molecules - Large Chem-Informatics dataset for drug discovery.

This package provides tools for exploring and searching through a dataset of
7B+ molecules with binary fingerprints (MACCS, PubChem, ECFP4, FCFP4).
"""

from usearch_molecules.dataset import (
    FingerprintedDataset,
    FingerprintedEntry,
    FingerprintedShard,
    SHARD_SIZE,
    BATCH_SIZE,
)
from usearch_molecules.to_fingerprint import (
    smiles_to_maccs_ecfp4_fcfp4,
    FingerprintShape,
    shape_maccs,
    shape_mixed,
)

__version__ = "1.0.0"
__author__ = "Ash Vardanian"
__email__ = "ash.vardanian@unum.cloud"

__all__ = [
    "FingerprintedDataset",
    "FingerprintedEntry",
    "FingerprintedShard",
    "SHARD_SIZE",
    "BATCH_SIZE",
    "smiles_to_maccs_ecfp4_fcfp4",
    "FingerprintShape",
    "shape_maccs",
    "shape_mixed",
]
