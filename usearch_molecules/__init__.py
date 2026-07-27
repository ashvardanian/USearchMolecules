"""USearchMolecules - Large Multi-Modal Chem-Informatics dataset for drug discovery.

This package builds and queries multi-modal representations of 7B+ small
molecules: binary fingerprints (MACCS, PubChem, ECFP4, FCFP4) and 3D conformer
geometries, indexed with USearch for real-time retrieval at billion scale.
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
