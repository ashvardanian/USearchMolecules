"""USearchMolecules - Large Multi-Modal Chem-Informatics dataset for drug discovery.

This package builds and queries multi-modal representations of 7B+ small
molecules: binary fingerprints (MACCS, PubChem, ECFP4, FCFP4) and 3D conformer
geometries, indexed with USearch for real-time retrieval at billion scale.
"""

from importlib.metadata import PackageNotFoundError, version

from usearchmolecules.dataset import (
    FingerprintedDataset,
    FingerprintedEntry,
    FingerprintedShard,
    SHARD_SIZE,
    BATCH_SIZE,
)
from usearchmolecules.to_fingerprint import (
    smiles_to_maccs_ecfp4_fcfp4,
    FingerprintShape,
    shape_maccs,
    shape_mixed,
)

try:
    __version__ = version("usearchmolecules")
except PackageNotFoundError:  # running from a source checkout without an install
    __version__ = "0.0.0"

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
