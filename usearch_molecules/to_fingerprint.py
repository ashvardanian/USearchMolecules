"""Molecular fingerprint generation using RDKit and Chemistry Development Kit.

Converts SMILES strings into binary fingerprint representations for similarity search:
- MACCS: 166-bit structural keys
- ECFP4: Extended-Connectivity Fingerprints (radius 2, 2048 bits)
- FCFP4: Functional-Class Fingerprints (radius 2, 2048 bits)
- PubChem: 881-bit fingerprints via CDK

Learn more:
- SMILES notation: https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system
- RDKit fingerprints: https://www.rdkit.org/docs/GettingStartedInPython.html#fingerprinting-and-molecular-similarity
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np


try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys
    from jpype import isJVMStarted, startJVM, getDefaultJVMPath, JPackage
except ImportError:
    print("Can't fingerprint molecules without RDKit and JPype")


def molecule_to_maccs(x):
    """Generate MACCS keys (166-bit structural fingerprint) from RDKit molecule."""
    return MACCSkeys.GenMACCSKeys(x)


def molecule_to_ecfp4(x):
    """Generate ECFP4 fingerprint (Morgan, radius=2, 2048 bits) from RDKit molecule."""
    return AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048)


def molecule_to_fcfp4(x):
    """Generate FCFP4 fingerprint (Morgan with features, radius=2, 2048 bits) from RDKit molecule."""
    return AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048, useFeatures=True)


def smiles_to_maccs_ecfp4_fcfp4(
    smiles: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert SMILES to packed binary fingerprints (MACCS, ECFP4, FCFP4).

    Args:
        smiles: SMILES string representation of molecule

    Returns:
        Tuple of three packed numpy arrays (MACCS: 21 bytes, ECFP4: 256 bytes, FCFP4: 256 bytes)
    """
    molecule = Chem.MolFromSmiles(smiles)
    return (
        np.packbits(molecule_to_maccs(molecule)),
        np.packbits(molecule_to_ecfp4(molecule)),
        np.packbits(molecule_to_fcfp4(molecule)),
    )


_cdk = None
_cdk_smiles_parser = None
_cdk_fingerprinter = None


def smiles_to_pubchem(smiles: str) -> Tuple[np.ndarray]:
    """Convert SMILES to PubChem fingerprint using Chemistry Development Kit (CDK).

    Initializes JVM on first call and caches CDK components for subsequent calls.

    Args:
        smiles: SMILES string representation of molecule

    Returns:
        Tuple containing single packed numpy array (PubChem: 111 bytes from 881 bits)
    """
    global _cdk
    global _cdk_smiles_parser
    global _cdk_fingerprinter

    if not isJVMStarted():
        cdk_path = os.path.join(os.getcwd(), "cdk-2.2.jar")
        startJVM(getDefaultJVMPath(), "-Djava.class.path=%s" % cdk_path)
        _cdk = JPackage("org").openscience.cdk

    if _cdk_smiles_parser is None:
        _cdk_smiles_parser = _cdk.smiles.SmilesParser(_cdk.DefaultChemObjectBuilder.getInstance())

    if _cdk_fingerprinter is None:
        _cdk_fingerprinter = _cdk.fingerprint.PubchemFingerprinter(
            _cdk.silent.SilentChemObjectBuilder.getInstance()
        )

    molecule = _cdk_smiles_parser.parseSmiles(smiles)
    cdk_fingerprint = _cdk_fingerprinter.getBitFingerprint(molecule)
    cdk_set_bits = list(cdk_fingerprint.getSetbits())
    bitset = np.zeros(881, dtype=np.uint8)
    bitset[cdk_set_bits] = 1
    bitset = np.packbits(bitset)
    return (bitset,)


@dataclass
class FingerprintShape:
    """Represents the shape of a hybrid fingerprint, potentially containing multiple concatenated bit-vectors."""

    include_maccs: bool = False
    include_ecfp4: bool = False
    include_fcfp4: bool = False
    nbytes_padding: int = 0

    @property
    def nbytes(self) -> int:
        return (
            self.include_maccs * 21
            + self.nbytes_padding
            + self.include_ecfp4 * 256
            + self.include_fcfp4 * 256
        )

    @property
    def nbits(self) -> int:
        return self.nbytes * 8

    @property
    def index_name(self) -> str:
        parts = ["index"]
        if self.include_maccs:
            parts.append("maccs")
        if self.include_ecfp4:
            parts.append("ecfp4")
        if self.include_fcfp4:
            parts.append("fcfp4")
        return "-".join(parts) + ".usearch"


shape_maccs = FingerprintShape(
    include_maccs=True,
    nbytes_padding=3,
)

shape_mixed = FingerprintShape(
    include_maccs=True,
    include_ecfp4=True,
    nbytes_padding=3,
)
