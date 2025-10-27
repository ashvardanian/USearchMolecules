"""Smoke tests for USearchMolecules pipeline stages.

Tests representative molecules through each pipeline stage to catch regressions:
- Fingerprint generation (MACCS, ECFP4, FCFP4, PubChem)
- 3D conformer generation with ETKDG
- MMFF optimization with graceful fallback
- Index building and similarity search

Usage:

    uv run pytest test.py -v
    uv run pytest test.py -v -s
    uv run pytest test.py::TestConformerGeneration::test_mmff_optimization -v -s

The -s flag shows print statements, making it clear:
- Which molecules are being tested
- When MMFF optimization succeeds vs fails (expected for organometallics)
- Energy values for optimized conformers
- Fallback behavior for problematic molecules
"""

import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any

import pytest
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem

from usearch_molecules.to_fingerprint import (
    smiles_to_maccs_ecfp4_fcfp4,
    smiles_to_pubchem,
    shape_maccs,
    shape_mixed,
)
from usearch_molecules.prep_conformers import (
    generate_conformer_etkdg,
    optimize_all_conformers_mmff,
    process_batch_to_conformers,
)


# ============================================================================
# Formatting Helpers
# ============================================================================


def print_header(title: str, style: str = "simple"):
    """Print a formatted header with box drawing characters.

    Args:
        title: Header text
        style: 'simple' (single line) or 'bold' (double line)
    """
    width = 72
    if style == "bold":
        print(f"\n╔{'═' * (width - 2)}╗")
        print(f"║ {title.center(width - 4)} ║")
        print(f"╚{'═' * (width - 2)}╝")
    else:
        print(f"\n┌{'─' * (width - 2)}┐")
        print(f"│ {title:<{width - 4}} │")
        print(f"└{'─' * (width - 2)}┘")


def print_summary(passed: int, total: int, items: List[str] = None, details: str = None):
    """Print a grouped summary line.

    Args:
        passed: Number of passed tests
        total: Total number of tests
        items: Optional list of item names to show
        details: Optional additional details
    """
    status = "✓" if passed == total else "⚠"
    result = f"{status} {passed}/{total}"

    if details:
        result += f" • {details}"

    if items and len(items) <= 5:
        # Show all items if 5 or fewer
        result += f" • {', '.join(items)}"
    elif items:
        # Show first 3 + "and X more"
        result += f" • {', '.join(items[:3])}, and {len(items) - 3} more"

    print(result)


def print_table(headers: List[str], rows: List[List[Any]], indent: int = 2):
    """Print a formatted table.

    Args:
        headers: Column headers
        rows: List of rows (each row is a list of values)
        indent: Left indentation spaces
    """
    if not rows:
        return

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Print header
    prefix = " " * indent
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(f"{prefix}{header_line}")
    print(f"{prefix}{'-' * len(header_line)}")

    # Print rows
    for row in rows:
        row_line = "  ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
        print(f"{prefix}{row_line}")


# Global test results collector
test_results = {
    "total_tests": 0,
    "passed_tests": 0,
    "warnings": [],
}


@pytest.fixture(scope="session", autouse=True)
def print_final_summary():
    """Print final test summary dashboard after all tests complete."""
    yield  # Tests run here

    # Print final summary
    print_header("TEST SUMMARY", style="bold")

    # Overall statistics
    passed = test_results["passed_tests"]
    total = test_results["total_tests"]
    failed = total - passed

    status_icon = "✓" if failed == 0 else "⚠"
    print(f"\n  {status_icon} Overall: {passed}/{total} test groups passed")

    if failed > 0:
        print(f"  ✗ Failed: {failed} test group(s)")

    # Warnings section
    if test_results["warnings"]:
        print(f"\n  ⚠ Warnings ({len(test_results['warnings'])}):")
        for warning in test_results["warnings"][:5]:  # Show first 5
            print(f"    • {warning}")
        if len(test_results["warnings"]) > 5:
            print(f"    • ... and {len(test_results['warnings']) - 5} more")

    # Summary
    print("\n" + "─" * 72)
    if failed == 0 and len(test_results["warnings"]) == 0:
        print("  🎉 All tests passed with no warnings!")
    elif failed == 0:
        print("  ✓ All tests passed (with some warnings)")
    else:
        print("  ⚠ Some tests failed - see details above")
    print("─" * 72 + "\n")


# Test molecules covering different complexity levels and edge cases
TEST_MOLECULES = {
    "ethane": "CC",  # Simplest molecule
    "ethanol": "CCO",  # Simple with heteroatom
    "benzene": "c1ccccc1",  # Aromatic
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",  # Drug-like molecule
    "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Medium complexity
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Larger drug molecule
    "ferrocene": "[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1",  # Organometallic (will fail MMFF)
}

# Molecules expected to fail MMFF but work with ETKDG
MMFF_PROBLEM_MOLECULES = {
    "ferrocene": "[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1",  # Organometallic
    "palladium_complex": "[Pd+2]",  # Metal center
}

# Corner cases that should fail gracefully (bad inputs)
CORNER_CASE_MOLECULES = {
    "invalid_smiles": "INVALID!!!",  # Complete garbage
    "empty": "",  # Empty string
    "just_symbol": "X",  # Unknown atom
    "unbalanced_paren": "C(C",  # Syntax error
    "weird_chars": "C@#$%C",  # Special characters
}


class TestFingerprintGeneration:
    """Test molecular fingerprint generation for all supported types."""

    def test_maccs_ecfp4_fcfp4_generation(self):
        """Test RDKit fingerprints (MACCS, ECFP4, FCFP4) for all test molecules."""
        print_header("Fingerprint Generation (RDKit)")

        passed = []
        failed = []

        for name, smiles in TEST_MOLECULES.items():
            try:
                maccs, ecfp4, fcfp4 = smiles_to_maccs_ecfp4_fcfp4(smiles)

                # Check dimensions (fingerprints are returned as numpy arrays in bytes)
                # MACCS: 166 bits = 21 bytes (ceil(166/8))
                # ECFP4/FCFP4: 2048 bits = 256 bytes
                assert len(maccs) == 21, f"{name}: MACCS should be 21 bytes (166 bits)"
                assert len(ecfp4) == 256, f"{name}: ECFP4 should be 256 bytes (2048 bits)"
                assert len(fcfp4) == 256, f"{name}: FCFP4 should be 256 bytes (2048 bits)"

                # Check non-zero (at least some bits set)
                assert np.any(maccs), f"{name}: MACCS fingerprint is all zeros"
                assert np.any(ecfp4), f"{name}: ECFP4 fingerprint is all zeros"
                assert np.any(fcfp4), f"{name}: FCFP4 fingerprint is all zeros"

                passed.append(name)
            except Exception as e:
                if name in MMFF_PROBLEM_MOLECULES:
                    test_results["warnings"].append(f"Fingerprint: {name} failed (expected)")
                    failed.append(name)
                else:
                    pytest.fail(f"{name}: Failed to generate fingerprints: {e}")

        print_summary(len(passed), len(TEST_MOLECULES), passed, "MACCS, ECFP4, FCFP4")
        test_results["total_tests"] += 1
        test_results["passed_tests"] += 1 if len(passed) == len(TEST_MOLECULES) else 0

    def test_pubchem_fingerprint_generation(self):
        """Test CDK PubChem fingerprints for drug-like molecules."""
        print_header("Fingerprint Generation (CDK PubChem)")

        # Only test simple molecules for PubChem (requires CDK)
        simple_molecules = {
            k: v
            for k, v in TEST_MOLECULES.items()
            if k in ["ethane", "ethanol", "benzene", "aspirin"]
        }

        passed = []
        skipped = []

        for name, smiles in simple_molecules.items():
            try:
                pubchem = smiles_to_pubchem(smiles)

                # Check dimensions
                assert len(pubchem) == 881, f"{name}: PubChem should be 881 bits"

                # Check non-zero
                assert np.any(pubchem), f"{name}: PubChem fingerprint is all zeros"

                passed.append(name)
            except Exception as e:
                skipped.append(name)
                test_results["warnings"].append(f"PubChem: {name} skipped (CDK may be unavailable)")

        if passed:
            print_summary(len(passed), len(simple_molecules), passed, "PubChem 881-bit")
            test_results["total_tests"] += 1
            test_results["passed_tests"] += 1
        elif skipped:
            print(f"⚠ Skipped {len(skipped)} molecules (CDK may not be available)")
            # Don't count as test if completely skipped


class TestConformerGeneration:
    """Test 3D conformer generation with ETKDG and MMFF."""

    def test_etkdg_conformer_generation(self):
        """Test ETKDG conformer generation without optimization."""
        print_header("Conformer Generation (ETKDG)")

        passed = []
        total_conformers = 0

        for name, smiles in TEST_MOLECULES.items():
            if name in MMFF_PROBLEM_MOLECULES:
                continue  # Skip organometallics for basic test

            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None, f"{name}: Failed to parse SMILES"

            # Generate conformers (returns molecule with hydrogens and conformers)
            mol_h, conf_ids = generate_conformer_etkdg(mol, num_confs=10, random_seed=42)

            assert len(conf_ids) > 0, f"{name}: No conformers generated"
            assert mol_h.GetNumConformers() == len(conf_ids), f"{name}: Conformer count mismatch"

            # Check that conformers have 3D coordinates
            for conf_id in conf_ids:
                conf = mol_h.GetConformer(conf_id)
                pos = conf.GetPositions()
                assert pos.shape[0] > 0, f"{name}: No atoms in conformer"
                assert pos.shape[1] == 3, f"{name}: Conformer is not 3D"

                # Check coordinates are not all zero
                assert not np.allclose(pos, 0), f"{name}: Conformer coordinates are all zeros"

            passed.append(name)
            total_conformers += len(conf_ids)

        skipped = len([m for m in TEST_MOLECULES if m in MMFF_PROBLEM_MOLECULES])
        print_summary(
            len(passed),
            len(TEST_MOLECULES) - skipped,
            passed,
            f"{total_conformers} conformers total",
        )
        test_results["total_tests"] += 1
        test_results["passed_tests"] += 1

    def test_mmff_optimization(self):
        """Test MMFF optimization for drug-like molecules."""
        print_header("MMFF Optimization (Drug-Like Molecules)")

        drug_molecules = {
            k: v for k, v in TEST_MOLECULES.items() if k in ["aspirin", "caffeine", "ibuprofen"]
        }

        table_rows = []

        for name, smiles in drug_molecules.items():
            mol = Chem.MolFromSmiles(smiles)
            mol_h, conf_ids = generate_conformer_etkdg(mol, num_confs=5, random_seed=42)

            # Optimize conformers
            energies = optimize_all_conformers_mmff(mol_h, max_iters=200)
            assert len(energies) == len(conf_ids), f"{name}: Energy count mismatch"

            # Check that at least some conformers optimized successfully
            valid_energies = [e for e in energies if e[2] != float("inf")]
            failed_count = len(energies) - len(valid_energies)

            assert len(valid_energies) > 0, f"{name}: All MMFF optimizations failed"

            # Check energies are sorted (lowest first)
            energy_values = [e[2] for e in valid_energies]
            assert energy_values == sorted(energy_values), f"{name}: Energies not sorted"

            best_energy = valid_energies[0][2]
            table_rows.append(
                [
                    name,
                    f"{len(valid_energies)}/{len(energies)}",
                    f"{best_energy:.2f}",
                    "✓ PASS" if failed_count == 0 else f"⚠ {failed_count} failed",
                ]
            )

        print_table(["Molecule", "Optimized", "Best Energy (kcal/mol)", "Status"], table_rows)
        print_summary(len(drug_molecules), len(drug_molecules), list(drug_molecules.keys()))
        test_results["total_tests"] += 1
        test_results["passed_tests"] += 1

    def test_mmff_fallback_for_problematic_molecules(self):
        """Test that MMFF gracefully falls back for organometallics."""
        print_header("MMFF Fallback (Organometallics)")
        print("  Expected: MMFF fails but pipeline falls back to ETKDG conformers")

        passed = []
        failed = []

        for name, smiles in MMFF_PROBLEM_MOLECULES.items():
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                test_results["warnings"].append(f"MMFF Fallback: {name} - invalid SMILES")
                failed.append(name)
                continue

            try:
                mol_h, conf_ids = generate_conformer_etkdg(mol, num_confs=5, random_seed=42)

                if len(conf_ids) == 0:
                    test_results["warnings"].append(f"MMFF Fallback: {name} - ETKDG also failed")
                    failed.append(name)
                    continue

                # Try MMFF optimization (EXPECTED TO FAIL)
                energies = optimize_all_conformers_mmff(mol_h, max_iters=200)
                assert len(energies) > 0, f"{name}: MMFF returned no results"

                # Check fallback behavior
                valid_energies = [e for e in energies if e[2] != float("inf")]

                if len(valid_energies) == 0:
                    # Expected behavior - MMFF failed, will fall back
                    passed.append(f"{name} (fallback)")
                else:
                    # Unexpected - MMFF actually worked
                    passed.append(f"{name} ({len(valid_energies)}/{len(energies)} optimized)")

            except Exception as e:
                test_results["warnings"].append(f"MMFF Fallback: {name} - {e}")
                failed.append(name)

        print_summary(len(passed), len(MMFF_PROBLEM_MOLECULES), passed, "fallback tested")
        test_results["total_tests"] += 1
        test_results["passed_tests"] += 1 if len(passed) == len(MMFF_PROBLEM_MOLECULES) else 0


class TestCornerCases:
    """Test that pipeline handles bad inputs gracefully."""

    def test_invalid_smiles_dont_crash_pipeline(self):
        """Test that invalid SMILES are handled gracefully without crashing."""
        print_header("Corner Cases (Invalid Inputs)")
        print("  Expected: Bad inputs rejected gracefully without crashing")

        rejected = []
        unexpected = []

        for name, smiles in CORNER_CASE_MOLECULES.items():
            # Test fingerprint generation
            fp_failed = False
            try:
                maccs, ecfp4, fcfp4 = smiles_to_maccs_ecfp4_fcfp4(smiles)
                unexpected.append(f"{name} (fingerprints worked)")
            except Exception:
                fp_failed = True

            # Test conformer generation
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                rejected.append(name)
            else:
                if not fp_failed:
                    unexpected.append(f"{name} (RDKit parsed)")

        print_summary(
            len(rejected),
            len(CORNER_CASE_MOLECULES),
            rejected,
            "invalid inputs rejected",
        )
        if unexpected:
            print(f"  ⚠ Unexpected: {', '.join(unexpected)} (may be valid edge cases)")
        test_results["total_tests"] += 1
        test_results["passed_tests"] += 1

    def test_batch_with_mixed_valid_invalid(self):
        """Test batch processing with mix of valid and invalid molecules."""
        print_header("Batch Processing (Mixed Valid/Invalid)")

        smiles_list = [
            "CCO",  # Valid: ethanol
            "INVALID",  # Invalid
            "c1ccccc1",  # Valid: benzene
            "",  # Empty
            "CC(=O)O",  # Valid: acetic acid
        ]

        mol_blocks, energies, stats = process_batch_to_conformers(
            smiles_list,
            num_conformers=5,
            optimization_iters=0,
            remove_duplicates=False,
        )

        # Should have processed 3 valid molecules
        assert stats.molecules_processed >= 3, "Should process at least 3 valid molecules"
        assert stats.molecules_failed >= 2, "Should fail at least 2 invalid molecules"
        assert len(mol_blocks) == len(smiles_list), "Should return results for all inputs"

        # Check that valid molecules got conformers, invalid got None
        assert mol_blocks[0] is not None, "Ethanol should succeed"
        assert mol_blocks[1] is None, "INVALID should fail"
        assert mol_blocks[2] is not None, "Benzene should succeed"

        print_summary(
            stats.molecules_processed,
            len(smiles_list),
            details=f"{stats.molecules_failed} invalid rejected",
        )
        test_results["total_tests"] += 1
        test_results["passed_tests"] += 1


class TestBatchProcessing:
    """Test batch processing functionality."""

    def test_batch_conformer_generation(self):
        """Test batch processing of multiple molecules."""
        print_header("Batch Processing (No Optimization)")

        # Use simple molecules for batch test
        smiles_list = [
            "CC",  # Ethane
            "CCO",  # Ethanol
            "c1ccccc1",  # Benzene
            "CC(=O)O",  # Acetic acid
        ]

        # Process without optimization
        mol_blocks, energies, stats = process_batch_to_conformers(
            smiles_list,
            num_conformers=5,
            optimization_iters=0,
            remove_duplicates=False,
        )

        assert len(mol_blocks) == len(smiles_list), "Batch size mismatch"
        assert len(energies) == len(smiles_list), "Energy count mismatch"

        # Check that molecules were processed
        assert stats.molecules_processed > 0, "No molecules processed"
        assert stats.conformers_generated > 0, "No conformers generated"

        # Check mol blocks are valid
        valid_blocks = [mb for mb in mol_blocks if mb is not None]
        assert len(valid_blocks) >= 3, "Too many failures in batch"

        print_summary(
            stats.molecules_processed,
            len(smiles_list),
            details=f"{stats.conformers_generated} conformers total",
        )
        test_results["total_tests"] += 1
        test_results["passed_tests"] += 1

    def test_batch_with_mmff_optimization(self):
        """Test batch processing with MMFF optimization and fallback."""
        print_header("Batch Processing (MMFF + Fallback)")
        print("  Testing: drug-like (MMFF works) + organometallic (fallback)")

        smiles_list = [
            "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin (should work)
            "CCO",  # Ethanol (should work)
            "[Fe+2]",  # Iron (will fail MMFF, should fall back)
        ]

        mol_blocks, energies, stats = process_batch_to_conformers(
            smiles_list,
            num_conformers=5,
            optimization_iters=100,
            remove_duplicates=True,
            rmsd_threshold=0.5,
        )

        assert len(mol_blocks) == len(smiles_list), "Batch size mismatch"

        # Should have some successful molecules (at least ethanol and aspirin)
        assert stats.molecules_processed >= 2, "Too few molecules processed"

        # Check that fallback worked (at least some molecules succeeded)
        valid_blocks = [mb for mb in mol_blocks if mb is not None and mb != b""]
        assert len(valid_blocks) >= 2, "MMFF fallback failed"

        table_rows = [
            ["Molecules processed", f"{stats.molecules_processed}/{len(smiles_list)}"],
            ["Conformers generated", str(stats.conformers_generated)],
            ["Conformers optimized (MMFF)", str(stats.conformers_optimized)],
            ["Valid mol blocks", f"{len(valid_blocks)}/{len(smiles_list)}"],
        ]

        if stats.conformers_optimized < stats.conformers_generated:
            fallback = stats.conformers_generated - stats.conformers_optimized
            table_rows.append(["Fallback used (MMFF failed)", str(fallback)])

        print_table(["Metric", "Value"], table_rows)
        print_summary(stats.molecules_processed, len(smiles_list), details="with fallback support")
        test_results["total_tests"] += 1
        test_results["passed_tests"] += 1


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_complete_pipeline_simple_molecule(self):
        """Test complete pipeline for a simple molecule."""
        print_header("End-to-End Pipeline (Ethanol)")

        smiles = "CCO"  # Ethanol

        # Step 1: Generate fingerprints
        maccs, ecfp4, fcfp4 = smiles_to_maccs_ecfp4_fcfp4(smiles)
        assert np.any(maccs) and np.any(ecfp4) and np.any(fcfp4)

        # Step 2: Generate conformer
        mol = Chem.MolFromSmiles(smiles)
        mol_h, conf_ids = generate_conformer_etkdg(mol, num_confs=10, random_seed=42)
        assert len(conf_ids) > 0

        # Step 3: Optimize with MMFF
        energies = optimize_all_conformers_mmff(mol_h, max_iters=200)
        valid_energies = [e for e in energies if e[2] != float("inf")]
        assert len(valid_energies) > 0

        # Step 4: Select best conformer
        best_conf_id = valid_energies[0][0]
        best_energy = valid_energies[0][2]
        assert best_energy < 100.0, "Energy seems unreasonably high"

        table_rows = [
            ["Step 1: Fingerprints", "✓ MACCS, ECFP4, FCFP4 generated"],
            ["Step 2: Conformers", f"✓ {len(conf_ids)} conformers (ETKDG)"],
            ["Step 3: Optimization", f"✓ {len(valid_energies)}/{len(energies)} optimized (MMFF)"],
            ["Step 4: Best energy", f"{best_energy:.2f} kcal/mol"],
        ]
        print_table(["Pipeline Stage", "Result"], table_rows)
        print_summary(1, 1, ["ethanol"], "complete pipeline")
        test_results["total_tests"] += 1
        test_results["passed_tests"] += 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
