"""3D Conformer Generation - Generate and optimize molecular conformers.

Generates 3D conformers using ETKDG (Experimental Torsion-angle Knowledge Distance Geometry)
and optionally optimizes them with MMFF94 force field, enabling structure-based analysis.

Input:
    - data/{dataset}/parquet/*.parquet - Parquet files with 'smiles' column

Output:
    - Same Parquet files augmented with conformer columns:
      - conformer_3d: binary mol block (SDF format, 2-5 KB per molecule)
      - conformer_energy: float64 (MMFF94 energy in kcal/mol, or 0.0 if MMFF failed)
    - Optional separate files:
      - data/{dataset}/sdf/*.sdf (multi-molecule SDF format)
      - data/{dataset}/mol2/*.mol2 (Tripos MOL2 for docking software)

Conformer Generation Methods:
    - ETKDGv3 (default): Knowledge-based distance geometry with torsion constraints
    - MMFF94: Force field optimization for energy minimization (optional, --optimizations N)
    - GPU: nvMolKit acceleration when --use-gpu is specified

MMFF94 Limitations & Fallback Behavior:
    MMFF94 force field has LIMITED COVERAGE and will fail for certain molecule types.
    This is EXPECTED BEHAVIOR, not an error.

    Unsupported molecule types:
    - Transition metals: Pd, Pt, Ir, Ru, Rh, Au, Ag, Fe, Co, Ni, Cu, Zn
    - Lanthanides/Actinides: La, Lu, Gd, Sm, Eu, U, Th
    - Unusual oxidation states: S_6+6, Mn3+2, Fe2+2, Pd3+2, etc.
    - Organometallics, coordination complexes, metallorganics

    Fallback behavior when MMFF fails:
    1. Keeps ETKDG-generated conformer (still chemically reasonable)
    2. Sets conformer_energy = 0.0 (indicates no optimization)
    3. Logs at DEBUG level (use logging.DEBUG to see details)
    4. Tracks failure in stats.molecules_failed (visible in progress bar)
    5. Continues processing remaining molecules

    Expected failure rates:
    - Drug-like molecules: ~5% (high MMFF coverage)
    - PubChem: ~15% (includes some organometallics)
    - Enamine REAL: ~30% (many metal catalysts)

    Recommendation:
    - Use --optimizations for drug discovery (high success rate)
    - Skip optimization for catalysis/materials (low success rate)

Energy & Similarity:
    - Energy: MMFF94 force field energy in kcal/mol (lower = more stable)
    - Energy = 0.0: MMFF optimization was skipped (either disabled or failed)
    - Boltzmann weights: Population weights based on energy: exp(-ΔE/kT)
    - RMSD: Root Mean Square Deviation for structural similarity
    - Best practice: Select lowest-energy conformer for single-conformer storage

Usage:

    uv run python -m usearch_molecules.prep_conformers --datasets example
    uv run python -m usearch_molecules.prep_conformers --datasets example --optimizations 200
    uv run python -m usearch_molecules.prep_conformers --datasets example --conformers 50 --batch-size 1000
    uv run python -m usearch_molecules.prep_conformers --datasets example --export-sdf --export-mol2
    pixi run python -m usearch_molecules.prep_conformers --datasets example --use-gpu

My defaults for benchmarking:

    pixi run python -m usearch_molecules.prep_conformers --datasets example --conformers 20 --batch-size 100 --optimizations 20
    pixi run python -m usearch_molecules.prep_conformers --datasets example --conformers 20 --batch-size 100 --optimizations 20 --use-gpu

With larger batches the 16-core CPU yields 7 mols/s and 70 conf/s and the H100 yields 25 mols/s and 250 conf/s:

    pixi run python -m usearch_molecules.prep_conformers --datasets example --conformers 10 --batch-size 1000 --optimizations 20
    pixi run python -m usearch_molecules.prep_conformers --datasets example --conformers 10 --batch-size 1000 --optimizations 20 --use-gpu
"""

import os
import sys
import time
import logging
import argparse
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, asdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
except ImportError:
    print("Error: RDKit is required for conformer generation")
    print("Install with: uv pip install 'usearch-molecules[dev]'")
    sys.exit(1)

# Optional GPU acceleration with nvMolKit
try:
    from nvmolkit import embedMolecules, mmffOptimization
    from nvmolkit.types import HardwareOptions

    NVMOLKIT_AVAILABLE = True
except ImportError:
    NVMOLKIT_AVAILABLE = False

# Suppress RDKit warnings about unusual atom types, charges, etc.
# These are expected for organometallics and exotic molecules in large databases
RDLogger.DisableLog("rdApp.*")

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


@dataclass
class ConformerStats:
    """Statistics for conformer generation profiling with throughput metrics."""

    molecules_processed: int = 0
    molecules_failed: int = 0
    conformers_generated: int = 0
    conformers_optimized: int = 0
    total_generation_time: float = 0.0
    total_optimization_time: float = 0.0
    total_export_time: float = 0.0
    min_energy: float = float("inf")
    max_energy: float = float("-inf")
    sum_energy: float = 0.0

    @property
    def success_rate(self) -> float:
        """Percentage of molecules successfully processed."""
        total = self.molecules_processed + self.molecules_failed
        return 100.0 * self.molecules_processed / total if total > 0 else 0.0

    @property
    def molecules_per_second(self) -> float:
        """Throughput: molecules processed per second."""
        total_time = self.total_generation_time + self.total_optimization_time
        return self.molecules_processed / total_time if total_time > 0 else 0.0

    @property
    def conformers_per_second(self) -> float:
        """Throughput: conformers generated per second."""
        return self.conformers_generated / self.total_generation_time if self.total_generation_time > 0 else 0.0

    @property
    def avg_generation_time(self) -> float:
        """Average time to generate conformers per molecule (seconds)."""
        return self.total_generation_time / self.molecules_processed if self.molecules_processed > 0 else 0.0

    @property
    def avg_optimization_time(self) -> float:
        """Average time to optimize conformers per molecule (seconds)."""
        return self.total_optimization_time / self.molecules_processed if self.molecules_processed > 0 else 0.0

    @property
    def avg_energy(self) -> float:
        """Average MMFF94 energy across all conformers (kcal/mol)."""
        return self.sum_energy / self.molecules_processed if self.molecules_processed > 0 else 0.0

    def to_dict(self) -> Dict:
        """Convert stats to dictionary for JSON export."""
        return {
            **asdict(self),
            "success_rate": self.success_rate,
            "molecules_per_second": self.molecules_per_second,
            "conformers_per_second": self.conformers_per_second,
            "avg_generation_time": self.avg_generation_time,
            "avg_optimization_time": self.avg_optimization_time,
            "avg_energy": self.avg_energy,
        }


def generate_conformer_etkdg(
    mol: Chem.Mol,
    num_confs: int = 10,
    random_seed: int = 42,
    use_random_coords: bool = False,
) -> Tuple[Chem.Mol, List[int]]:
    """Generate conformers using ETKDGv3 (Experimental Torsion-angle Knowledge Distance Geometry).

    ETKDGv3 is the default since RDKit 2024.03 and uses torsion angle preferences
    from the Cambridge Structural Database to generate high-quality conformers
    without requiring subsequent energy minimization.

    Args:
        mol: RDKit molecule object
        num_confs: Number of conformers to generate
        random_seed: Random seed for reproducibility
        use_random_coords: Use random coordinates instead of ETKDG (faster but lower quality)

    Returns:
        Tuple of (molecule with hydrogens and conformers, list of conformer IDs)
    """
    # Add hydrogens if not present (returns new molecule)
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.numThreads = 0  # Use all available threads
    params.useRandomCoords = use_random_coords

    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    return mol, list(conf_ids)


def optimize_conformer_mmff(
    mol: Chem.Mol,
    conf_id: int = -1,
    max_iters: int = 200,
) -> Tuple[bool, float]:
    """Optimize a single conformer using MMFF94 force field.

    Args:
        mol: RDKit molecule with conformers
        conf_id: Conformer ID to optimize (-1 for all conformers)
        max_iters: Maximum number of optimization iterations

    Returns:
        Tuple of (converged: bool, energy: float in kcal/mol)
    """
    # Set up MMFF94 force field
    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is None:
        # Expected for organometallics and exotic molecules - tracked in stats
        logger.debug(f"Could not get MMFF properties for molecule")
        return False, float("inf")

    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
    if ff is None:
        logger.debug(f"Could not create force field for conformer {conf_id}")
        return False, float("inf")

    # Optimize
    converged = ff.Minimize(maxIts=max_iters)
    energy = ff.CalcEnergy()

    return converged == 0, energy


def optimize_all_conformers_mmff(
    mol: Chem.Mol,
    max_iters: int = 200,
) -> List[Tuple[int, bool, float]]:
    """Optimize all conformers and return their energies.

    Args:
        mol: RDKit molecule with conformers
        max_iters: Maximum optimization iterations per conformer

    Returns:
        List of (conf_id, converged, energy) tuples sorted by energy
    """
    results = []
    for conf_id in range(mol.GetNumConformers()):
        converged, energy = optimize_conformer_mmff(mol, conf_id, max_iters)
        results.append((conf_id, converged, energy))

    # Sort by energy (lowest first)
    results.sort(key=lambda x: x[2])
    return results


def select_lowest_energy_conformer(mol: Chem.Mol, energies: List[Tuple[int, bool, float]]) -> int:
    """Select the conformer with the lowest MMFF94 energy.

    Best practice: The lowest energy conformer is most likely to represent
    the bioactive conformation for drug discovery applications.

    Args:
        mol: RDKit molecule with conformers
        energies: List of (conf_id, converged, energy) tuples

    Returns:
        Conformer ID with lowest energy
    """
    if not energies:
        return 0
    return energies[0][0]  # Already sorted by energy


def compute_boltzmann_weights(
    energies: List[float],
    temperature: float = 298.15,
) -> np.ndarray:
    """Calculate Boltzmann population weights for conformers.

    Boltzmann weighting reflects the thermodynamic population of conformers
    at a given temperature. Lower energy conformers have higher populations.

    Formula: w_i = exp(-E_i/kT) / Σ exp(-E_j/kT)

    Args:
        energies: List of conformer energies in kcal/mol
        temperature: Temperature in Kelvin (default: 298.15 K = 25°C)

    Returns:
        Array of normalized Boltzmann weights summing to 1.0
    """
    R = 0.001987204  # Gas constant in kcal/(mol·K)
    energies = np.array(energies)

    # Shift energies relative to minimum to avoid numerical overflow
    min_energy = np.min(energies)
    delta_energies = energies - min_energy

    # Calculate Boltzmann factors
    boltzmann_factors = np.exp(-delta_energies / (R * temperature))

    # Normalize to get weights
    weights = boltzmann_factors / np.sum(boltzmann_factors)
    return weights


def conformer_to_molblock(mol: Chem.Mol, conf_id: int = -1) -> bytes:
    """Serialize conformer to mol block for Parquet storage.

    Args:
        mol: RDKit molecule with conformers
        conf_id: Conformer ID to serialize (-1 for default)

    Returns:
        Mol block as bytes
    """
    mol_block = Chem.MolToMolBlock(mol, confId=conf_id)
    return mol_block.encode("utf-8")


def export_to_sdf(
    molecules: List[Tuple[str, Chem.Mol, int, float]],
    output_path: str,
):
    """Export conformers to SDF file.

    Args:
        molecules: List of (smiles, mol, conf_id, energy) tuples
        output_path: Path to output SDF file
    """
    writer = Chem.SDWriter(output_path)
    for smiles, mol, conf_id, energy in molecules:
        mol.SetProp("SMILES", smiles)
        mol.SetProp("Energy_kcal_mol", f"{energy:.4f}")
        writer.write(mol, confId=conf_id)
    writer.close()


def export_to_mol2(
    molecules: List[Tuple[str, Chem.Mol, int, float]],
    output_dir: str,
):
    """Export conformers to MOL2 files (one per molecule).

    MOL2 format is commonly used in docking software like AutoDock, GOLD, and Glide.

    Args:
        molecules: List of (smiles, mol, conf_id, energy) tuples
        output_dir: Directory to write MOL2 files
    """
    os.makedirs(output_dir, exist_ok=True)

    for idx, (smiles, mol, conf_id, energy) in enumerate(molecules):
        mol2_path = os.path.join(output_dir, f"mol_{idx:08d}.mol2")
        Chem.MolToMolFile(mol, mol2_path, confId=conf_id)


def compute_rmsd(mol: Chem.Mol, conf_id1: int, conf_id2: int) -> float:
    """Calculate RMSD between two conformers.

    Args:
        mol: RDKit molecule with conformers
        conf_id1: First conformer ID
        conf_id2: Second conformer ID

    Returns:
        RMSD in Angstroms
    """
    return AllChem.GetConformerRMS(mol, conf_id1, conf_id2, prealigned=True)


def remove_duplicate_conformers(
    mol: Chem.Mol,
    energies: List[Tuple[int, bool, float]],
    rmsd_threshold: float = 0.5,
) -> List[Tuple[int, bool, float]]:
    """Remove conformers that are too similar based on RMSD.

    Args:
        mol: RDKit molecule with conformers
        energies: List of (conf_id, converged, energy) tuples (should be sorted by energy)
        rmsd_threshold: RMSD threshold in Angstroms for duplicate detection

    Returns:
        Filtered list of unique conformers
    """
    if len(energies) <= 1:
        return energies

    unique_conformers = [energies[0]]  # Keep lowest energy conformer

    for conf_id, converged, energy in energies[1:]:
        is_duplicate = False
        for unique_id, _, _ in unique_conformers:
            rmsd = compute_rmsd(mol, conf_id, unique_id)
            if rmsd < rmsd_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_conformers.append((conf_id, converged, energy))

    return unique_conformers


def process_batch_to_conformers_gpu(
    smiles_list: List[str],
    num_conformers: int = 50,
    optimization_iters: int = 0,
    remove_duplicates: bool = True,
    rmsd_threshold: float = 0.5,
    random_seed: int = 42,
) -> Tuple[List[Optional[bytes]], List[float], ConformerStats]:
    """Process a batch of SMILES strings using GPU-accelerated nvMolKit.

    Uses nvMolKit's batch processing for GPU-accelerated conformer generation and optimization.
    Significantly faster than CPU for large batches.

    GPU Memory Requirements:
        - Small molecules (~10 heavy atoms): ~1-2 MB per conformer
        - Medium molecules (~25 heavy atoms): ~5-10 MB per conformer
        - Large molecules (~50 heavy atoms): ~20-30 MB per conformer
        - Rule of thumb: batch_size × num_conformers × 10 MB < GPU_memory
        - Example: 50 molecules × 100 conformers = 5000 conformers ≈ 50 GB

    Args:
        smiles_list: List of SMILES strings to process
        num_conformers: Number of conformers to generate per molecule
        optimization_iters: MMFF optimization iterations (0 to skip)
        remove_duplicates: Remove duplicate conformers based on RMSD
        rmsd_threshold: RMSD threshold for duplicate detection (Angstroms)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (mol_blocks, energies, stats)
        - mol_blocks: List of byte strings, one per input SMILES (None for failures)
        - energies: List of floats, one per input SMILES (inf for failures)
        - stats: Performance statistics

    Raises:
        RuntimeError: If nvMolKit is not available or GPU runs out of memory
    """
    if not NVMOLKIT_AVAILABLE:
        raise RuntimeError("nvMolKit is not available. Install it to use GPU acceleration.")

    stats = ConformerStats()
    generation_start = time.time()

    # Pre-allocate output arrays with correct length
    batch_size = len(smiles_list)
    mol_blocks: List[Optional[bytes]] = [None] * batch_size
    energies_list: List[float] = [float("inf")] * batch_size

    # Parse SMILES and prepare molecules for GPU processing
    molecules = []
    molecule_to_batch_idx = []  # Maps molecules list index → smiles_list index

    for idx, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                stats.molecules_failed += 1
                continue

            mol = Chem.AddHs(mol)
            molecules.append(mol)
            molecule_to_batch_idx.append(idx)
        except Exception:
            stats.molecules_failed += 1

    if not molecules:
        stats.total_generation_time = time.time() - generation_start
        assert len(mol_blocks) == batch_size, f"Output size mismatch: {len(mol_blocks)} != {batch_size}"
        assert len(energies_list) == batch_size, f"Energy size mismatch: {len(energies_list)} != {batch_size}"
        return mol_blocks, energies_list, stats

    # Configure GPU hardware options for optimal performance
    hardware_opts = HardwareOptions(
        preprocessingThreads=-1,  # Auto-detect CPU threads
        batchSize=-1,  # Auto-tune batch size for GPU
        batchesPerGpu=-1,  # Auto-tune concurrent batches
        gpuIds=[],  # Use all available GPUs
    )

    # Set up ETKDG parameters for nvMolKit
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.useRandomCoords = True  # Required for nvMolKit

    # Generate conformers on GPU with proper error handling
    try:
        embedMolecules.EmbedMolecules(
            molecules,
            params,
            confsPerMolecule=num_conformers,
            maxIterations=-1,
            hardwareOptions=hardware_opts,
        )
    except RuntimeError as e:
        error_msg = str(e)
        if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
            total_conformers = len(molecules) * num_conformers
            raise RuntimeError(
                f"GPU out of memory: batch_size={len(molecules)}, conformers={num_conformers}, "
                f"total={total_conformers}. Reduce --batch-size or --conformers."
            ) from e
        raise

    # Count generated conformers
    for mol in molecules:
        stats.conformers_generated += mol.GetNumConformers()

    # Optimize conformers on GPU if requested and capture energies
    # Pre-filter molecules to avoid nvMolKit crashes on MMFF-incompatible molecules
    mol_energies_list: List[List[float]] = [[] for _ in molecules]  # Initialize with empty lists
    if optimization_iters > 0:
        # Pre-check each molecule for MMFF compatibility (same check nvMolKit does internally)
        valid_indices = []
        valid_molecules = []
        invalid_indices = []

        for idx, mol in enumerate(molecules):
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is not None:
                valid_indices.append(idx)
                valid_molecules.append(mol)
            else:
                invalid_indices.append(idx)
                logger.debug(f"Molecule {idx} in batch cannot be MMFF optimized (organometallic/exotic atoms)")

        # Log MMFF filtering stats
        if invalid_indices:
            logger.info(
                f"Pre-filtered batch: {len(valid_molecules)}/{len(molecules)} molecules MMFF-compatible, "
                f"{len(invalid_indices)} will use ETKDG conformers with energy=0.0"
            )

        # Only optimize MMFF-compatible molecules on GPU
        if valid_molecules:
            try:
                # nvMolKit returns list[list[float]] - energies for each valid molecule's conformers
                valid_energies = mmffOptimization.MMFFOptimizeMoleculesConfs(
                    valid_molecules, maxIters=optimization_iters, hardwareOptions=hardware_opts
                )
                # Map results back to original molecule indices
                for valid_idx, orig_idx in enumerate(valid_indices):
                    mol_energies_list[orig_idx] = valid_energies[valid_idx]

                stats.conformers_optimized += sum(len(e) for e in valid_energies)
            except RuntimeError as e:
                error_msg = str(e)
                if "out of memory" in error_msg.lower():
                    total_conformers = len(valid_molecules) * num_conformers
                    raise RuntimeError(
                        f"GPU OOM during optimization: batch_size={len(valid_molecules)}, "
                        f"conformers={num_conformers}, total={total_conformers}. "
                        f"Reduce --batch-size or --conformers."
                    ) from e
                raise

    # Process results for each molecule
    for mol_idx, mol in enumerate(molecules):
        batch_idx = molecule_to_batch_idx[mol_idx]

        try:
            if mol.GetNumConformers() == 0:
                stats.molecules_failed += 1
                continue

            # Get energies from nvMolKit or set to 0 if no optimization
            # Note: mol_energies_list[mol_idx] will be empty for MMFF-incompatible molecules
            if optimization_iters > 0 and mol_idx < len(mol_energies_list) and mol_energies_list[mol_idx]:
                conformer_energies = mol_energies_list[mol_idx]
                # Build (conf_id, converged, energy) tuples sorted by energy
                energies = [
                    (conf_id, True, conformer_energies[conf_id])
                    for conf_id in range(min(len(conformer_energies), mol.GetNumConformers()))
                ]
                energies.sort(key=lambda x: x[2])
            else:
                # No optimization OR MMFF-incompatible - keep ETKDG conformer with energy=0
                energies = [(0, True, 0.0)]

            if not energies:
                stats.molecules_failed += 1
                continue

            # Remove duplicates if requested (CPU-based, can be slow)
            if remove_duplicates and len(energies) > 1:
                energies = remove_duplicate_conformers(mol, energies, rmsd_threshold)

            # Select lowest energy conformer
            best_conf_id = energies[0][0]
            best_energy = energies[0][2]

            # Serialize best conformer
            mol_block = conformer_to_molblock(mol, best_conf_id)
            mol_blocks[batch_idx] = mol_block
            energies_list[batch_idx] = best_energy

            # Update stats
            stats.molecules_processed += 1
            stats.sum_energy += best_energy
            stats.min_energy = min(stats.min_energy, best_energy)
            stats.max_energy = max(stats.max_energy, best_energy)

        except Exception as e:
            logger.debug(f"Failed to process molecule at batch index {batch_idx}: {e}")
            stats.molecules_failed += 1

    stats.total_generation_time = time.time() - generation_start

    # Validate output lengths match input
    assert len(mol_blocks) == batch_size, f"Output size mismatch: {len(mol_blocks)} != {batch_size}"
    assert len(energies_list) == batch_size, f"Energy size mismatch: {len(energies_list)} != {batch_size}"

    return mol_blocks, energies_list, stats


def process_batch_to_conformers(
    smiles_list: List[str],
    num_conformers: int = 50,
    optimization_iters: int = 0,
    remove_duplicates: bool = True,
    rmsd_threshold: float = 0.5,
    random_seed: int = 42,
) -> Tuple[List[Optional[bytes]], List[float], ConformerStats]:
    """Process a batch of SMILES strings to generate conformers efficiently (CPU).

    Processes molecules serially but RDKit internally uses multithreading for conformer generation.
    For batch_size=1000 and num_conformers=50, generates up to 50,000 conformers
    which are then filtered and regrouped into 1000 outputs.

    Args:
        smiles_list: List of SMILES strings to process
        num_conformers: Number of conformers to generate per molecule
        optimization_iters: MMFF optimization iterations (0 to skip)
        remove_duplicates: Remove duplicate conformers based on RMSD
        rmsd_threshold: RMSD threshold for duplicate detection (Angstroms)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (mol_blocks, energies, stats)
        - mol_blocks: List of byte strings, one per input SMILES (None for failures)
        - energies: List of floats, one per input SMILES (inf for failures)
        - stats: Performance statistics
    """
    stats = ConformerStats()
    generation_start = time.time()

    # Pre-allocate output arrays with correct length
    batch_size = len(smiles_list)
    mol_blocks: List[Optional[bytes]] = [None] * batch_size
    energies_list: List[float] = [float("inf")] * batch_size

    # Process molecules in batch (RDKit handles internal parallelism)
    for idx, smiles in enumerate(smiles_list):
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                stats.molecules_failed += 1
                continue

            # Generate conformers (returns molecule with hydrogens + conf IDs)
            mol, conf_ids = generate_conformer_etkdg(mol, num_conformers, random_seed)

            if len(conf_ids) == 0:
                stats.molecules_failed += 1
                continue

            stats.conformers_generated += len(conf_ids)

            # Optimize conformers if requested
            if optimization_iters > 0:
                energies = optimize_all_conformers_mmff(mol, optimization_iters)

                # Filter out conformers that failed MMFF (have infinite energy)
                valid_energies = [e for e in energies if e[2] != float("inf")]

                if valid_energies:
                    # Optimization succeeded for at least some conformers
                    stats.conformers_optimized += len(valid_energies)

                    # Remove duplicates if requested
                    if remove_duplicates and len(valid_energies) > 1:
                        valid_energies = remove_duplicate_conformers(mol, valid_energies, rmsd_threshold)

                    # Select lowest energy conformer
                    best_conf_id = valid_energies[0][0]
                    best_energy = valid_energies[0][2]
                else:
                    # MMFF failed for all conformers - fall back to unoptimized
                    logger.debug(f"MMFF optimization failed for {smiles}, using unoptimized conformer")
                    best_conf_id = conf_ids[0]
                    best_energy = 0.0
            else:
                # No optimization - just pick first conformer
                best_conf_id = conf_ids[0]
                best_energy = 0.0

            # Serialize best conformer
            mol_block = conformer_to_molblock(mol, best_conf_id)
            mol_blocks[idx] = mol_block
            energies_list[idx] = best_energy

            # Update stats
            stats.molecules_processed += 1
            stats.sum_energy += best_energy
            stats.min_energy = min(stats.min_energy, best_energy)
            stats.max_energy = max(stats.max_energy, best_energy)

        except Exception as e:
            logger.debug(f"Failed to process SMILES {smiles}: {e}")
            stats.molecules_failed += 1

    stats.total_generation_time = time.time() - generation_start

    # Validate output lengths match input
    assert len(mol_blocks) == batch_size, f"Output size mismatch: {len(mol_blocks)} != {batch_size}"
    assert len(energies_list) == batch_size, f"Energy size mismatch: {len(energies_list)} != {batch_size}"

    return mol_blocks, energies_list, stats


def augment_parquet_with_conformers(
    parquet_path: str,
    batch_size: int = 1000,
    num_conformers: int = 50,
    optimization_iters: int = 0,
    remove_duplicates: bool = True,
    rmsd_threshold: float = 0.5,
    use_gpu: bool = False,
) -> ConformerStats:
    """Augment a single Parquet file with 3D conformers using batch processing.

    Processes molecules in batches for efficiency. With batch_size=1000 and
    num_conformers=50, RDKit generates up to 50,000 conformers per batch,
    which are filtered and regrouped into 1000 best conformers.

    Args:
        parquet_path: Path to Parquet file
        batch_size: Number of molecules to process in each batch
        num_conformers: Number of conformers to generate per molecule
        optimization_iters: MMFF optimization iterations (0 to skip)
        remove_duplicates: Remove duplicate conformers based on RMSD
        rmsd_threshold: RMSD threshold for duplicate detection (Angstroms)
        use_gpu: Use GPU-accelerated nvMolKit for conformer generation

    Returns:
        ConformerStats with performance metrics
    """
    # Check if already processed
    meta = pq.read_metadata(parquet_path)
    column_names = meta.schema.names
    if "conformer_3d" in column_names and "conformer_energy" in column_names:
        logger.info(f"Skipping {parquet_path}: already has conformers")
        return ConformerStats()

    logger.info(f"Processing {parquet_path}")
    table = pq.read_table(parquet_path)
    smiles_array = table["smiles"].to_pylist()

    conformer_3d_list = []
    conformer_energy_list = []
    total_stats = ConformerStats()

    # Track timing for throughput calculation
    overall_start = time.time()
    total_conformers_generated = 0
    molecules_processed = 0

    # Process in batches
    num_batches = (len(smiles_array) + batch_size - 1) // batch_size

    pbar = tqdm(
        total=len(smiles_array),
        desc="Generating conformers",
        unit=" mol",
        postfix={"mol/s": 0, "conf/s": 0, "failed": 0},
    )

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(smiles_array))
        batch_smiles = [str(s) for s in smiles_array[start_idx:end_idx]]

        # Process entire batch (GPU or CPU)
        process_fn = process_batch_to_conformers_gpu if use_gpu else process_batch_to_conformers
        mol_blocks, energies, batch_stats = process_fn(
            batch_smiles,
            num_conformers=num_conformers,
            optimization_iters=optimization_iters,
            remove_duplicates=remove_duplicates,
            rmsd_threshold=rmsd_threshold,
        )

        # Accumulate results
        for mol_block, energy in zip(mol_blocks, energies):
            if mol_block is None:
                conformer_3d_list.append(b"")
                conformer_energy_list.append(float("inf"))
            else:
                conformer_3d_list.append(mol_block)
                conformer_energy_list.append(energy)

        # Update stats
        total_stats.molecules_processed += batch_stats.molecules_processed
        total_stats.molecules_failed += batch_stats.molecules_failed
        total_stats.conformers_generated += batch_stats.conformers_generated
        total_stats.conformers_optimized += batch_stats.conformers_optimized
        total_stats.total_generation_time += batch_stats.total_generation_time
        total_stats.sum_energy += batch_stats.sum_energy
        total_stats.min_energy = min(total_stats.min_energy, batch_stats.min_energy)
        total_stats.max_energy = max(total_stats.max_energy, batch_stats.max_energy)

        molecules_processed += len(batch_smiles)
        total_conformers_generated += batch_stats.conformers_generated

        # Update progress bar
        pbar.update(len(batch_smiles))
        elapsed = time.time() - overall_start
        if elapsed > 0:
            pbar.set_postfix(
                {
                    "mol/s": f"{molecules_processed / elapsed:.1f}",
                    "conf/s": f"{total_conformers_generated / elapsed:.0f}",
                    "failed": total_stats.molecules_failed,
                }
            )

    pbar.close()

    # Add columns to table
    conformer_3d_array = pa.array([bytes(x) for x in conformer_3d_list], pa.binary())
    conformer_energy_array = pa.array(conformer_energy_list, pa.float64())

    conformer_3d_field = pa.field("conformer_3d", pa.binary(), nullable=True)
    conformer_energy_field = pa.field("conformer_energy", pa.float64(), nullable=True)

    table = table.append_column(conformer_3d_field, conformer_3d_array)
    table = table.append_column(conformer_energy_field, conformer_energy_array)

    # Write back to Parquet
    pq.write_table(table, parquet_path)

    return total_stats


main_epilog = """
Examples:
  # Generate conformers with defaults (ETKDGv3 only, no optimization)
  uv run python -m usearch_molecules.prep_conformers --datasets example

  # Generate many conformers with MMFF optimization
  uv run python -m usearch_molecules.prep_conformers --datasets example --conformers 50 --optimizations 200

  # Large batch processing for efficiency
  uv run python -m usearch_molecules.prep_conformers --datasets example --batch-size 2000 --conformers 100

  # Enable RMSD deduplication (disabled by default for performance)
  uv run python -m usearch_molecules.prep_conformers --datasets example --remove-duplicates 1

  # GPU acceleration
  uv run python -m usearch_molecules.prep_conformers --datasets example --use-gpu
"""


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate 3D conformers with ETKDG and MMFF94 (batch processing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=main_epilog,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["example", "pubchem", "gdb13", "real"],
        default=["example"],
        help="Which datasets to process (default: example)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of molecules to process in each batch (default: 1000)",
    )
    parser.add_argument(
        "--conformers",
        type=int,
        default=50,
        help="Number of conformers to generate per molecule (default: 50)",
    )
    parser.add_argument(
        "--optimizations",
        type=int,
        default=0,
        help="MMFF optimization iterations per conformer, 0 to skip (default: 0)",
    )
    parser.add_argument(
        "--remove-duplicates",
        type=int,
        choices=[0, 1],
        default=0,
        help="Remove duplicate conformers based on RMSD: 1=yes, 0=no (default: 0 for performance)",
    )
    parser.add_argument(
        "--rmsd-threshold",
        type=float,
        default=0.5,
        help="RMSD threshold in Angstroms for duplicate detection (default: 0.5)",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use nvMolKit GPU acceleration (requires CUDA, nvMolKit)",
    )
    parser.add_argument(
        "--export-sdf",
        action="store_true",
        help="Export conformers to SDF files",
    )
    parser.add_argument(
        "--export-mol2",
        action="store_true",
        help="Export conformers to MOL2 files (for docking software)",
    )

    args = parser.parse_args()

    # Check for GPU/CUDA availability
    if args.use_gpu:
        if not NVMOLKIT_AVAILABLE:
            logger.warning("GPU requested but nvMolKit not installed, falling back to CPU")
            args.use_gpu = False
        else:
            # Try to create a HardwareOptions object to verify CUDA works
            try:
                _ = HardwareOptions()
                logger.info("GPU acceleration enabled with nvMolKit")
            except Exception as e:
                logger.warning(f"GPU requested but CUDA initialization failed: {e}, falling back to CPU")
                args.use_gpu = False

    logger.info("Generating 3D conformers with ETKDG and MMFF94")
    logger.info(
        f"Datasets: {', '.join(args.datasets)} | "
        f"Batch size: {args.batch_size} | "
        f"Conformers: {args.conformers} | "
        f"Optimizations: {args.optimizations} | "
        f"Remove duplicates: {args.remove_duplicates} | "
        f"RMSD threshold: {args.rmsd_threshold} Å | "
        f"GPU: {args.use_gpu}"
    )

    for dataset in args.datasets:
        parquet_dir = f"data/{dataset}/parquet"
        if not os.path.exists(parquet_dir):
            logger.warning(f"Skipping {dataset}: directory {parquet_dir} not found")
            continue

        logger.info("")
        logger.info(f"Processing dataset: {dataset}")

        filenames = sorted([f for f in os.listdir(parquet_dir) if f.endswith(".parquet")])
        dataset_stats = ConformerStats()

        for filename in filenames:
            parquet_path = os.path.join(parquet_dir, filename)

            try:
                stats = augment_parquet_with_conformers(
                    parquet_path,
                    batch_size=args.batch_size,
                    num_conformers=args.conformers,
                    optimization_iters=args.optimizations,
                    remove_duplicates=bool(args.remove_duplicates),
                    rmsd_threshold=args.rmsd_threshold,
                    use_gpu=args.use_gpu,
                )

                # Accumulate stats
                dataset_stats.molecules_processed += stats.molecules_processed
                dataset_stats.molecules_failed += stats.molecules_failed
                dataset_stats.conformers_generated += stats.conformers_generated
                dataset_stats.conformers_optimized += stats.conformers_optimized
                dataset_stats.total_generation_time += stats.total_generation_time
                dataset_stats.total_optimization_time += stats.total_optimization_time
                dataset_stats.sum_energy += stats.sum_energy
                dataset_stats.min_energy = min(dataset_stats.min_energy, stats.min_energy)
                dataset_stats.max_energy = max(dataset_stats.max_energy, stats.max_energy)

            except Exception as e:
                logger.error(f"Failed to process {filename}: {e}", exc_info=True)
                raise

        logger.info(f"✓ Successfully processed {dataset}")

    logger.info("")
    logger.info("Completed!")


if __name__ == "__main__":
    main()
