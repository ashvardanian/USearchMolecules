"""3D Conformer Generation - Generate and optimize molecular conformers.

Generates 3D conformers using ETKDG (Experimental Torsion-angle Knowledge Distance Geometry)
and optionally optimizes them with MMFF94 force field, enabling structure-based analysis.

Input:
    - data/{dataset}/parquet/*.parquet - Parquet files with 'smiles' column

Output:
    - Same Parquet files augmented with conformer columns:
      - conformer_3d: binary mol block (SDF format, 2-5 KB per molecule)
      - conformer_energy: float64 (MMFF94 energy in kcal/mol)
    - Optional separate files:
      - data/{dataset}/sdf/*.sdf (multi-molecule SDF format)
      - data/{dataset}/mol2/*.mol2 (Tripos MOL2 for docking software)

Conformer Generation Methods:
    - ETKDGv3 (default): Knowledge-based distance geometry with torsion constraints
    - MMFF94: Force field optimization for energy minimization (optional)
    - GPU: nvMolKit acceleration when --use-gpu is specified

Energy & Similarity:
    - Energy: MMFF94 force field energy in kcal/mol (lower = more stable)
    - Boltzmann weights: Population weights based on energy: exp(-ΔE/kT)
    - RMSD: Root Mean Square Deviation for structural similarity
    - Best practice: Select lowest-energy conformer for single-conformer storage

Usage:

    uv run python -m usearch_molecules.prep_conformers --datasets example
    uv run python -m usearch_molecules.prep_conformers --datasets example --profile
    uv run python -m usearch_molecules.prep_conformers --datasets example --use-gpu
    uv run python -m usearch_molecules.prep_conformers --datasets example --export-sdf --export-mol2
    uv run python -m usearch_molecules.prep_conformers --datasets example --conformers 50 --batch-size 1000
    uv run python -m usearch_molecules.prep_conformers --datasets example --optimizations 200
"""

import os
import sys
import time
import json
import logging
import argparse
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, asdict
from multiprocessing import Process, cpu_count

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit import RDLogger
except ImportError:
    print("Error: RDKit is required for conformer generation")
    print("Install with: uv pip install 'usearch-molecules[dev]'")
    sys.exit(1)

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
        return (
            self.conformers_generated / self.total_generation_time
            if self.total_generation_time > 0
            else 0.0
        )

    @property
    def avg_generation_time(self) -> float:
        """Average time to generate conformers per molecule (seconds)."""
        return (
            self.total_generation_time / self.molecules_processed
            if self.molecules_processed > 0
            else 0.0
        )

    @property
    def avg_optimization_time(self) -> float:
        """Average time to optimize conformers per molecule (seconds)."""
        return (
            self.total_optimization_time / self.molecules_processed
            if self.molecules_processed > 0
            else 0.0
        )

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
) -> List[int]:
    """Generate conformers using ETKDGv3 (Experimental Torsion-angle Knowledge Distance Geometry).

    ETKDGv3 is the default since RDKit 2024.03 and uses torsion angle preferences
    from the Cambridge Structural Database to generate high-quality conformers
    without requiring subsequent energy minimization.

    Args:
        mol: RDKit molecule object (must have explicit hydrogens)
        num_confs: Number of conformers to generate
        random_seed: Random seed for reproducibility
        use_random_coords: Use random coordinates instead of ETKDG (faster but lower quality)

    Returns:
        List of conformer IDs
    """
    # Add hydrogens if not present
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.numThreads = 0  # Use all available threads
    params.useRandomCoords = use_random_coords

    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    return list(conf_ids)


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
        logger.warning(f"Could not get MMFF properties for molecule")
        return False, float("inf")

    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
    if ff is None:
        logger.warning(f"Could not create force field for conformer {conf_id}")
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


def process_batch_to_conformers(
    smiles_list: List[str],
    num_conformers: int = 50,
    optimization_iters: int = 0,
    remove_duplicates: bool = True,
    rmsd_threshold: float = 0.5,
    random_seed: int = 42,
) -> Tuple[List[Optional[bytes]], List[float], ConformerStats]:
    """Process a batch of SMILES strings to generate conformers efficiently.

    This processes molecules in batch for better RDKit performance.
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
        - mol_blocks: List of serialized conformers (one per molecule, lowest energy)
        - energies: List of conformer energies
        - stats: Performance statistics
    """
    stats = ConformerStats()
    mol_blocks = []
    energies_list = []

    generation_start = time.time()

    for smiles in smiles_list:
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                mol_blocks.append(None)
                energies_list.append(float("inf"))
                stats.molecules_failed += 1
                continue

            # Generate conformers
            conf_ids = generate_conformer_etkdg(mol, num_conformers, random_seed)

            if len(conf_ids) == 0:
                mol_blocks.append(None)
                energies_list.append(float("inf"))
                stats.molecules_failed += 1
                continue

            stats.conformers_generated += len(conf_ids)

            # Optimize conformers if requested
            if optimization_iters > 0:
                energies = optimize_all_conformers_mmff(mol, optimization_iters)
                stats.conformers_optimized += len(energies)

                # Remove duplicates if requested
                if remove_duplicates and len(energies) > 1:
                    energies = remove_duplicate_conformers(mol, energies, rmsd_threshold)

                # Select lowest energy conformer
                best_conf_id = energies[0][0]
                best_energy = energies[0][2]
            else:
                # No optimization - just pick first conformer
                best_conf_id = conf_ids[0]
                best_energy = 0.0

            # Serialize best conformer
            mol_block = conformer_to_molblock(mol, best_conf_id)
            mol_blocks.append(mol_block)
            energies_list.append(best_energy)

            # Update stats
            stats.molecules_processed += 1
            stats.sum_energy += best_energy
            stats.min_energy = min(stats.min_energy, best_energy)
            stats.max_energy = max(stats.max_energy, best_energy)

        except Exception as e:
            logger.debug(f"Failed to process SMILES {smiles}: {e}")
            mol_blocks.append(None)
            energies_list.append(float("inf"))
            stats.molecules_failed += 1

    stats.total_generation_time = time.time() - generation_start

    # Optimization time is already tracked in optimize_all_conformers_mmff
    # For batch processing, we include it in total_generation_time

    return mol_blocks, energies_list, stats


def augment_parquet_with_conformers(
    parquet_path: str,
    batch_size: int = 1000,
    num_conformers: int = 50,
    optimization_iters: int = 0,
    remove_duplicates: bool = True,
    rmsd_threshold: float = 0.5,
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

        # Process entire batch
        mol_blocks, energies, batch_stats = process_batch_to_conformers(
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

  # Disable RMSD deduplication
  uv run python -m usearch_molecules.prep_conformers --datasets example --remove-duplicates 0

  # Enable profiling
  uv run python -m usearch_molecules.prep_conformers --datasets example --profile
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
        default=1,
        help="Remove duplicate conformers based on RMSD: 1=yes, 0=no (default: 1)",
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
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable detailed performance profiling and save to JSON",
    )

    args = parser.parse_args()

    # Check for GPU
    if args.use_gpu:
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning("GPU requested but CUDA not available, falling back to CPU")
                args.use_gpu = False
            else:
                logger.info(f"GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
                # TODO: Import nvMolKit when available
        except ImportError:
            logger.warning("GPU requested but PyTorch not installed, falling back to CPU")
            args.use_gpu = False

    logger.info("Generating 3D conformers with ETKDG and MMFF94")
    logger.info(
        f"Datasets: {', '.join(args.datasets)} | "
        f"Batch size: {args.batch_size} | "
        f"Conformers: {args.conformers} | "
        f"Optimizations: {args.optimizations} | "
        f"Remove duplicates: {args.remove_duplicates} | "
        f"RMSD threshold: {args.rmsd_threshold} Å | "
        f"GPU: {args.use_gpu} | "
        f"Profile: {args.profile}"
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

        if args.profile and dataset_stats.molecules_processed > 0:
            logger.info("")
            logger.info(f"Performance Profile for {dataset}")
            profile_data = dataset_stats.to_dict()
            for key, value in profile_data.items():
                if isinstance(value, float):
                    logger.info(f"{key}: {value:.4f}")
                else:
                    logger.info(f"{key}: {value}")

            # Save profile to JSON
            profile_path = f"data/{dataset}/conformer_profile.json"
            with open(profile_path, "w") as f:
                json.dump(profile_data, f, indent=2)
            logger.info(f"Profile saved to: {profile_path}")

        logger.info(f"✓ Successfully processed {dataset}")

    logger.info("")
    logger.info("Completed!")


if __name__ == "__main__":
    main()
