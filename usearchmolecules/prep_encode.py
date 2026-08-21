"""Fingerprint Generation - Add molecular fingerprints to Parquet files.

Augments Parquet files with molecular fingerprints computed using RDKit and CDK,
enabling similarity search and clustering operations.

Input:
    - data/{dataset}/parquet/*.parquet - Parquet files with 'smiles' column

Output:
    - Same Parquet files augmented with fingerprint columns:
      - maccs: MACCS keys (166 bits → 21 bytes)
      - ecfp4: Extended Connectivity FP diameter 4 (2048 bits → 256 bytes)
      - fcfp4: Functional Class FP diameter 4 (2048 bits → 256 bytes)
      - pubchem: PubChem fingerprints (881 bits → 111 bytes)

Usage:

    uv run python -m usearchmolecules.prep_encode
    uv run python -m usearchmolecules.prep_encode --datasets example
    uv run python -m usearchmolecules.prep_encode --datasets example --skip-cdk
    uv run python -m usearchmolecules.prep_encode --datasets example --processes 8
"""

import argparse
import logging
import os
from collections.abc import Callable
from multiprocessing import Pool, Process, cpu_count

import pyarrow as pa
import pyarrow.parquet as pq

from usearchmolecules.dataset import is_fingerprint_shard, write_table
from usearchmolecules.to_fingerprint import (
    smiles_to_maccs_ecfp4_fcfp4,
    smiles_to_pubchem,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# The column layout every subset is published with. Both augmentations append, so the order a
# shard ends up in is the order the stages happened to run — which is not a property a published
# schema should depend on. `normalize_column_order` settles it once, whatever ran first.
FINGERPRINT_COLUMNS = ("smiles", "maccs", "ecfp4", "fcfp4", "pubchem")


def normalize_column_order(parquet_path: os.PathLike):
    """Rewrite a shard so its columns sit in `FINGERPRINT_COLUMNS` order, if they do not already."""
    table: pa.Table = pq.read_table(parquet_path)
    present = [name for name in FINGERPRINT_COLUMNS if name in table.column_names]
    extra = [name for name in table.column_names if name not in FINGERPRINT_COLUMNS]
    wanted = present + extra
    if wanted == table.column_names:
        return
    logger.info(f"Reordering columns of {parquet_path}")
    write_table(table.select(wanted), parquet_path)


def _rdkit_fingerprints(batch: list[str]) -> tuple[list[bytes], list[bytes], list[bytes]]:
    """Fingerprint one batch of SMILES, substituting zeros for anything RDKit refuses."""
    maccs_list, ecfp4_list, fcfp4_list = [], [], []
    refused = 0
    for smiles in batch:
        try:
            fingers = smiles_to_maccs_ecfp4_fcfp4(smiles)
            maccs_list.append(fingers[0].tobytes())
            ecfp4_list.append(fingers[1].tobytes())
            fcfp4_list.append(fingers[2].tobytes())
        except Exception:
            maccs_list.append(bytes(21))
            ecfp4_list.append(bytes(256))
            fcfp4_list.append(bytes(256))
            refused += 1
    return maccs_list, ecfp4_list, fcfp4_list, refused


def augment_with_rdkit(parquet_path: os.PathLike, processes: int = 1):
    meta = pq.read_metadata(parquet_path)
    column_names: list[str] = meta.schema.names
    if "maccs" in column_names and "ecfp4" in column_names and "fcfp4" in column_names:
        return

    logger.info(f"Starting file {parquet_path}")
    table: pa.Table = pq.read_table(parquet_path)
    smiles_rows = table["smiles"].to_pylist()

    if processes > 1:
        stride = len(smiles_rows) // processes + 1
        batches = [smiles_rows[start : start + stride] for start in range(0, len(smiles_rows), stride)]
        with Pool(processes) as pool:
            results = pool.map(_rdkit_fingerprints, batches)
    else:
        results = [_rdkit_fingerprints(smiles_rows)]

    maccs_list = [value for result in results for value in result[0]]
    ecfp4_list = [value for result in results for value in result[1]]
    fcfp4_list = [value for result in results for value in result[2]]
    refused = sum(result[3] for result in results)
    share = 100.0 * refused / max(1, len(maccs_list))
    if share > 1.0:
        logger.warning(f"RDKit refused {refused:,} of {len(maccs_list):,} molecules ({share:.1f}%) in {parquet_path}")
    if refused == len(maccs_list):
        raise RuntimeError(f"every molecule in {parquet_path} was refused; the fingerprint columns would be all zeros")

    maccs_list = pa.array(maccs_list, pa.binary(21))
    ecfp4_list = pa.array(ecfp4_list, pa.binary(256))
    fcfp4_list = pa.array(fcfp4_list, pa.binary(256))
    maccs_field = pa.field("maccs", pa.binary(21), nullable=False)
    ecfp4_field = pa.field("ecfp4", pa.binary(256), nullable=False)
    fcfp4_field = pa.field("fcfp4", pa.binary(256), nullable=False)

    table = table.append_column(maccs_field, maccs_list)
    table = table.append_column(ecfp4_field, ecfp4_list)
    table = table.append_column(fcfp4_field, fcfp4_list)
    write_table(table, parquet_path)


def _cdk_fingerprints(batch: list[str]) -> tuple[list[bytes], int]:
    """PubChem-fingerprint one batch of SMILES, and count what CDK refused.

    A refused molecule is stored as zeros, which is the right answer for a handful and a silent
    catastrophe for all of them. The count is what tells those two apart.
    """
    values, refused = [], 0
    for smiles in batch:
        try:
            fingers = smiles_to_pubchem(smiles)
            values.append(fingers[0].tobytes())
        except Exception:
            values.append(bytes(111))
            refused += 1
    return values, refused


def augment_with_cdk(parquet_path: os.PathLike, processes: int = 1):
    meta = pq.read_metadata(parquet_path)
    column_names: list[str] = meta.schema.names
    if "pubchem" in column_names:
        return

    logger.info(f"Starting file {parquet_path}")
    table: pa.Table = pq.read_table(parquet_path)
    smiles_rows = table["smiles"].to_pylist()
    if processes > 1:
        stride = len(smiles_rows) // processes + 1
        batches = [smiles_rows[start : start + stride] for start in range(0, len(smiles_rows), stride)]
        with Pool(processes) as pool:
            results = pool.map(_cdk_fingerprints, batches)
    else:
        results = [_cdk_fingerprints(smiles_rows)]
    pubchem_list = [value for result in results for value in result[0]]
    refused = sum(result[1] for result in results)

    share = 100.0 * refused / max(1, len(pubchem_list))
    if share > 1.0:
        logger.warning(f"CDK refused {refused:,} of {len(pubchem_list):,} molecules ({share:.1f}%) in {parquet_path}")
    else:
        logger.info(f"CDK refused {refused:,} of {len(pubchem_list):,} molecules ({share:.2f}%)")
    if refused == len(pubchem_list):
        raise RuntimeError(f"every molecule in {parquet_path} was refused; the PubChem column would be all zeros")

    pubchem_list = pa.array(pubchem_list, pa.binary(111))
    pubchem_field = pa.field("pubchem", pa.binary(111), nullable=False)

    table = table.append_column(pubchem_field, pubchem_list)
    write_table(table, parquet_path)


def augment_parquets_shard(
    parquet_dir: os.PathLike,
    augmentation: Callable,
    shard_index: int,
    shards_count: int,
):
    filenames: list[str] = sorted(f for f in os.listdir(parquet_dir) if is_fingerprint_shard(f))
    files_count = len(filenames)
    try:
        for file_idx in range(shard_index, files_count, shards_count):
            try:
                filename = filenames[file_idx]
                augmentation(os.path.join(parquet_dir, filename))
                logger.info(f"Augmented shard {filename}. Process # {shard_index} / {shards_count}")
            except KeyboardInterrupt as e:
                raise e

    except KeyboardInterrupt as e:
        logger.info(f"Stopping shard {shard_index} / {shards_count}")
        raise e


def augment_parquet_shards(
    parquet_dir: os.PathLike,
    augmentation: Callable,
    processes: int = 1,
):
    if processes > 1:
        process_pool = []
        for i in range(processes):
            p = Process(
                target=augment_parquets_shard,
                args=(parquet_dir, augmentation, i, processes),
            )
            p.start()
            process_pool.append(p)

        for p in process_pool:
            p.join()
    else:
        augment_parquets_shard(parquet_dir, augmentation, 0, 1)


main_epilog = """
Examples:
  # Process all available datasets
  uv run python -m usearchmolecules.prep_encode

  # Process specific dataset
  uv run python -m usearchmolecules.prep_encode --datasets example

  # Skip CDK fingerprints (faster, but no PubChem fingerprints)
  uv run python -m usearchmolecules.prep_encode --skip-cdk

  # Use specific number of processes
  uv run python -m usearchmolecules.prep_encode --processes 16
"""


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Step 2/5: Add molecular fingerprints to Parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=main_epilog,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["example", "pubchem", "gdb13", "real"],
        default=["example", "pubchem", "gdb13", "real"],
        help="Which datasets to process (default: all available)",
    )
    parser.add_argument(
        "--skip-cdk",
        action="store_true",
        help="Skip PubChem fingerprints (CDK is slow/requires Java)",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=max(cpu_count() - 4, 1),
        help=f"Number of parallel processes (default: {max(cpu_count() - 4, 1)})",
    )

    args = parser.parse_args()

    logger.info("Adding molecular fingerprints to Parquet files")
    logger.info(f"Datasets: {', '.join(args.datasets)} | Processes: {args.processes} | Skip CDK: {args.skip_cdk}")

    for dataset in args.datasets:
        parquet_dir = f"data/{dataset}/parquet"
        if not os.path.exists(parquet_dir):
            logger.warning(f"Skipping {dataset}: directory {parquet_dir} not found")
            continue

        logger.info("")
        logger.info(f"Processing dataset: {dataset}")

        try:
            shard_names = sorted(f for f in os.listdir(parquet_dir) if is_fingerprint_shard(f))
            by_row = len(shard_names) < args.processes
            if by_row:
                # Fewer shards than cores. One shard per process would leave most of the machine
                # idle, and a single-shard dataset would encode on one core, so spread the rows.
                logger.info(f"{len(shard_names)} shards under {args.processes} processes, splitting rows instead")

            def run(label: str, augmentation):
                logger.info(label)
                if by_row:
                    for filename in shard_names:
                        augmentation(os.path.join(parquet_dir, filename), args.processes)
                else:
                    augment_parquet_shards(parquet_dir, augmentation, args.processes)

            run("Adding MACCS, ECFP4, FCFP4 fingerprints (RDKit)...", augment_with_rdkit)
            if not args.skip_cdk:
                run("Adding PubChem fingerprints (CDK)...", augment_with_cdk)

            logger.info("Settling column order...")
            augment_parquet_shards(parquet_dir, normalize_column_order, args.processes)

            logger.info(f"✓ Successfully processed {dataset}")
        except Exception as e:
            logger.error(f"✗ Failed to process {dataset}: {e}", exc_info=True)
            raise

    logger.info("")
    logger.info("Completed!")


if __name__ == "__main__":
    main()
