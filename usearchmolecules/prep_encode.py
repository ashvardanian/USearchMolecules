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
from multiprocessing import Process, cpu_count

import pyarrow as pa
import pyarrow.parquet as pq

from usearchmolecules.dataset import write_table
from usearchmolecules.to_fingerprint import (
    smiles_to_maccs_ecfp4_fcfp4,
    smiles_to_pubchem,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def augment_with_rdkit(parquet_path: os.PathLike):
    meta = pq.read_metadata(parquet_path)
    column_names: list[str] = meta.schema.names
    if "maccs" in column_names and "ecfp4" in column_names and "fcfp4" in column_names:
        return

    logger.info(f"Starting file {parquet_path}")
    table: pa.Table = pq.read_table(parquet_path)
    maccs_list = []
    ecfp4_list = []
    fcfp4_list = []
    for smiles in table["smiles"]:
        try:
            fingers = smiles_to_maccs_ecfp4_fcfp4(str(smiles))
            maccs_list.append(fingers[0].tobytes())
            ecfp4_list.append(fingers[1].tobytes())
            fcfp4_list.append(fingers[2].tobytes())
        except Exception:
            maccs_list.append(bytes(bytearray(21)))
            ecfp4_list.append(bytes(bytearray(256)))
            fcfp4_list.append(bytes(bytearray(256)))

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


def augment_with_cdk(parquet_path: os.PathLike):
    meta = pq.read_metadata(parquet_path)
    column_names: list[str] = meta.schema.names
    if "pubchem" in column_names:
        return

    logger.info(f"Starting file {parquet_path}")
    table: pa.Table = pq.read_table(parquet_path)
    pubchem_list = []
    for smiles in table["smiles"]:
        try:
            fingers = smiles_to_pubchem(str(smiles))
            pubchem_list.append(fingers[0].tobytes())
        except Exception:
            pubchem_list.append(bytes(bytearray(111)))

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
    filenames: list[str] = sorted(os.listdir(parquet_dir))
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
            if not args.skip_cdk:
                logger.info("Adding PubChem fingerprints (CDK)...")
                augment_parquet_shards(parquet_dir, augment_with_cdk, args.processes)

            logger.info("Adding MACCS, ECFP4, FCFP4 fingerprints (RDKit)...")
            augment_parquet_shards(parquet_dir, augment_with_rdkit, args.processes)

            logger.info(f"✓ Successfully processed {dataset}")
        except Exception as e:
            logger.error(f"✗ Failed to process {dataset}: {e}", exc_info=True)
            raise

    logger.info("")
    logger.info("Completed!")


if __name__ == "__main__":
    main()
