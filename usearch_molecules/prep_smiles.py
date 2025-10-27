"""SMILES Export - Extract SMILES strings from Parquet to .smi files.

Exports SMILES strings from Parquet files into newline-delimited .smi files
for faster parsing with StringZilla and compatibility with other cheminformatics tools.

Input:
    - data/{dataset}/parquet/*.parquet - Parquet files with SMILES column

Output:
    - data/{dataset}/smiles/*.smi - Newline-delimited SMILES files (one per shard)

Format:
    - Each .smi file contains one SMILES string per line
    - No headers, no delimiters, just raw SMILES strings
    - Ideal for StringZilla memory-mapped file access
    - Compatible with standard cheminformatics tools (RDKit, OpenBabel, CDK)

Usage:

    uv run python -m usearch_molecules.prep_smiles
    uv run python -m usearch_molecules.prep_smiles --datasets example
    uv run python -m usearch_molecules.prep_smiles --datasets example pubchem
"""

import os
import logging
import argparse

from tqdm import tqdm
from stringzilla import File, Str

from usearch_molecules.dataset import FingerprintedDataset

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def export_smiles(data):
    for shard in tqdm(data.shards):
        table = shard.load_table(["smiles"])
        smiles_path = str(shard.table_path)
        smiles_path = smiles_path.replace(".parquet", ".smi")
        smiles_path = smiles_path.replace("/parquet/", "/smiles/")
        if os.path.exists(smiles_path):
            continue

        with open(smiles_path, "w") as f:
            for line in table["smiles"]:
                f.write(str(line) + "\n")

        smiles_file = File(smiles_path)
        reconstructed = smiles_file.splitlines()
        for row, line in enumerate(table["smiles"]):
            assert str(reconstructed[row]) == str(line)
        shard.table_cached = None


main_epilog = """
Examples:
  # Process all available datasets
  uv run python -m usearch_molecules.prep_smiles

  # Process specific dataset
  uv run python -m usearch_molecules.prep_smiles --datasets example
"""


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Export SMILES strings from Parquet to .smi files",
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

    args = parser.parse_args()

    logger.info("Exporting SMILES strings to .smi files")
    logger.info(f"Datasets: {', '.join(args.datasets)}")

    for dataset_name in args.datasets:
        dataset_path = f"data/{dataset_name}"
        if not os.path.exists(dataset_path):
            logger.warning(f"Skipping {dataset_name}: directory {dataset_path} not found")
            continue

        logger.info("")
        logger.info(f"Processing dataset: {dataset_name}")

        try:
            dataset = FingerprintedDataset.open(dataset_path)
            export_smiles(dataset)
            logger.info(f"✓ Successfully exported SMILES for {dataset_name}")
        except Exception as e:
            logger.error(f"✗ Failed to export SMILES for {dataset_name}: {e}", exc_info=True)
            raise

    logger.info("")
    logger.info("Completed!")


if __name__ == "__main__":
    main()
