"""Index Generation - Build USearch similarity indexes for molecular fingerprints.

Builds HNSW (Hierarchical Navigable Small World) indexes using USearch for fast
approximate nearest neighbor search on molecular fingerprints.

Input:
    - data/{dataset}/parquet/*.parquet - Parquet files with fingerprint columns:
      "maccs", "ecfp4", "fcfp4", "pubchem"

Output:
    - data/{dataset}/index-maccs.usearch - MACCS fingerprint index (166 bits)
    - data/{dataset}/index-maccs-ecfp4.usearch - Hybrid MACCS+ECFP4 index (2214 bits)

Index Types:
    - MACCS: 166-bit structural keys for basic similarity search
    - Mixed: MACCS (166 bits) + ECFP4 (2048 bits) for comprehensive similarity

HNSW Parameters:
    - Construction: Multi-threaded graph building
    - Distance: Tanimoto similarity (custom Numba-compiled metric)
    - Accuracy: Self-recall evaluation available (optional)

Usage:

    uv run python -m usearchmolecules.prep_index
    uv run python -m usearchmolecules.prep_index --datasets example
    uv run python -m usearchmolecules.prep_index --datasets example pubchem
"""

import argparse
import logging
import os

import numpy as np
import pyarrow as pa
from usearch.index import CompiledMetric, Index, MetricKind, MetricSignature, ScalarKind

from usearchmolecules.dataset import FingerprintedDataset, FingerprintedEntry
from usearchmolecules.metrics_numba import tanimoto_maccs, tanimoto_mixed
from usearchmolecules.to_fingerprint import FingerprintShape, shape_maccs, shape_mixed

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _binary_index(shape: FingerprintShape, metric_pointer: int) -> Index:
    """A binary USearch index over `shape.nbits` bits using the given compiled Tanimoto metric."""
    return Index(
        ndim=shape.nbits,
        dtype=ScalarKind.B1,
        metric=CompiledMetric(
            pointer=metric_pointer,
            kind=MetricKind.Tanimoto,
            signature=MetricSignature.ArrayArray,
        ),
    )


def _shard_vectors(
    table: pa.Table,
    first_key: int,
    shape: FingerprintShape,
    use_ecfp4: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (keys, packed fingerprint vectors) for one shard's table under `shape`.

    Rows keep their on-disk order — already shuffled at ingest — so row `i` gets key `first_key + i`.
    """
    n = len(table)
    keys = np.arange(first_key, first_key + n)
    maccs = [table["maccs"][i].as_buffer() for i in range(n)]
    ecfp4 = [table["ecfp4"][i].as_buffer() for i in range(n)] if use_ecfp4 else [None] * n
    vectors = np.vstack(
        [FingerprintedEntry.from_parts(None, maccs[i], ecfp4[i], None, shape).fingerprint for i in range(n)]
    )
    return keys, vectors


def _mono_index(
    dataset: FingerprintedDataset,
    shape: FingerprintShape,
    metric_pointer: int,
    columns: list[str],
    index_filename: str,
    save_every: int,
):
    """Build one monolithic index spanning every shard, checkpointing every `save_every` shards.

    Resumable: a shard whose `first_key` is already present is skipped, so re-running continues
    from where an interrupted run left off. Ctrl-C exits cleanly with the partial index saved.
    """
    os.makedirs(dataset.directory, exist_ok=True)
    index_path = os.path.join(dataset.directory, index_filename)
    index = _binary_index(shape, metric_pointer)
    use_ecfp4 = "ecfp4" in columns

    try:
        for shard_idx, shard in enumerate(dataset.shards):
            if shard.first_key in index:
                logger.info(f"Skipping {shard_idx + 1} / {len(dataset.shards)}")
                continue

            logger.info(f"Starting {shard_idx + 1} / {len(dataset.shards)}")
            keys, vectors = _shard_vectors(shard.load_table(columns), shard.first_key, shape, use_ecfp4)
            index.add(keys, vectors, log=f"Building {index_path}")
            if shard_idx % save_every == 0:
                index.save(index_path)

            shard.table_cached = None  # release before the next shard

        index.save(index_path)
        index.reset()
    except KeyboardInterrupt:
        pass


def mono_index_maccs(dataset: FingerprintedDataset):
    """Monolithic MACCS index across every shard."""
    _mono_index(dataset, shape_maccs, tanimoto_maccs.address, ["maccs"], "index-maccs.usearch", save_every=100)


def mono_index_mixed(dataset: FingerprintedDataset):
    """Monolithic mixed MACCS+ECFP4 index across every shard."""
    _mono_index(
        dataset,
        shape_mixed,
        tanimoto_mixed.address,
        ["maccs", "ecfp4"],
        "index-maccs-ecfp4.usearch",
        save_every=50,
    )


main_epilog = """
Examples:
  # Process all available datasets
  uv run python -m usearchmolecules.prep_index

  # Process specific dataset
  uv run python -m usearchmolecules.prep_index --datasets example
"""


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Build USearch similarity indexes for molecular fingerprints",
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

    logger.info("Building USearch similarity indexes")
    logger.info(f"Datasets: {', '.join(args.datasets)}")

    for dataset_name in args.datasets:
        dataset_path = f"data/{dataset_name}"
        if not os.path.exists(dataset_path):
            logger.warning(f"Skipping {dataset_name}: directory {dataset_path} not found")
            continue

        logger.info("")
        logger.info(f"Processing dataset: {dataset_name}")

        try:
            loaded_dataset = FingerprintedDataset.open(dataset_path)
            mono_index_maccs(loaded_dataset)
            mono_index_mixed(loaded_dataset)
            logger.info(f"✓ Successfully indexed {dataset_name}")
        except Exception as e:
            logger.error(f"✗ Failed to index {dataset_name}: {e}", exc_info=True)
            raise

    logger.info("")
    logger.info("Completed!")


if __name__ == "__main__":
    main()
