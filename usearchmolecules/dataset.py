"""Dataset management for molecular fingerprints stored in sharded Parquet files.

Provides efficient loading and searching of molecular datasets using USearch for
similarity search. Datasets are organized into shards for scalable storage and
memory-mapped access, supporting hybrid fingerprints (MACCS, ECFP4, FCFP4).

Example usage:
    dataset = FingerprintedDataset.open("data/pubchem", shape=shape_maccs)
    results = dataset.search("CC(=O)OC1=CC=CC=C1C(=O)O", count=10)  # Aspirin
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from itertools import islice

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import stringzilla as sz
from tqdm import tqdm
from usearch.index import Index, Key, Matches

from usearchmolecules.to_fingerprint import (
    FingerprintShape,
    shape_maccs,
    smiles_to_maccs_ecfp4_fcfp4,
)

SEED = 42  # For reproducibility
SHARD_SIZE = 1_000_000  # This would result in files between 150 and 300 MB
BATCH_SIZE = 100_000  # A good threshold to split insertions


@dataclass
class FingerprintedEntry:
    """SMILES string augmented with a potentially hybrid fingerprint of known `FingerprintShape` shape."""

    smiles: str
    fingerprint: np.ndarray
    key: int | None = None

    @staticmethod
    def from_table_row(table: pa.Table, row: int, shape: FingerprintShape) -> FingerprintedEntry:
        fingerprint = np.zeros(shape.nbytes, dtype=np.uint8)
        progress = 0
        if shape.include_maccs:
            fingerprint[progress : progress + 21] = table["maccs"][row].as_buffer()
            progress += 21
        if shape.nbytes_padding:
            progress += shape.nbytes_padding
        if shape.include_ecfp4:
            fingerprint[progress : progress + 256] = table["ecfp4"][row].as_buffer()
            progress += 256
        if shape.include_fcfp4:
            fingerprint[progress : progress + 256] = table["fcfp4"][row].as_buffer()
            progress += 256
        return FingerprintedEntry(smiles=table["smiles"][row], fingerprint=fingerprint)

    @staticmethod
    def from_parts(
        smiles: str | None,
        maccs: np.ndarray | None,
        ecfp4: np.ndarray | None,
        fcfp4: np.ndarray | None,
        shape: FingerprintShape,
    ) -> FingerprintedEntry:
        # Each modality is read only when the shape includes it, so an omitted
        # modality may be passed as None; likewise `smiles` is optional for index-only rows.
        fingerprint = np.zeros(shape.nbytes, dtype=np.uint8)
        progress = 0
        if shape.include_maccs:
            assert maccs is not None
            fingerprint[progress : progress + 21] = maccs
            progress += 21
        if shape.nbytes_padding:
            progress += shape.nbytes_padding
        if shape.include_ecfp4:
            assert ecfp4 is not None
            fingerprint[progress : progress + 256] = ecfp4
            progress += 256
        if shape.include_fcfp4:
            assert fcfp4 is not None
            fingerprint[progress : progress + 256] = fcfp4
            progress += 256
        return FingerprintedEntry(smiles=smiles, fingerprint=fingerprint)


def is_fingerprint_shard(filename: str) -> bool:
    """Whether a `parquet/` entry is a 2D fingerprint shard rather than a `*.3D.parquet` conformer shard.

    Both families share the directory and the stem, so a bare `*.parquet` glob picks up conformer shards
    too — and their stems parse as a duplicate `first_key` pointing at a `.3D.smi` that does not exist.
    """
    return filename.endswith(".parquet") and ".3D." not in filename


def shard_name(directory: str, from_index: int, to_index: int, kind: str):
    """Generate standardized shard filename with zero-padded indices."""
    return os.path.join(directory, kind, f"{from_index:0>10}-{to_index:0>10}.{kind}")


def write_table(table: pa.Table, path_out: os.PathLike):
    """Write PyArrow table to Parquet file with optimized settings for molecular data."""
    return pq.write_table(
        table,
        path_out,
        # Without compression the file size may be too large.
        # compression="NONE",
        write_statistics=False,
        store_schema=True,
        use_dictionary=False,
    )


@dataclass
class FingerprintedShard:
    """Potentially cached table and smiles path containing up to `SHARD_SIZE` entries."""

    first_key: int
    name: str

    table_path: os.PathLike
    smiles_path: os.PathLike
    table_cached: pa.Table | None = None
    smiles_caches: sz.Strs | None = None

    @property
    def is_complete(self) -> bool:
        return os.path.exists(self.table_path) and os.path.exists(self.smiles_path)

    @property
    def table(self) -> pa.Table:
        return self.load_table()

    @property
    def smiles(self) -> sz.Strs:
        return self.load_smiles()

    def load_table(self, columns=None, view=False) -> pa.Table:
        """Load Parquet table into memory with optional memory-mapping and column filtering."""
        if not self.table_cached:
            self.table_cached = pq.read_table(
                self.table_path,
                memory_map=view,
                columns=columns,
            )
        return self.table_cached

    def load_smiles(self) -> sz.Strs:
        """Load SMILES strings using StringZilla for fast line-by-line access."""
        if not self.smiles_caches:
            self.smiles_caches = sz.Str(sz.File(self.smiles_path)).splitlines()
        return self.smiles_caches


@dataclass
class FingerprintedDataset:
    directory: os.PathLike
    shards: list[FingerprintedShard]
    shape: FingerprintShape | None = None
    index: Index | None = None

    @staticmethod
    def open(
        directory: os.PathLike,
        shape: FingerprintShape | None = None,
        max_shards: int | None = None,
    ) -> FingerprintedDataset:
        """Open a sharded molecular dataset and optionally load its USearch index.

        Args:
            directory: Root directory containing parquet/ and smiles/ subdirectories
            shape: Fingerprint configuration (determines which index file to load)
            max_shards: Optional limit on number of shards to load (useful for testing)

        Returns:
            FingerprintedDataset with loaded shards and index (if available)
        """

        if directory is None:
            return FingerprintedDataset(directory=None, shards=[], shape=shape)

        shards = []
        filenames = sorted(f for f in os.listdir(os.path.join(directory, "parquet")) if is_fingerprint_shard(f))
        if max_shards:
            filenames = filenames[:max_shards]

        for filename in tqdm(filenames, unit="shard"):
            stem = filename.removesuffix(".parquet")
            shards.append(
                FingerprintedShard(
                    first_key=int(stem.partition("-")[0]),
                    name=stem,
                    table_path=os.path.join(directory, "parquet", stem + ".parquet"),
                    smiles_path=os.path.join(directory, "smiles", stem + ".smi"),
                )
            )

        print(f"Fetched {len(shards)} shards")

        index = None
        if shape:
            index_path = os.path.join(directory, shape.index_name)
            if os.path.exists(index_path):
                index = Index.restore(index_path)

        return FingerprintedDataset(directory=directory, shards=shards, shape=shape, index=index)

    def shard_containing(self, key: int) -> FingerprintedShard:
        """Find the shard containing a given key/index, or raise if none does."""
        for shard in self.shards:
            if shard.first_key <= key <= shard.first_key + SHARD_SIZE:
                return shard
        raise KeyError(f"No shard contains key {key}")

    def head(
        self,
        max_rows: int,
        shape: FingerprintShape | None = None,
        shuffle: bool = False,
    ) -> tuple[list[str], list[int], np.ndarray]:
        """Load the first N rows from the dataset for preview and testing.

        Args:
            max_rows: Maximum number of molecules to load
            shape: Fingerprint configuration (uses dataset default if None)
            shuffle: Whether to randomly shuffle the results

        Returns:
            Tuple of (SMILES strings, keys, fingerprint matrix)
        """

        if self.directory is None:
            return None

        shape = shape or self.shape
        entries = list(islice(self._iter_entries(shape), max_rows))

        smiles = np.array([entry.smiles for entry in entries], dtype=object)
        keys = np.arange(len(entries), dtype=Key)
        fingers = np.vstack([entry.fingerprint for entry in entries])
        if shuffle:
            permutation = np.random.permutation(len(entries))
            smiles, keys, fingers = smiles[permutation], keys[permutation], fingers[permutation]
        return smiles, keys, fingers

    def _iter_entries(self, shape: FingerprintShape):
        """Yield every entry across all shards in on-disk order."""
        for shard in self.shards:
            table = shard.load_table()
            for row in range(len(table)):
                yield FingerprintedEntry.from_table_row(table, row, shape)

    def search(
        self,
        smiles: str,
        count: int = 10,
        log: bool = False,
    ) -> list[tuple[int, str, float]]:
        """Search for similar molecules using approximate nearest neighbor search.

        Args:
            smiles: Query molecule in SMILES notation
            count: Number of similar molecules to return
            log: Whether to log progress (can be a callback function)

        Returns:
            List of tuples (key, smiles, distance) sorted by similarity
        """

        fingers: tuple = smiles_to_maccs_ecfp4_fcfp4(smiles)
        entry = FingerprintedEntry.from_parts(
            smiles,
            fingers[0],
            fingers[1],
            fingers[2],
            self.shape,
        )
        results: Matches = self.index.search(entry.fingerprint, count, log=log)

        filtered_results = []
        for match in results:
            shard = self.shard_containing(match.key)
            row = int(match.key - shard.first_key)
            result = str(shard.smiles[row])
            filtered_results.append((match.key, result, match.distance))

        return filtered_results

    def __len__(self) -> int:
        return len(self.index)

    def random_smiles(self) -> str:
        """Get a random SMILES string from the dataset."""
        shard_idx = random.randint(0, len(self.shards) - 1)
        shard = self.shards[shard_idx]
        row = random.randint(0, len(shard.smiles) - 1)
        return str(shard.smiles[row])


if __name__ == "__main__":
    dataset = FingerprintedDataset.open("data/pubchem", shape=shape_maccs)
    dataset.search("C")
