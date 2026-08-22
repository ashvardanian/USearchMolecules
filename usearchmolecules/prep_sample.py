"""Cross-Subset Sampling - Build the `example` corpus by drawing from the published subsets.

The `example` subset is not converted from a raw source. It is drawn from PubChem, GDB13 and
Enamine REAL so that a single small download shows the whole collection's diversity rather than
one neighbourhood of one subset.

Half of each subset's budget is chosen for spread in 3D shape space, using the `usrcat` vectors
already stored in the conformer shards, and half for structural rarity, so the corpus is both a
credible similarity-search demo and a regression corpus that exercises the uncommon code paths.

Input:
    - data/{source}/parquet/*.3D.parquet - Conformer shards, for `smiles` and `usrcat`

Output:
    - data/example/parquet/*.parquet - Shards with 'smiles' column, 1M rows each
    - data/example/base/*.parquet - Staging copies the 3D generator reads

Usage:

    uv run python -m usearchmolecules.prep_sample
    uv run python -m usearchmolecules.prep_sample --count 1000000
    uv run python -m usearchmolecules.prep_sample --sources pubchem gdb13
"""

import argparse
import glob
import logging
import os
import random
import re
import shutil
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from usearchmolecules.dataset import SEED, SHARD_SIZE, shard_name, write_table

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Elements MMFF94 can type. Anything else lands in the `off_mmff_element` bucket.
MMFF_ELEMENTS = frozenset("H C N O F Si P S Cl Br I Fe Zn Na K Ca Mg Cu Li".split())

# Atom counts either side of the pipeline's dispatch tiers, where kernel sizing changes.
ATOM_BIN_EDGES = frozenset({16, 17, 32, 33, 48, 49, 64, 65, 80, 81, 96, 97, 112, 113, 127, 128, 129})

BRACKET_ATOM = re.compile(r"\[(\d*)([A-Z][a-z]?)")
# Four or more single-halogen branches on one atom, which is how `SF5` and its relatives are
# written. MMFF94 refuses them for their valence rather than their element, so the element scan
# above cannot see them.
HYPERVALENT = re.compile(r"(?:\((?:F|Cl|Br|I)\)){4,}")
RING_CLOSURE = re.compile(r"[%\d]")

# region Candidate Pool


def iter_candidate_blocks(source: str, target_molecules: int, row_group_stride: int, rng: random.Random):
    """Yield one block per row group, spread evenly across the subset's shards.

    Reading shards in order until the pool fills would cover only the head of the list, which is one
    deposition batch rather than a sample, so the groups to read are chosen first. Blocks rather than
    rows keeps the `usrcat` column out of Python objects.
    """

    root = os.path.join("data", source, "parquet")
    paths = sorted(glob.glob(os.path.join(root, "*.3D.parquet")))
    if not paths:
        logger.warning(f"{source}: no conformer shards under {root}")
        return

    probe = pq.ParquetFile(paths[0])
    per_group = max(1, probe.metadata.num_rows // max(1, probe.metadata.num_row_groups) // 3)
    groups_needed = max(1, -(-target_molecules // per_group))

    plan: list[tuple[str, int]] = []
    if groups_needed <= len(paths):
        step = len(paths) / groups_needed
        for index in range(groups_needed):
            path = paths[min(len(paths) - 1, int(index * step))]
            count = pq.ParquetFile(path).metadata.num_row_groups
            plan.append((path, rng.randrange(count)))
    else:
        per_shard = -(-groups_needed // len(paths))
        for path in paths:
            count = pq.ParquetFile(path).metadata.num_row_groups
            for offset in range(0, min(count, per_shard * row_group_stride), row_group_stride):
                plan.append((path, offset))
    logger.info(f"{source}: reading {len(plan):,} row groups across {len({p for p, _ in plan}):,} of {len(paths):,} shards")

    columns = ["smiles", "conformer_index", "status", "n_heavy_atoms", "n_atoms", "usrcat"]
    for path, group in plan:
        table = pq.ParquetFile(path).read_row_group(group, columns=columns)
        keep = pc.and_(pc.equal(table["conformer_index"], 0), pc.equal(table["status"], 0))
        # A `success` row always carries its shape vector, but the flatten below would silently
        # misalign every later row if one ever did not.
        keep = pc.and_(keep, pc.is_valid(table["usrcat"]))
        table = table.filter(keep)
        if table.num_rows == 0:
            continue
        vectors = np.asarray(table["usrcat"].combine_chunks().flatten(), dtype=np.float32)
        if vectors.size != table.num_rows * 60:
            logger.warning(f"{source}: {os.path.basename(path)} group {group} has ragged usrcat, skipped")
            continue
        yield (
            table["smiles"].to_pylist(),
            np.asarray(table["n_heavy_atoms"], dtype=np.int32),
            np.asarray(table["n_atoms"], dtype=np.int32),
            vectors.reshape(table.num_rows, 60),
            path,
            group,
        )


# endregion

# region Bucketing


def rarity_buckets(smiles: str, heavy_atoms: int | None, total_atoms: int | None) -> list[str]:
    """Which structural buckets a molecule belongs to, read off the SMILES and the stored counts.

    No RDKit parse: every test here is decidable without a molecule graph, and parsing millions of
    candidates costs hours. A molecule can land in several buckets; the caller charges the emptiest.
    """

    buckets: list[str] = []
    if total_atoms is not None:
        if total_atoms in ATOM_BIN_EDGES:
            buckets.append("atoms_bin_edge")
        if total_atoms > 128:
            buckets.append("atoms_gt_128")
    if heavy_atoms is not None and heavy_atoms <= 16:
        buckets.append("atoms_le_16")
    if "." in smiles:
        buckets.append("multi_fragment")

    isotope = False
    charged = False
    off_mmff = False
    for digits, symbol in BRACKET_ATOM.findall(smiles):
        if digits:
            isotope = True
        if symbol not in MMFF_ELEMENTS:
            off_mmff = True
    if isotope:
        buckets.append("isotope")
    if off_mmff:
        buckets.append("off_mmff_element")
    # A charge is only ever written inside brackets, so the scan stays on bracketed spans.
    for span in re.findall(r"\[[^\]]*\]", smiles):
        if "+" in span or "-" in span:
            charged = True
            break
    if charged:
        buckets.append("formal_charge")
    if HYPERVALENT.search(smiles):
        buckets.append("hypervalent")
    if smiles.count("@") >= 8:
        buckets.append("many_stereo")
    if not RING_CLOSURE.search(smiles):
        buckets.append("acyclic")
    # CXSMILES enhanced-stereo groups ride in a trailing `|...|` block and nothing else carries one.
    if smiles.rstrip().endswith("|"):
        buckets.append("enhanced_stereo")
    if not buckets:
        buckets.append("drug_like")
    return buckets


def shape_bucket_keys(vectors: np.ndarray, budget: int, max_components: int, divisions: int,
                      rng: random.Random) -> np.ndarray:
    """Quantize USRCAT vectors onto a coarse grid over their leading principal components.

    Farthest-point selection is quadratic and out of reach at this scale; a grid spreads the draw in
    one pass. Components are added only while occupied cells stay under the budget, since a grid
    finer than the candidate pool leaves every molecule alone in its own cell.
    """

    sample_size = min(len(vectors), 200_000)
    sample_rows = rng.sample(range(len(vectors)), sample_size)
    sample = vectors[sample_rows].astype(np.float32)
    center = sample.mean(axis=0)
    _, _, basis = np.linalg.svd(sample - center, full_matrices=False)
    basis = basis[:max_components]

    projected = (vectors.astype(np.float32) - center) @ basis.T
    target_cells = max(1024, budget // 4)

    keys = np.zeros(len(projected), dtype=np.int64)
    used = 0
    for axis in range(basis.shape[0]):
        # Quantile edges rather than equal widths, so a heavy-tailed component does not put every
        # molecule in one cell.
        edges = np.quantile(projected[sample_rows, axis], np.linspace(0.0, 1.0, divisions + 1)[1:-1])
        candidate = keys * divisions + np.searchsorted(edges, projected[:, axis])
        occupied = len(np.unique(candidate))
        if used and occupied > target_cells:
            break
        keys = candidate
        used = axis + 1
        if occupied > target_cells:
            break
    logger.info(f"  shape grid: {used} of {basis.shape[0]} components, {len(np.unique(keys)):,} occupied cells")
    return keys


# endregion

# region Selection


def draw_round_robin(buckets: dict, budget: int, rng: random.Random) -> list[int]:
    """Take from every bucket in turn until the budget is met, so a rare bucket weighs as much as a common one."""

    order = sorted(buckets)
    for key in order:
        rng.shuffle(buckets[key])
    cursors = dict.fromkeys(order, 0)
    taken: list[int] = []
    exhausted = 0
    while len(taken) < budget and exhausted < len(order):
        exhausted = 0
        for key in order:
            if len(taken) >= budget:
                break
            cursor = cursors[key]
            if cursor >= len(buckets[key]):
                exhausted += 1
                continue
            taken.append(buckets[key][cursor])
            cursors[key] = cursor + 1
    return taken


def select_from_source(source: str, budget: int, arguments, rng: random.Random) -> list[str]:
    """Draw `budget` molecules from one subset, half by shape spread and half by structural rarity."""

    smiles: list[str] = []
    heavy_blocks: list[np.ndarray] = []
    atom_blocks: list[np.ndarray] = []
    vector_blocks: list[np.ndarray] = []
    # Which shard and row group each candidate came from, held as one entry per block and an
    # index per candidate, so the conformer rows can be fetched later without a corpus-wide scan.
    origins: list[tuple[str, int]] = []
    origin_blocks: list[np.ndarray] = []
    pool_ceiling = budget * arguments.pool_factor
    for block_smiles, block_heavy, block_atoms, block_vectors, block_path, block_group in iter_candidate_blocks(
        source, pool_ceiling, arguments.row_group_stride, rng
    ):
        room = pool_ceiling - len(smiles)
        if room <= 0:
            break
        if len(block_smiles) > room:
            block_smiles = block_smiles[:room]
            block_heavy = block_heavy[:room]
            block_atoms = block_atoms[:room]
            block_vectors = block_vectors[:room]
        smiles.extend(block_smiles)
        heavy_blocks.append(block_heavy)
        atom_blocks.append(block_atoms)
        vector_blocks.append(block_vectors)
        origin_blocks.append(np.full(len(block_smiles), len(origins), dtype=np.int32))
        origins.append((block_path, block_group))
    if not smiles:
        return []
    heavy = np.concatenate(heavy_blocks)
    atoms = np.concatenate(atom_blocks)
    vectors = np.concatenate(vector_blocks)
    origin_of = np.concatenate(origin_blocks)
    logger.info(f"{source}: pooled {len(smiles):,} candidates for a budget of {budget:,}")

    shape_budget = budget // 2
    rarity_budget = budget - shape_budget

    rarity_pool: dict[str, list[int]] = defaultdict(list)
    for index, entry in enumerate(smiles):
        options = rarity_buckets(entry, int(heavy[index]), int(atoms[index]))
        rarity_pool[min(options, key=lambda name: len(rarity_pool[name]))].append(index)
    logger.info(f"{source}: rarity buckets " + ", ".join(f"{k}={len(v):,}" for k, v in sorted(rarity_pool.items())))
    rarity_taken = draw_round_robin(rarity_pool, rarity_budget, rng)

    # GDB13 carries no salts, isotopes, radicals or charges, so its rarity buckets cannot fill.
    # The shortfall goes back to the shape half rather than to another subset.
    shape_budget += rarity_budget - len(rarity_taken)

    chosen = set(rarity_taken)
    remaining = np.array([index for index in range(len(smiles)) if index not in chosen], dtype=np.int64)
    shape_pool: dict[int, list[int]] = defaultdict(list)
    if len(remaining):
        keys = shape_bucket_keys(vectors[remaining], shape_budget, arguments.components,
                                 arguments.divisions, rng)
        for position, key in enumerate(keys):
            shape_pool[int(key)].append(int(remaining[position]))
        logger.info(f"{source}: {len(shape_pool):,} occupied shape cells")
    shape_taken = draw_round_robin(shape_pool, shape_budget, rng)

    picked = rarity_taken + shape_taken
    logger.info(f"{source}: took {len(rarity_taken):,} by rarity and {len(shape_taken):,} by shape")
    return [(smiles[index], *origins[origin_of[index]]) for index in picked]


# endregion

# region Output


def write_shards(rows: list[str], directory: str, shard_size: int) -> int:
    """Write `rows` as `%010d-%010d.parquet` shards, plus a staging copy for the 3D generator.

    Stems are the key space, so a partial trailing shard is dropped rather than written short.
    """

    parquet_dir = os.path.join(directory, "parquet")
    base_dir = os.path.join(directory, "base")
    os.makedirs(parquet_dir, exist_ok=True)
    os.makedirs(base_dir, exist_ok=True)

    written = 0
    for start in range(0, len(rows) - shard_size + 1, shard_size):
        chunk = rows[start : start + shard_size]
        path = shard_name(directory, start, start + shard_size, "parquet")
        schema = pa.schema([pa.field("smiles", pa.string(), nullable=False)])
        write_table(pa.table({"smiles": pa.array(chunk, type=pa.string())}, schema=schema), path)
        shutil.copyfile(path, os.path.join(base_dir, os.path.basename(path)))
        written += 1
        logger.info(f"wrote {os.path.basename(path)}")
    return written


# endregion

main_epilog = """
Examples:
  # Build the standard ten-shard example corpus
  uv run python -m usearchmolecules.prep_sample

  # A smaller corpus for a quick check
  uv run python -m usearchmolecules.prep_sample --count 1000000
"""


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Build the example corpus by drawing from the published subsets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=main_epilog,
    )
    parser.add_argument("--dataset", default="example", help="Destination subset (default: example)")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["pubchem", "gdb13", "real"],
        default=["pubchem", "gdb13", "real"],
        help="Subsets to draw from, in equal shares (default: all three)",
    )
    parser.add_argument("--count", type=int, default=10 * SHARD_SIZE, help="Total molecules to draw")
    parser.add_argument("--pool-factor", type=int, default=3, help="Candidates pooled per molecule drawn")
    parser.add_argument("--overdraw", type=float, default=1.02,
                        help="Draw this multiple of the target, so cross-source duplicates cannot leave\n"
                             "the corpus a few molecules short of a whole trailing shard")
    parser.add_argument("--row-group-stride", type=int, default=4, help="Take every Nth row group of a shard")
    parser.add_argument("--components", type=int, default=8,
                        help="Most principal components the shape grid may span; it uses fewer if the grid\n"
                             "would hold more cells than molecules worth spreading across them")
    parser.add_argument("--divisions", type=int, default=10, help="Quantile divisions per component")
    parser.add_argument("--provenance", help="Write a build-time map of molecule to source shard and row group")
    parser.add_argument("--provenance-only", action="store_true",
                        help="Write only the provenance map, leaving existing shards untouched")
    parser.add_argument("--seed", type=int, default=SEED, help="Selection seed")
    arguments = parser.parse_args()

    rng = random.Random(arguments.seed)
    directory = os.path.join("data", arguments.dataset)
    per_source = int(arguments.count * arguments.overdraw) // len(arguments.sources)

    logger.info(f"Drawing {arguments.count:,} molecules into {directory}")
    logger.info(f"Sources: {', '.join(arguments.sources)} at {per_source:,} each")

    rows: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for source in arguments.sources:
        for entry in select_from_source(source, per_source, arguments, rng):
            if entry[0] in seen:
                continue
            seen.add(entry[0])
            rows.append(entry)

    rng.shuffle(rows)
    logger.info(f"Selected {len(rows):,} distinct molecules, trimming to {arguments.count:,}")
    rows = rows[: arguments.count]

    if arguments.provenance:
        # Where every selected molecule's conformers already live. Build-time only: the published
        # shards carry no such column, and this file is scratch the conformer copy reads once.
        table = pa.table({
            "smiles": pa.array([row[0] for row in rows], type=pa.string()),
            "source_shard": pa.array([row[1] for row in rows], type=pa.string()),
            "source_row_group": pa.array([row[2] for row in rows], type=pa.int32()),
        })
        os.makedirs(os.path.dirname(os.path.abspath(arguments.provenance)), exist_ok=True)
        pq.write_table(table, arguments.provenance)
        logger.info(f"Wrote provenance for {table.num_rows:,} molecules to {arguments.provenance}")
        if arguments.provenance_only:
            return

    written = write_shards([row[0] for row in rows], directory, SHARD_SIZE)
    logger.info(f"Wrote {written} shards of {SHARD_SIZE:,} rows to {directory}")
    if written * SHARD_SIZE < len(rows):
        logger.info(f"Dropped {len(rows) - written * SHARD_SIZE:,} rows that would have left a short trailing shard")


if __name__ == "__main__":
    main()
