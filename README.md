# USearchMolecules

Drug discovery is, mechanically, a search problem.
A medicinal chemist starts with a molecule that almost works — too toxic, too unstable, too hard to make — and needs to find a molecule that's chemically similar but fixes the problem.
Today this search runs through expensive in-house screening libraries of ~1M molecules, commercial catalogs like Enamine REAL with ~6B virtual compounds, and proprietary internal data.
Each search takes hours-to-weeks because the underlying tools are slow and siloed.

![USearchMolecules 7B dataset thumbnail](https://github.com/ashvardanian/ashvardanian/raw/master/repositories/USearchMolecules.jpg?raw=true)

USearchMolecules is a public, open-source search index over every published drug-like molecule that lets a chemist or an automated pipeline ask, in milliseconds:

- _"Show me 100 molecules shaped like this one but without acrylates."_
- _"Show me everything in Enamine that swaps a methyl-pyridine for an ethyl-imidazole on this scaffold."_
- _"Cluster my 50K screening hits by their 3D shape similarity."_
- _"Find all molecules with the same Bemis-Murcko scaffold as paclitaxel that aren't PAINS-flagged."_

More specifically, it covers __7'132'507'184 molecules__ – sourced from:

- 115'627'267 molecules from the __PubChem__ dataset.
- 977'468'267 molecules from the __GDB13__ dataset.
- 6'039'411'650 molecules from the __Enamine REAL__ dataset.

A fourth subset, __Example__, holds 2'000'000 molecules drawn from those three and is not added to the total.
It is the one to start with: a single download that carries the whole collection's diversity rather than one corner of one subset.

All molecules come with several important categories of information:

- Traditional structural dictionary-based keys, commonly used for substructure search:
  - __MACCS__: Molecular ACCess System keys with __166__ dimensions.
  - __PubChem__: Structure Fingerprints with __881__ dimensions.
- Hashed circular fingerprints often used for similarity search:
  - __ECFP4__: Extended Connectivity Fingerprint of diameter 4 with __2048__ dimensions.
  - __FCFP4__: Functional Class Fingerprint of diameter 4 with __2048__ dimensions.
- Multiple 3D conformations produced computationally:
  - __ETKDG__ initialization from RDKit's experimental torsion preferences.
  - __MMFF94__ relaxation of each seed by the Merck Molecular Force Field.
  - __Three+ conformers__ per molecule, kept without deduplication, because the seeds are already diverse.
  - __Per-conformer energies__ in kcal/mol stored alongside coordinates, lowest first.
  - __USRCAT__ shape vectors — __60__ dimensions per conformer, `float16` — for rotation-invariant 3D similarity search without alignment.
    The 12-dimensional USR descriptor is exactly the first 12 values, so it is not stored separately.

Those fingerprints were then indexed using [Unum's USearch](https://github.com/unum-cloud/USearch) to enable real-time search and clustering of molecular structures for drug discovery and broader chemistry.
USearch takes a user-defined metric, and [NumKong](https://github.com/ashvardanian/NumKong) supplies the geometric kernels one can call — Kabsch and Umeyama alignment over the conformer coordinates themselves.
The conformers come from [SmallField](https://github.com/unum-science/SmallField), which runs ETKDG embedding and MMFF94 relaxation entirely on the GPU.
It is published on three interchangeable mirrors — [AWS Open Data](https://registry.opendata.aws/usearch-molecules/), Nebius and Hugging Face — each anonymously accessible and byte-for-byte identical, covered under [Choosing a Mirror](#choosing-a-mirror).

## Dataset Structure

Each subset's `parquet/` folder co-locates two shard families at the same input ranges: `*.parquet` with the 2D fingerprints and `*.3D.parquet` with the 3D conformers and shape descriptors.
Every mirror verifies content on transfer, so a completed download needs no separate manifest.

```sh
.
├── data
│   ├── pubchem
│   │   ├── index-maccs.usearch # 18.6 GB
│   │   ├── index-maccs-ecfp4.usearch # 46.1 GB
│   │   └── parquet # 30 GB 2D + 117 GB 3D
│   │       ├── 0000000000-0001000000.parquet # 265 MB, 2D fingerprints
│   │       ├── 0000000000-0001000000.3D.parquet # ~1.0 GB, 3D conformers
│   │       ├── ...
│   │       └── 0115000000-0116000000.3D.parquet
│   ├── gdb13
│   │   ├── index-maccs.usearch # 157.0 GB
│   │   ├── index-maccs-ecfp4.usearch # 390.1 GB
│   │   └── parquet # 189 GB 2D + 674 GB 3D
│   │       ├── 0000000000-0001000000.parquet # 198 MB, 2D fingerprints
│   │       ├── 0000000000-0001000000.3D.parquet # ~690 MB, 3D conformers
│   │       ├── ...
│   │       └── 0977000000-0978000000.3D.parquet
│   └── real
│       └── parquet # 1.5 TB 2D + 6.6 TB 3D
│           ├── 0000000000-0001000000.parquet # 262 MB, 2D fingerprints
│           ├── 0000000000-0001000000.3D.parquet # ~1.2 GB, 3D conformers
│           ├── ...
│           └── 6039000000-6040000000.3D.parquet
└── README.md
```

Pre-constructed search and clustering indexes for the Enamine REAL dataset are much harder to distribute and deploy.
Those are not yet available in the bucket but are available per request.
To view the dataset structure, one can use Python:

```sh
  $ pip install pyarrow
  $ python
>>> import pyarrow.parquet as pq
>>> pq.read_table('data/real/parquet/0000000000-0001000000.parquet')  # 2D fingerprints

pyarrow.Table
smiles: string not null
maccs: fixed_size_binary[21] not null
pubchem: fixed_size_binary[111] not null
ecfp4: fixed_size_binary[256] not null
fcfp4: fixed_size_binary[256] not null
```

The `*.3D.parquet` shards carry one row per conformer — three per molecule, lowest energy first — with the geometry and shape descriptors:

```sh
>>> pq.read_table('data/pubchem/parquet/0000000000-0001000000.3D.parquet')  # 3D conformers

pyarrow.Table
input_shard: string          # source 2D shard stem
input_row: uint64            # row within the source shard
smiles: string
conformer_index: uint8       # energy rank, 0 = lowest
status: uint8
n_heavy_atoms: uint16
n_atoms: uint16
n_bonds: uint16
molecular_weight: float
n_conformers: uint8
conformer_coords: list<halffloat>   # tightly packed (n_atoms, 3), centered frame
conformer_energy: float             # MMFF94 kcal/mol
usrcat: list<halffloat>             # 60-dim shape vector; USR is the first 12
```

Join a conformer back to its 2D fingerprints on `(input_shard, input_row)`.
Schema-level metadata records provenance: `producer_git_sha`, `num_conformers=3`, `max_atoms=192`, `dataset`, `coordinate_frame=centered`, `schema_version`, `dedup=none`.
`max_atoms` counts hydrogens, so it is the ceiling on `n_atoms` rather than on `n_heavy_atoms`.

In a tabular form that will look like:

| Column    | Value                                                      |
| :-------- | :--------------------------------------------------------- |
| `smiles`  | `CNCC(C)NC(=O)C1(C(C)(C)OC)CC1`                            |
| `maccs`   | `0x00000200000002002021227C488B9C02100615FFCC`             |
| `pubchem` | `0x00733000000000000000000000001800…` truncated, 111 bytes |
| `ecfp4`   | `0x40000000000000000000800000002400…` truncated, 256 bytes |
| `fcfp4`   | `0xE0001400000000000000000000000000…` truncated, 256 bytes |

The `data/example` subset is the one to start with: 2 million molecules drawn from all three, small enough to download in one go and diverse enough to stand in for the whole collection.
It carries everything the larger subsets do — fingerprints, conformers, SMILES and both indexes — so an application can be built against it and pointed at PubChem or REAL unchanged.

```sh
.
└── data
    └── example # 4.0 GB
        ├── index-maccs.usearch # 329 MB
        ├── index-maccs-ecfp4.usearch # 817 MB
        ├── parquet # 498 MB 2D + 2.2 GB 3D
        │   ├── 0000000000-0001000000.parquet # 249 MB, 2D fingerprints
        │   ├── 0000000000-0001000000.3D.parquet # 1.1 GB, 3D conformers
        │   ├── 0001000000-0002000000.parquet
        │   └── 0001000000-0002000000.3D.parquet
        └── smiles # 96 MB
            ├── 0000000000-0001000000.smi
            └── 0001000000-0002000000.smi
```

### Molecule Sizes

Two counts matter and they are far apart.
The `smiles` column implies a heavy-atom count, but any force field or 3D pipeline works on the molecule after explicit hydrogens are added, which is roughly twice as large.
Measured on a stratified sample — shards spread across each subset's full index range, a random row group per shard, uniform rows within it, largest fragment only:

| Subset    | Heavy p50 / p95 / p99 / max | With Hydrogens p50 / p95 / p99 / max | Over 192 Atoms |
| :-------- | :-------------------------- | :----------------------------------- | -------------: |
| `pubchem` | 24 / 49 / 79 / 202          | 46 / 90 / 168 / 412                  |         0.68 % |
| `gdb13`   | 13 / 13 / 13 / 13           | 27 / 32 / 34 / 39                    |         0.00 % |
| `real`    | 27 / 29 / 30 / 34           | 52 / 61 / 64 / 70                    |         0.00 % |

PubChem is the only subset with a large-molecule tail, and its p99 is where the two counts diverge most — 79 heavy atoms become 168 with hydrogens.
GDB-13 is a spike rather than a distribution, since every molecule is enumerated at exactly 13 heavy atoms.
Enamine REAL is lead-like and narrow, which is why it never approaches a 192-atom ceiling.

__Shards are not ordered by molecule size.__
Across all 8 sampled REAL shards the mean with-hydrogen count spans 0.81 atoms on a mean of 51.1, non-monotone, with a least-squares slope of −0.04 atoms per shard.
PubChem gives a Spearman correlation of −0.01 against shard index over 16 shards.
So the first N shards of a subset are a fair sample of its size distribution, and a benchmark that consumes shards in filename order is not size-biased.

__The conformers of one molecule are far apart enough that deduplication would discard little.__
Median inter-conformer Kabsch RMSD is 1.45 Å for PubChem, 1.08 Å for GDB13 and 1.50 Å for Enamine REAL, over 36'000 pairs drawn the same way.
Pairs falling within 0.5 Å are 4.5 %, 4.7 % and 1.0 % respectively — REAL is the most diverse of the three, and GDB13 the least, since its rigid caged systems have the fewest rotatable bonds to move.

## Installation

Install straight from GitHub with [uv](https://github.com/astral-sh/uv), no clone required.

```sh
uv pip install "git+https://github.com/unum-science/USearchMolecules.git"                          # read and search the dataset
uv pip install "usearchmolecules[dev] @ git+https://github.com/unum-science/USearchMolecules.git"  # build fingerprints and indexes
uv pip install "usearchmolecules[viz] @ git+https://github.com/unum-science/USearchMolecules.git"  # StreamLit visualizer
uv pip install "usearchmolecules[all] @ git+https://github.com/unum-science/USearchMolecules.git"  # everything
```

To hack on the code, clone it and install editable instead.

```sh
git clone https://github.com/unum-science/USearchMolecules.git
cd USearchMolecules
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[all]"
```

The pip extras are CPU-only; for GPU-accelerated conformer generation, set up the environment via pixi as shown below.

### Via Pixi for GPU Acceleration

For GPU acceleration with nvMolKit, we recommend using [pixi](https://pixi.sh) which handles conda dependencies such as RDKit and nvMolKit seamlessly:

```sh
pixi install
pixi run python -m usearchmolecules.prep_conformers --datasets example --use-gpu --conformers 20 --batch-size 20
```

## Usage

### Exploring Dataset via Command Line Interface

Download the example dataset of 2M molecules:

```sh
mkdir -p data/example
aws s3 sync --no-sign-request s3://usearch-molecules/data/example data/example/
```

If you need just one of the subsets:

```sh
aws s3 sync --no-sign-request s3://usearch-molecules/data/example/ data/example/
aws s3 sync --no-sign-request s3://usearch-molecules/data/pubchem/ data/pubchem/
aws s3 sync --no-sign-request s3://usearch-molecules/data/gdb13/ data/gdb13/
aws s3 sync --no-sign-request s3://usearch-molecules/data/real/ data/real/
```

#### Choosing a Mirror

The dataset lives on three mirrors at identical paths — Parquet shards, `.usearch` indexes, and `.smi` files — so pick whichever is closest or cheapest and fetch it with that backend's own client:

```sh
# AWS Open Data, anonymous. Drop the path to browse: aws s3 ls --no-sign-request s3://usearch-molecules
aws s3 sync --no-sign-request s3://usearch-molecules/data/pubchem/ data/pubchem/

# Nebius, S3-compatible
aws s3 sync --endpoint-url https://storage.us-central1.nebius.cloud \
    s3://usearch-molecules/data/pubchem/ data/pubchem/

# Hugging Face
hf download unum-cloud/USearchMolecules --repo-type dataset --include "data/pubchem/*" --local-dir data
```

Every mirror holds byte-for-byte identical files, so the choice is purely about proximity and cost.

You can immediately check if the indexes are readable:

```sh
  $ python
>>> from usearch.index import Index
>>> Index.metadata("data/pubchem/index-maccs.usearch")

{'kind_metric': <MetricKind.Tanimoto: 116>,
 'kind_scalar': <ScalarKind.B1: 1>,
 'count_present': 115627267,
 'count_deleted': 0,
 'dimensions': 192}
```

With those out of the way, you can now query the downloaded files:

```py
from usearchmolecules.dataset import FingerprintedDataset, shape_mixed

data = FingerprintedDataset.open("data/example", shape=shape_mixed)

# No inspiration? Pick a random molecule with `data.random_smiles()`
results = data.search('CC(O)C(CN)=NNCC(C)(C)C', 100)

results_keys = [r[0] for r in results]
results_smiles = [r[1] for r in results]
results_scores = [r[2] for r in results]
```

Every key also carries its stored geometry, so a hit can be rendered or aligned without re-running a force field.
`conformer` returns the molecule's lowest-energy conformer as `(xyz, energy)`, or `None` where the pipeline produced none:

```py
xyz, energy = data.conformer(results_keys[0])   # (n_atoms, 3) float16, MMFF94 kcal/mol
```

Coordinates follow the atom order of the largest fragment after `AddHs`, so they line up with a molecule rebuilt from the matching `smiles`.

## Exploring Dataset via Graphical Interface

The dataset also comes with a graphical sandbox implemented with StreamLit and 3DMol.js to help visualize similarities between molecules.

```sh
streamlit run streamlit_app.py
```

![USearchMolecules StreamLit demo preview](/assets/USearchMoleculesStreamLitPreview.gif)

## Methodology

### Dataset Sources

Original data came from:

- __NCBI PubChem__: [CID-SMILES archive](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz)
- __University of Bern GDB13__: [GDB13 archive](https://zenodo.org/record/5172018/files/gdb13.tgz?download=1)
- __Enamine REAL__: [CXSMILES archives](https://enamine.net/compound-collections/real-compounds/real-database), split by heavy-atom count
- __Example__: drawn from the three above by `prep_sample`, not downloaded

### Pipeline

The data processing pipeline consists of 5 steps, each implemented as a standalone script:

1. `prep_parquet.py`: Convert raw datasets into standardized Parquet shards with SMILES strings.
2. `prep_encode.py`: Add molecular fingerprints (MACCS, ECFP4, FCFP4, PubChem) to Parquet files.
3. `prep_index.py`: Build USearch similarity indexes for fast nearest neighbor search.
4. `prep_conformers.py`: Generate 3D conformers using ETKDG and optionally optimize with MMFF94.
5. `prep_smiles.py`: Export SMILES strings to newline-delimited `.smi` files for [StringZilla][stringzilla].

Every script is designed to work with bigger-than-memory data.
In other words, processing 1 TB of molecules doesn't require 1 TB of RAM.
Everything happens in a "gliding-window" fashion, with computationally intensive parts split between processes and threads.

```sh
uv run python -m usearchmolecules.prep_parquet --datasets example
uv run python -m usearchmolecules.prep_encode --datasets example
uv run python -m usearchmolecules.prep_index --datasets example
uv run python -m usearchmolecules.prep_smiles --datasets example
uv run python -m usearchmolecules.prep_conformers --datasets example
```

Once completed, datasets have been uploaded to S3:

```sh
aws s3 sync data/example/parquet/ s3://usearch-molecules/data/example/parquet/
aws s3 sync data/pubchem/parquet/ s3://usearch-molecules/data/pubchem/parquet/
aws s3 sync data/gdb13/parquet/ s3://usearch-molecules/data/gdb13/parquet/
aws s3 sync data/real/parquet/ s3://usearch-molecules/data/real/parquet/
```

[stringzilla]: https://github.com/ashvardanian/StringZilla

### Understanding Fingerprints

Not all per-molecule vectors in this dataset are interchangeable, and using one where another is required produces silently wrong answers.
They divide into three families with distinct query semantics:

1. __Similarity vectors__: hashed circular fingerprints such as ECFP4 and FCFP4, and the atom-pair and topological-torsion variants.
2. __Substructure flags__: dictionary fingerprints MACCS and PubChem, plus the SMARTS-catalog bits from the alerts, functional-group, pharmacophore, and reaction-template collections.
3. __Geometric shape descriptors__: USR and USRCAT, computed per 3D conformer.

The first family encodes local atomic environments and folds them via hashing into a fixed bitvector.
Individual bits have no chemical meaning — hash collisions are expected and harmless, because these vectors are designed for whole-molecule comparison via Tanimoto or cosine similarity.
They power _"show me molecules similar to X"_ queries, but cannot be used for substructure search: a bit-subset relationship between query and target does not imply structural containment.

The second family assigns one bit per named SMARTS pattern, so each bit has a defined chemical meaning.
They are sound for substructure screening — if every bit set in a query is also set in a candidate, the candidate is guaranteed to contain the corresponding patterns.
They make poor general similarity vectors, because most bits are zero for any single molecule and small structural changes flip many bits at once.

The third family is the only one that is not binary.
USR and USRCAT are short floating-point vectors derived from each conformer's distribution of atom-to-reference-point distances, translation- and rotation-invariant by construction.
Cosine similarity in this space corresponds to overall 3D shape similarity without Kabsch alignment or atom-count matching, which is what enables _"find molecules shaped like this one even if their 2D structure differs"_ — the canonical scaffold-hop query that 2D fingerprints cannot answer.

A useful search pipeline chains all three: a similarity-vector prefilter to find a candidate neighbourhood, a substructure-flag gate to drop alerts and filter by chemical class, and a shape-descriptor ranking by 3D similarity at the end.

### What's Persisted

Each Parquet shard stores the columns below.
The guiding principle is: __cache what's expensive to recompute, skip what's cheap to reconstruct from SMILES__.

Conformer generation with ETKDG + MMFF costs 60-600 ms per molecule depending on size and conformer count.
By contrast, parsing a SMILES string back into a full molecular graph with atom types, bond topology, formal charges, and stereochemistry takes under 0.2 ms.
That 300-3000x cost gap is why we persist 3D coordinates but not the molecular graph.

The 2D fingerprints and the 3D conformers live in __two separate Parquet files per shard__, sharing a stem: `parquet/<range>.parquet` for fingerprints with one row per molecule and `parquet/<range>.3D.parquet` for conformers with __one row per conformer__.
A 3D row is joined back to its fingerprints by the `input_shard` and `input_row` keys.

__Fingerprint shard__ `parquet/<range>.parquet` — one row per molecule:

| Column    | Type          | Typical Size | Description                                              |
| --------- | ------------- | ------------ | -------------------------------------------------------- |
| `smiles`  | `utf8`        | ~50 B        | Canonical graph: atoms, bonds, charges, stereochemistry. |
| `maccs`   | `binary(21)`  | 21 B         | MACCS structural keys, 166 bits.                         |
| `pubchem` | `binary(111)` | 111 B        | PubChem substructure fingerprint, 881 bits. Needs CDK.   |
| `ecfp4`   | `binary(256)` | 256 B        | Extended-connectivity fingerprint, radius 2, 2048 bits.  |
| `fcfp4`   | `binary(256)` | 256 B        | Functional-class fingerprint, radius 2, 2048 bits.       |

__Conformer shard__ `parquet/<range>.3D.parquet` — one row per conformer, with `K = 3` rows per molecule and a molecule that failed all conformers keeping a single null row:

| Column             | Type            | Description                                                                |
| ------------------ | --------------- | -------------------------------------------------------------------------- |
| `input_shard`      | `utf8`          | Stem of the source fingerprint shard (join key).                           |
| `input_row`        | `uint64`        | 0-based row within that shard (join key). Shard-local, not a global id.    |
| `smiles`           | `utf8`          | Carried for convenience.                                                   |
| `conformer_index`  | `uint8`         | 0..K-1, the __energy rank__ where 0 is the lowest-energy conformer.        |
| `status`           | `uint8`         | Per-conformer verdict; see the status-code table below.                    |
| `n_heavy_atoms`    | `uint16`        | Heavy-atom count (null if the molecule failed preprocess).                 |
| `n_atoms`          | `uint16`        | Total atoms incl. H, to reshape coordinates.                               |
| `n_bonds`          | `uint16`        | Bond count.                                                                |
| `molecular_weight` | `float32`       | Exact mass in Daltons.                                                     |
| `n_conformers`     | `uint8`         | K, fixed at 3, no deduplication.                                           |
| `conformer_coords` | `list<float16>` | `3 * n_atoms` values, row-major `(n_atoms, 3)`; null unless `status == 0`. |
| `conformer_energy` | `float32`       | This conformer's MMFF94 energy in kcal/mol; null unless `status == 0`.     |
| `usrcat`           | `list<float16>` | 60-dim USRCAT shape vector; null unless `status == 0`. USR = its first 12. |

__The `status` column__ is the per-conformer verdict, and only `success` carries geometry — `conformer_coords`, `conformer_energy` and `usrcat` are null under every other code.
Codes are append-only, and shipped shards carry only 0, 1, 4, 6, 8, 9 and 14.

| Code | Name                     | Meaning                                         |
| ---: | :----------------------- | :---------------------------------------------- |
|    0 | `success`                | Embedded, polished, finite energy.              |
|    1 | `stereo_failure`         | A declared centre or E/Z bond went unsatisfied. |
|    2 | `mmff_unconverged`       | The minimiser reached no stationary point.      |
|    3 | `invalid_smiles`         | RDKit refused the SMILES.                       |
|    4 | `too_many_atoms`         | Above the 192-atom pipeline ceiling.            |
|    5 | `preprocess_timeout`     | Reserved; nothing emits it.                     |
|    6 | `unsupported_element`    | No MMFF94 atom type — metals, boron, `SF5`.     |
|    7 | `too_many_bonds`         | Wider than the descriptor's bond arrays.        |
|    8 | `too_many_rings`         | More SSSR rings than the ring bitmap holds.     |
|    9 | `bounds_infeasible`      | Smoothing left a lower bound above its upper.   |
|   10 | `nonphysical_energy`     | The energy is not finite.                       |
|   11 | `batch_overflow`         | The batch, not the molecule, failed to fit.     |
|   12 | `slot_unwritten`         | No kernel reached the slot.                     |
|   13 | `polish_inverted_center` | The polish inverted a tetrahedral centre.       |
|   14 | `polish_inverted_bond`   | The polish inverted an E/Z bond.                |

Codes 4, 6 and 8 refuse the molecule before embedding, so it keeps a single row with null sizes; codes 1, 9 and 14 keep the full row block with only the geometry columns null.

Coordinates are centroid-centered — orientation is not canonicalized.
Conformer rows within a molecule are emitted energy-sorted, lowest first.
Each shard also carries schema-level key-value metadata recording provenance: producer git SHA, `num_conformers`, `max_atoms`, dataset, coordinate frame, and schema version.
`max_atoms` is 192 and counts hydrogens; a molecule above it is refused whole and keeps a single row with `status = 4`.

Coordinates are stored as IEEE 754 `float16` and not `bfloat16`, because PyArrow and the Parquet specification natively support `float16`, while `bfloat16` has no Parquet encoding.
The quantization error from `float64` to `float16` is under 0.002 Angstroms for any coordinate within 8 Angstroms of the centroid, and under 0.008 Angstroms beyond it, where the `float16` grid spacing doubles.
Both bounds sit well below thermal noise at room temperature of ~0.1 Angstroms.

__What we intentionally don't store:__

- __Bond topology__ — atom pairs plus bond orders — this is literally what SMILES encodes.
  `C-C(=O)-O` directly specifies which atoms connect and by what bond type.
- __Atom types__ — element per atom index — every letter in the SMILES string IS the atom type.
  After `AddHs`, hydrogen placement is deterministic.
- __Formal charges__ — integer per atom — encoded explicitly in SMILES brackets, e.g. `[NH3+]`, `[O-]`.
- __Stereochemistry__ — chirality, E/Z geometry — encoded with `@`/`@@` and `/`/`\` in SMILES, and also inferable from the 3D coordinates.

All four are losslessly recoverable from the `smiles` column in under 0.2 ms via `Chem.MolFromSmiles` + `AddHs`.

__One caveat for multi-fragment SMILES.__
Where a SMILES names more than one disconnected fragment — a salt, a counter-ion, a mixture — the conformer covers only the fragment the pipeline kept, while the `smiles` column retains the whole string.
About 9 % of PubChem rows are affected; GDB13 and Enamine REAL are not.
Reproduce the pipeline's choice with RDKit's `LargestFragmentChooser` under `preferOrganic`, which keeps the organic parent even when a metal cluster carries more atoms:

```python
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

chooser = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
mol = chooser.choose(Chem.MolFromSmiles(row["smiles"]))
Chem.SanitizeMol(mol)   # ring perception is stale once atoms are removed
mol = Chem.AddHs(mol)   # now mol.GetNumAtoms() == row["n_atoms"]
```

Selecting the largest fragment by raw atom count instead reproduces most of these rows but not all of them.

To read conformers back into NumPy arrays:

```python
import numpy as np

# One row is one conformer; conformer_coords is a flat list of 3 * n_atoms float16 values.
xyz = np.asarray(row["conformer_coords"], dtype=np.float16).reshape(row["n_atoms"], 3)
usrcat = np.asarray(row["usrcat"], dtype=np.float16)   # 60-dim shape descriptor; USR is usrcat[:12]
energy = row["conformer_energy"]                       # MMFF94 energy, kcal/mol
```

For a typical drug-like molecule of ~40 atoms including hydrogens with 3 conformers, the coordinates take ~1.4 KB in `float16` versus ~15 KB for the previous SDF text format - a ~10x reduction; the 60-dim `usrcat` adds ~360 B per conformer.

### Limits and Caveats

__Conformer yield is not 100 %, and the shortfall is a size effect.__
Measured over a stratified sample of the three sources, and over all of `example`:

| Subset    | 3 Conformers |      2 |      1 |      0 |
| :-------- | -----------: | -----: | -----: | -----: |
| `example` |      99.71 % | 0.20 % | 0.09 % |      — |
| `pubchem` |      97.93 % | 0.21 % | 0.09 % | 1.77 % |
| `gdb13`   |     100.00 % |      — |      — |      — |
| `real`    |      99.99 % | 0.00 % | 0.00 % | 0.01 % |

`example` reads better than its sources because it is drawn from molecules whose first conformer succeeded, so it inherits none of their outright failures.
PubChem holds the only meaningful shortfall, and it is concentrated in the tail.
Yield stays near 99.95 % below 75 atoms and falls to 77.6 % in the 175-to-192-atom band.
GDB13 and Enamine REAL are narrow enough that they never approach the ceiling.

__A `success` status bounds the energy's finiteness, not its magnitude.__
Conformers far above their own molecule's minimum are structures the minimiser never relaxed, and they are marked successful.
Filter on the gap to the molecule's lowest conformer rather than on absolute energy, which has no absolute zero in MMFF94:

```python
# conformer_index is the energy rank, so index 0 is the molecule's minimum
usable = [c for c in conformers if c["conformer_energy"] - conformers[0]["conformer_energy"] < 50.0]
```

At a 100 kcal/mol gap this affects roughly 7 % of GDB13 molecules, whose small bridged cages ETKDG struggles to embed, against about 1 % of PubChem and Enamine REAL.
Because the rows are energy-sorted, the affected conformer is almost always the last of the three, and `conformer_index = 0` is the safest single geometry to consume.

## Related Work

PubChem3D is the closest analogue, offering public 3D shape similarity as a standing service rather than a download.
It covers about 85 million molecules, and being a service it answers only the question it was built to answer — you cannot point it at your own corpus, cluster a set you bring, or join shape to anything else.

ZINC-22 and VirtualFlow ship more 3D than that and no search at all.
Retrieving ZINC-22's 4.5 billion structures means moving about a petabyte first, and VirtualFlow's 68.7 billion sit behind a credentialed bucket.
Both are distribution rather than retrieval.

OpenEye's ROCS and FastROCS score Gaussian volume overlap over the whole geometry rather than a descriptor of it, but the work scales with the corpus on every query, the scoring is closed, and the alignment is a local optimum over whatever conformers were generated.
BioSolveIT's SpaceGrow never enumerates, searching billions on one CPU core by growing molecules from synthons, but works only inside a reaction space and cannot cluster an arbitrary set.

USearchMolecules keeps the corpus indexed across modalities, so a substructure screen, a similarity sweep and a shape query all run against the same shards.

## Citation

If USearchMolecules helps your research or product, please cite it:

```bibtex
@software{Vardanian_USearchMolecules,
  author = {Vardanian, Ash},
  title = {{USearchMolecules: A Multi-Modal Atlas of 7 Billion Small Molecules}},
  doi = {10.5281/zenodo.21613663},
  url = {https://github.com/unum-science/USearchMolecules},
  license = {Apache-2.0}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is provided at the repository root.
