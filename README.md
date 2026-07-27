#  USearchMolecules

Drug discovery is, mechanically, a search problem.
A medicinal chemist starts with a molecule that almost works — too toxic, too unstable, too hard to make — and needs to find a molecule that's chemically similar but fixes the problem.
Today this search runs through expensive in-house screening libraries (~1M molecules), commercial catalogs (Enamine REAL, ~6B virtual compounds), and proprietary internal data.
Each search takes hours-to-weeks because the underlying tools are slow and siloed.

![USearchMolecules 7B dataset thumbnail](https://github.com/ashvardanian/ashvardanian/raw/master/repositories/USearchMolecules.jpg?raw=true)

USearchMolecules is a public, open-source search index over every published drug-like molecule that lets a chemist (or an automated pipeline) ask, in milliseconds:

- _"Show me 100 molecules shaped like this one but without acrylates."_
- _"Show me everything in Enamine that swaps a methyl-pyridine for an ethyl-imidazole on this scaffold."_
- _"Cluster my 50K screening hits by their 3D shape similarity."_
- _"Find all molecules with the same Bemis-Murcko scaffold as paclitaxel that aren't PAINS-flagged."_

More specifically, it covers __7'131'914'291 molecules__ with up to 50 "heavy" (non-hydrogen) atoms – sourced from:

- 115'034'339 molecules from the __PubChem__ dataset.
- 977'468'301 molecules from the __GDB13__ dataset.
- 6'039'411'651 molecules from the Enamine __REAL__ dataset.

All molecules come with several important categories of information:

- Traditional structural dictionary-based keys, commonly used for substructure search:
  - __MACCS__: Molecular ACCess System keys with __166__ dimensions.
  - __PubChem__: Structure Fingerprints with __881__ dimensions.
- Hashed circular fingerprints often used for similarity search:
  - __ECFP4__: Extended Connectivity Fingerprint of diameter 4 with __2048__ dimensions.
  - __FCFP4__: Functional Class Fingerprint of diameter 4 with __2048__ dimensions.
- Curated SMARTS pattern catalogs, each producing one annotation per molecule:
  - __Pharma alerts__: 2'448 patterns from Eli Lilly, GSK, BMS, Novartis, Pfizer, and AbbVie, combined with academic catalogs from the PAINS, Brenk, NIH, ZINC, Kazius, ChEMBL, and Toxtree projects.
  - __Functional groups__: __1'000+__ distinct groups combining the Toxtree dictionary, Ertl's IFG algorithm, DataWarrior, and RDKit's Fragments vocabulary, used for descriptor generation and chemical-class labeling.
  - __3D pharmacophore features__: __150__ base features for hydrogen-bond donors and acceptors, halogen bonds, aromatic ring stacking, hydrophobic groups, ionizable centres, and metal binders, expandable into Ph4FP triplet and quadruplet fingerprints with millions of dimensions.
  - __Reaction templates__: __500'000+__ SMIRKS templates mined from the USPTO patent corpus, supplemented by the Hartenfeller-58, Schneider BRAINS, and NameRxn named-reaction taxonomies, for retrosynthesis and library enumeration.
  - __Patent Markush patterns__: __1 to 3 million__ generic patent claim structures extracted via the MarkushGrapher pipeline from bulk USPTO, EPO, JPO, KIPO, and CNIPA patent literature, comparable in scope to the commercial CAS MARPAT corpus of ~1.1M Markush records.
- Multiple 3D conformations produced computationally:
  - __ETKDG__ initialization with RDKit's experimentally-tuned distance-geometry generator.
  - __MMFF94__ relaxation of each seed by the Merck Molecular Force Field.
  - __Fixed three conformers__ per molecule, kept without deduplication — the seeds are already diverse (median inter-conformer Kabsch RMSD ~1.44 Å; fewer than 3 % fall within 0.5 Å), so pruning would discard little.
  - __Per-conformer energies__ in kcal/mol stored alongside coordinates, lowest first.
  - __USR__ shape vectors with __12__ dimensions and __USRCAT__ with __60__ dimensions per conformer for rotation-invariant 3D similarity without alignment.

Those fingerprints were then indexed using [Unum's USearch](https://github.com/unum-cloud/usearch) to enable real-time search and clustering of molecular structures for drug discovery and broader chemistry.
The dataset is included in [AWS Open Data platform](https://registry.opendata.aws/usearch-molecules/) and is publicly available from the `s3://usearch-molecules` bucket, accessible even without AWS credentials, entirely anonymously:

```sh
aws s3 ls --no-sign-request s3://usearch-molecules
```

## Dataset Structure

```sh
.
├── data
│   ├── pubchem
│   │   ├── index-maccs.usearch # 18.6 GB
│   │   ├── index-maccs-ecfp4.usearch # 46.1 GB
│   │   └── parquet # 30 GB
│   │       ├── 0000000000-0001000000.parquet # 265 MB
│   │       ├── 0001000000-0002000000.parquet # 265 MB
│   │       ├── ... 
│   │       └── 0115000000-0116000000.parquet # 177 MB
│   ├── gdb13
│   │   ├── index-maccs.usearch # 157.0 GB
│   │   ├── index-maccs-ecfp4.usearch # 390.1 GB
│   │   └── parquet # 189 GB
│   │       ├── 0000000000-0001000000.parquet # 198 MB
│   │       ├── 0001000000-0002000000.parquet # 198 MB
│   │       ├── ... 
│   │       └── 0977000000-0978000000.parquet # 93 MB
│   └── real
│       └── parquet # 477 GB
│           ├── 0000000000-0001000000.parquet # 262 MB
│           ├── 0001000000-0002000000.parquet # 262 MB
│           ├── ... 
│           └── 6039000000-6040000000.parquet # 108 MB
└── README.md
```

Pre-constructed search and clustering indexes for the Enamine REAL dataset are much harder to distribute and deploy.
Those are not yet available in the bucket but are available per request.
To view the dataset structure, one can use Python:

```sh
  $ pip install pyarrow
  $ python
>>> import pyarrow.parquet as pq
>>> pq.read_table('data/real/parquet/0000000000-0001000000.parquet')

pyarrow.Table
smiles: string not null
maccs: fixed_size_binary[21] not null
pubchem: fixed_size_binary[111] not null
ecfp4: fixed_size_binary[256] not null
fcfp4: fixed_size_binary[256] not null
```

In a tabular form that will look like:

<table>
<thead>
<tr><th></th><th>0</th><th>1</th></tr>
</thead>
<tbody>
<tr>
<td><code>smiles</code></td>
<td><code>CNCC(C)NC(=O)C1(C(C)(C)OC)CC1</code></td>
<td><code>CN(C(=O)C1=CC2=C(F)C=C(F)C=C2N1)C1CN(C(=O)CC2=CC=CN=C2O)C1</code></td>
</tr>
<tr>
<td><code>maccs</code></td>
<td><code>0x00000200000002002021227C488B9C02100615FFCC</code></td>
<td><code>0x00900000002000004011172DAC534CE55EF3EB7FFC</code></td>
</tr>
<tr>
<td><code>pubchem</code></td>
<td><code>0x00733000000000000000000000001800000000000000000000000000000000
<br>000000001E00100000000E6CC18006020002C004000800011010000000000000
<br>000000810800000040160080001400000636008000000000000F800000000000
<br>00000000000000000000000000000000</code></td>
<td><code>0x007BB1800000000000000000000000005801600000003C4000000000000000
<br>01F000001F00100800000C28C19E0C3EC4F3C99200A803357754008280203722
<br>2008D921BC6CDC0866F2C295B394710864D611C8D987BE99809E000000000002
<br>00000000000000040000000000000000</code></td>
</tr>
<tr>
<td><code>ecfp4</code></td>
<td><code>0x40000000000000000000800000002400000000000000000000000000000000
<br>0000000010000002000000000000000000008000000000000000000000000000
<br>0000000000000200000000000200000000002000000000010000000000000000
<br>0000000000010000000040000000000000000000020000000800000000000000
<br>0000000000480000000000000000000002802000000000000000000200000000
<br>0000000000000000000000010000000000000002000000000000000000040000
<br>0001000000000000000000000000000000010004000000000000000000000800
<br>0000000000000000008000000000000004000000000000000000100000200000
<br>00</code></td>
<td><code>0x00000000000001000000800000200100000100000000000000000000000000
<br>0200000000000000000400000000080020000000000000008080000000000000
<br>0000020000000000000000000100000000002000000000001400000000100020
<br>0100000000014040000000000000104000000000020100400000000000000040
<br>1000001100400000008800002000000000001000000000000004000000000000
<br>0000000000000000010404000008000000000000000008000000010000000000
<br>0000000000000000042000000000004000020000000000014000004200200000
<br>0000000000000080000020400000000004008000000000000000000040010000
<br>00</code></td>
</tr>
<tr>
<td><code>fcfp4</code></td>
<td><code>0xE0001400000000000000000000000000000000000000000200000000000000
<br>0000000000000000000000000000000000000000000000000000000010004010
<br>0000000000000000000000040000000000000000000000100000000000000000
<br>0000100080000004000000000000000000000000000000000000000000000000
<br>0000000000000000040008000000000000000000000000000010000002000000
<br>0000000000000000000000000000002000000000000000000000000000000000
<br>0000000000000000000000000000000000004000000000000000001000000000
<br>0000000000000800200000040000000000000000000000000000000000000000
<br>80</code></td>
<td><code>0xBE800000000000000001000000000000000080000000080000000000000000
<br>0000000000000000000002000000000000000000000009000000000000000100
<br>0000000001000000000002000000000000000000000000000000000020000000
<br>0000000080080000000000000000000000040000008000000000002000000080
<br>0000000000004000040000000000000000100000000000000000000000000000
<br>0000000040000000000000001400000000000800000000000000000000000000
<br>0000000800000000000000000000000400080000000000001000400000000100
<br>0000000000000000400040000000000024040000000000000000020200400031
<br>80</code></td>
</tr>
</tbody>
</table>

I've also added a tiny sample dataset under the `data/example` directory, with only 2 shards totaling 2 million entries, with pre-constructed indexes to simplify the entry.
Those come in handy if you want to test your application without downloading the whole dataset or visualize a few molecules using the StreamLit app.

```sh
.
└── data
    └── example # 1.8 GB
        ├── index-maccs.usearch # 329 MB
        ├── index-maccs-ecfp4.usearch # 817 MB
        ├── parquet # 30 GB
        │   ├── 0000000000-0001000000.parquet # 265 MB
        │   └── 0001000000-0002000000.parquet # 265 MB
        └── smiles # 30 GB
            ├── 0000000000-0001000000.smi # 58 MB
            └── 0001000000-0002000000.smi # 58 MB
```

## Installation

The project supports multiple installation profiles for different use cases.

### Via UV for CPU Backends

We recommend using [uv](https://github.com/astral-sh/uv) for fast, reliable Python dependency management.

```sh
git clone https://github.com/unum-bio/USearchMolecules.git
cd USearchMolecules

uv venv --python 3.12                   # or your preferred Python version
source .venv/bin/activate               # to activate the virtual environment
uv pip install setuptools wheel         # to pull the latest build tools
uv pip install -e . --force-reinstall   # to build locally from source
uv pip install -e ".[dev]"              # for fingerprinting & generation of indexes
uv pip install -e ".[gpu]"              # for GPU-accelerated processing with nvMolKit
uv pip install -e ".[viz]"              # for visualization with StreamLit
uv pip install -e ".[all]"              # for all features
```

Or install from PyPI:

```sh
uv pip install usearch-molecules
uv pip install "usearch-molecules[dev]" # for fingerprinting & generation of indexes
uv pip install "usearch-molecules[gpu]" # for GPU-accelerated processing with nvMolKit
uv pip install "usearch-molecules[viz]" # for visualization with StreamLit
uv pip install "usearch-molecules[all]" # for all features
```

### Via Pixi for GPU acceleration

For GPU acceleration with nvMolKit, we recommend using [pixi](https://pixi.sh) which handles conda dependencies (RDKit, nvMolKit) seamlessly:

```sh
pixi install
pixi run python -m usearch_molecules.prep_conformers --datasets example --use-gpu --conformers 20 --batch-size 20
```

## Usage

### Exploring Dataset via Command Line Interface

Download the example dataset (2M molecules):

```sh
mkdir -p data/example
aws s3 sync --no-sign-request s3://usearch-molecules/data/example data/example/
```

If you need just one of the subsets:

```sh
aws s3 sync --no-sign-request s3://usearch-molecules/data/pubchem/ data/pubchem/
aws s3 sync --no-sign-request s3://usearch-molecules/data/gdb13/ data/gdb13/
aws s3 sync --no-sign-request s3://usearch-molecules/data/real/ data/real/
```

You can immediately check if the indexes are readable:

```sh
  $ python
>>> from usearch.index import Index
>>> Index.metadata("data/pubchem/index-maccs.usearch") # example of reading metadata

{'matrix_included': True,
 'matrix_uses_64_bit_dimensions': False,
 'version': '2.8.10',
 'kind_metric': <MetricKind.Tanimoto: 116>,
 'kind_scalar': <ScalarKind.B1: 1>,
 'kind_key': <ScalarKind.U64: 8>,
 'kind_compressed_slot': <ScalarKind.U32: 9>,
 'count_present': 115627267,
 'count_deleted': 0,
 'dimensions': 192}

>>> Index.restore("data/pubchem/index-maccs-ecfp4.usearch") # example of parsing it

usearch.Index
- config
-- data type: ScalarKind.B1
-- dimensions: 2240
-- metric: MetricKind.Tanimoto
-- connectivity: 16
-- expansion on addition:128 candidates
-- expansion on search: 64 candidates
- binary
-- uses OpenMP: 1
-- uses SimSIMD: 1
-- uses hardware acceleration: avx512+popcnt
- state
-- size: 115,627,267 vectors
-- memory usage: 69,631,939,864 bytes
-- max level: 4
--- 0. 115,627,267 nodes
--- 1. 7,148,410 nodes
--- 2. 461,450 nodes
--- 3. 37,714 nodes
--- 4. 5,152 nodes
```

With those out of the way, you can now query the downloaded files:

```py
from usearch_molecules.dataset import FingerprintedDataset, shape_mixed

data = FingerprintedDataset.open("data/example", shape=shape_mixed)

# No inspiration? Pick a random molecule with `data.random_smiles()`
results = data.search('CC(O)C(CN)=NNCC(C)(C)C', 100)

results_keys = [r[0] for r in results]
results_smiles = [r[1] for r in results]
results_scores = [r[2] for r in results]
```

## Exploring Dataset via Graphical Interface

The dataset also comes with a graphical sandbox implemented with StreamLit and 3DMol.js to help visualize similarities between molecules.

```sh
streamlit run streamlit_app.py
```

![USearchMolecules StreamLit demo preview](/assets/USearchMoleculesStreamLitPreview.gif)

## Methodology

### Dataset Sources

Original data came from:

- __PubChem__: [CID-SMILES](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz).gz
- __GDB13__: [gdb13](https://zenodo.org/record/5172018/files/gdb13.tgz?download=1).tgz
- Enamine __REAL__, split by Heavy Atom Counts:
    - HAC 6-21: [CXSMILES](https://ftp.enamine.net/download/REAL/Enamine_REAL_HAC_6_21_420M_CXSMILES.cxsmiles.bz2).cxsmiles.bz2
    - HAC 22-23: [CXSMILES](https://ftp.enamine.net/download/REAL/Enamine_REAL_HAC_22_23_471M_CXSMILES.cxsmiles.bz2).cxsmiles.bz2
    - HAC 24: [CXSMILES](https://ftp.enamine.net/download/REAL/Enamine_REAL_HAC_24_394M_CXSMILES.cxsmiles.bz2).cxsmiles.bz2
    - HAC 25: [CXSMILES](https://ftp.enamine.net/download/REAL/Enamine_REAL_HAC_25_557M_CXSMILES.cxsmiles.bz2).cxsmiles.bz2
    - HAC 26:
      - [CXSMILES Part 1](https://ftp.enamine.net/download/REAL/Enamine_REAL_HAC_26_833M_Part_1_CXSMILES.cxsmiles.bz2).cxsmiles.bz2
      - [CXSMILES Part 2](https://ftp.enamine.net/download/REAL/Enamine_REAL_HAC_26_833M_Part_2_CXSMILES.cxsmiles.bz2).cxsmiles.bz2
    - HAC 27:
      - [CXSMILES Part 1](https://ftp.enamine.net/download/REAL/Enamine_REAL_HAC_27_1.1B_Part_1_CXSMILES.cxsmiles.bz2).cxsmiles.bz2
      - [CXSMILES Part 2](https://ftp.enamine.net/download/REAL/Enamine_REAL_HAC_27_1.1B_Part_2_CXSMILES.cxsmiles.bz2).cxsmiles.bz2
    - HAC 28:
      - [CXSMILES Part 1](https://ftp.enamine.net/download/REAL/Enamine_REAL_HAC_28_1.2B_Part_1_CXSMILES.cxsmiles.bz2).cxsmiles.bz2
      - [CXSMILES Part 2](https://ftp.enamine.net/download/REAL/Enamine_REAL_HAC_28_1.2B_Part_2_CXSMILES.cxsmiles.bz2).cxsmiles.bz2
    - HAC 29-38:
      - [CXSMILES Part 1](https://ftp.enamine.net/download/REAL/Enamine_REAL_HAC_29_38_988M_Part_1_CXSMILES.cxsmiles.bz2).cxsmiles.bz2
      - [CXSMILES Part 2](https://ftp.enamine.net/download/REAL/Enamine_REAL_HAC_29_38_988M_Part_2_CXSMILES.cxsmiles.bz2).cxsmiles.bz2

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
uv run python -m usearch_molecules.prep_parquet --datasets example
uv run python -m usearch_molecules.prep_encode --datasets example
uv run python -m usearch_molecules.prep_index --datasets example
uv run python -m usearch_molecules.prep_smiles --datasets example
uv run python -m usearch_molecules.prep_conformers --datasets example
```

Once completed, datasets have been uploaded to S3:

```sh
aws s3 sync data/pubchem/parquet/ s3://usearch-molecules/data/pubchem/parquet/
aws s3 sync data/gdb13/parquet/ s3://usearch-molecules/data/gdb13/parquet/
aws s3 sync data/real/parquet/ s3://usearch-molecules/data/real/parquet/
```

[stringzilla]: https://github.com/ashvardanian/stringzilla

### Understanding Fingerprints

Not all per-molecule vectors in this dataset are interchangeable, and using one where another is required produces silently wrong answers.
They divide into three families with distinct query semantics:

1. __Similarity vectors__: hashed circular fingerprints (ECFP4, FCFP4) and the atom-pair and topological-torsion variants.
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

Conformer generation (ETKDG + MMFF) costs 60-600 ms per molecule depending on size and conformer count.
By contrast, parsing a SMILES string back into a full molecular graph with atom types, bond topology, formal charges, and stereochemistry takes under 0.2 ms.
That 300-3000x cost gap is why we persist 3D coordinates but not the molecular graph.

| Column               | Type          | Typical Size | Description                                              |
| -------------------- | ------------- | ------------ | -------------------------------------------------------- |
| `smiles`             | `utf8`        | ~50 B        | Canonical graph: atoms, bonds, charges, stereochemistry. |
| `maccs`              | `binary(21)`  | 21 B         | MACCS structural keys, 166 bits.                         |
| `ecfp4`              | `binary(256)` | 256 B        | Extended-connectivity fingerprint, radius 2, 2048 bits.  |
| `fcfp4`              | `binary(256)` | 256 B        | Functional-class fingerprint, radius 2, 2048 bits.       |
| `pubchem`            | `binary(111)` | 111 B        | PubChem substructure fingerprint, 881 bits. Needs CDK.   |
| `n_heavy_atoms`      | `uint16`      | 2 B          | Heavy atom count, for filtering without parsing.         |
| `n_atoms`            | `uint16`      | 2 B          | Total atoms with hydrogens, to reshape coordinates.      |
| `n_bonds`            | `uint16`      | 2 B          | Bond count, for graph models without reparsing.          |
| `molecular_weight`   | `float32`     | 4 B          | Exact mass in Daltons, for filtering.                    |
| `n_conformers`       | `uint8`       | 1 B          | Conformers generated per molecule (fixed `K`, no dedup). |
| `conformer_coords`   | `binary`      | K * N * 6 B  | `float16` coordinates, shape `(K, N, 3)`.                |
| `conformer_energies` | `binary`      | K * 4 B      | MMFF94 energies in kcal/mol, lowest first.               |

Coordinates are stored as IEEE 754 `float16` (not `bfloat16`) because PyArrow and the Parquet specification natively support `float16`, while `bfloat16` has no Parquet encoding.
The quantization error from `float64` to `float16` is under 0.002 Angstroms - well below thermal noise at room temperature (~0.1 Angstroms).

__What we intentionally don't store:__

- __Bond topology__ (atom pairs + bond orders) - this is literally what SMILES encodes. `C-C(=O)-O` directly specifies which atoms connect and by what bond type.
- __Atom types__ (element per atom index) - every letter in the SMILES string IS the atom type. After `AddHs`, hydrogen placement is deterministic.
- __Formal charges__ (integer per atom) - encoded explicitly in SMILES brackets, e.g. `[NH3+]`, `[O-]`.
- __Stereochemistry__ (chirality, E/Z geometry) - encoded with `@`/`@@` and `/`/`\` in SMILES, and also inferable from the 3D coordinates.

All four are losslessly recoverable from the `smiles` column in under 0.2 ms via `Chem.MolFromSmiles` + `AddHs`.

To read conformers back into NumPy arrays:

```python
import numpy as np

coords = np.frombuffer(row["conformer_coords"], dtype=np.float16)
coords = coords.reshape(row["n_conformers"], row["n_atoms"], 3)
energies = np.frombuffer(row["conformer_energies"], dtype=np.float32)
```

For a typical drug-like molecule (~40 atoms with H, 5 conformers), the coordinate column is ~1.2 KB per molecule in `float16` versus ~15 KB for the previous SDF text format - a 12x reduction.

## Citation

If USearchMolecules helps your research or product, please cite it:

```bibtex
@software{Vardanian_USearchMolecules,
  author = {Vardanian, Ash},
  title = {{USearchMolecules: A Multi-Modal Atlas of 7 Billion Small Molecules}},
  doi = {10.5281/zenodo.21613664},
  url = {https://github.com/unum-bio/USearchMolecules},
  license = {Apache-2.0}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is provided at the repository root.
