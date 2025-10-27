#  USearch Molecules

![USearch Molecules 7B dataset thumbnail](/assets/USearchMolecules.jpg)

USearch Molecules is a large Chem-Informatics dataset of small molecules.
It includes __7'131'914'291 molecules__ with up to 50 "heavy" (non-hydrogen) atoms gathered from:

- 115'034'339 molecules from the __PubChem__ dataset.
- 977'468'301 molecules from the __GDB13__ dataset.
- 6'039'411'651 molecules from the Enamine __REAL__ dataset.

All molecules have been encoded using `rdkit` and `cdk` to produce binary fingerprints (structural embeddings) of four kinds:

- __MACCS__: Molecular ACCess System keys with __166__ dimensions.
- __PubChem__: Structure Fingerprints with __881__ dimensions.
- __ECFP4__: Extended Connectivity Fingerprint of diameter 4 with __2048__ dimensions.
- __FCFP4__: Functional Class Fingerprint of diameter 4 with __2048__ dimensions.

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

|      | `smiles`                                                   |                                      `maccs` |                                                                                                                                                                                                                        `pubchem` |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            `ecfp4` |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            `fcfp4` |
| :--- | :--------------------------------------------------------- | -------------------------------------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| 0    | CNCC(C)NC(=O)C1(C(C)(C)OC)CC1                              | 0x00000200000002002021227C488B9C02100615FFCC | 0x00733000000000000000000000001800000000000000000000000000000000000000001E00100000000E6CC18006020002C004000800011010000000000000000000810800000040160080001400000636008000000000000F80000000000000000000000000000000000000000000 | 0x40000000000000000000800000002400000000000000000000000000000000000000001000000200000000000000000000800000000000000000000000000000000000000002000000000002000000000020000000000100000000000000000000000000010000000040000000000000000000020000000800000000000000000000000048000000000000000000000280200000000000000000020000000000000000000000000000000100000000000000020000000000000000000400000001000000000000000000000000000000010004000000000000000000000800000000000000000000800000000000000400000000000000000010000020000000 | 0xE0001400000000000000000000000000000000000000000200000000000000000000000000000000000000000000000000000000000000000000001000401000000000000000000000000400000000000000000000001000000000000000000000100080000004000000000000000000000000000000000000000000000000000000000000000004000800000000000000000000000000001000000200000000000000000000000000000000000020000000000000000000000000000000000000000000000000000000000000000000004000000000000000001000000000000000000000080020000004000000000000000000000000000000000000000080 |
| 1    | CN(C(=O)C1=CC2=C(F)C=C(F)C=C2N1)C1CN(C(=O)CC2=CC=CN=C2O)C1 | 0x00900000002000004011172DAC534CE55EF3EB7FFC | 0x007BB1800000000000000000000000005801600000003C400000000000000001F000001F00100800000C28C19E0C3EC4F3C99200A8033577540082802037222008D921BC6CDC0866F2C295B394710864D611C8D987BE99809E00000000000200000000000000040000000000000000 | 0x00000000000001000000800000200100000100000000000000000000000000020000000000000000040000000008002000000000000000808000000000000000000200000000000000000001000000000020000000000014000000001000200100000000014040000000000000104000000000020100400000000000000040100000110040000000880000200000000000100000000000000400000000000000000000000000000104040000080000000000000000080000000100000000000000000000000000042000000000004000020000000000014000004200200000000000000000008000002040000000000400800000000000000000004001000000 | 0xBE800000000000000001000000000000000080000000080000000000000000000000000000000000000200000000000000000000000900000000000000010000000000010000000000020000000000000000000000000000000000200000000000000080080000000000000000000000040000008000000000002000000080000000000000400004000000000000000010000000000000000000000000000000000000400000000000000014000000000008000000000000000000000000000000000800000000000000000000000400080000000000001000400000000100000000000000000040004000000000002404000000000000000002020040003180 |

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
We recommend using [uv](https://github.com/astral-sh/uv) for fast, reliable dependency management.
To install source for exploration & development:

```sh
git clone https://github.com/ashvardanian/USearchMolecules.git
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

Alternatively, you can install from PyPI:

```sh
uv pip install usearch-molecules
uv pip install "usearch-molecules[dev]" # for fingerprinting & generation of indexes
uv pip install "usearch-molecules[gpu]" # for GPU-accelerated processing with nvMolKit
uv pip install "usearch-molecules[viz]" # for visualization with StreamLit
uv pip install "usearch-molecules[all]" # for all features
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

![USearch Molecules StreamLit demo preview](/assets/USearchMoleculesStreamLitPreview.gif)

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

