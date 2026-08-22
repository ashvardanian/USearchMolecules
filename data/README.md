---
license: apache-2.0
pretty_name: USearch Molecules
size_categories:
- 10B<n<100B
task_categories:
- feature-extraction
tags:
- chemistry
- molecules
- drug-discovery
- cheminformatics
- smiles
- conformers
- molecular-fingerprints
- similarity-search
- vector-search
- 3d
- tabular
- pandas
- dask
- usearch
- rdkit
- numkong
configs:
- config_name: example
  data_files: data/example/parquet/??????????-??????????.parquet
- config_name: example-3d
  data_files: data/example/parquet/*.3D.parquet
- config_name: pubchem
  data_files: data/pubchem/parquet/??????????-??????????.parquet
- config_name: pubchem-3d
  data_files: data/pubchem/parquet/*.3D.parquet
- config_name: gdb13
  data_files: data/gdb13/parquet/??????????-??????????.parquet
- config_name: gdb13-3d
  data_files: data/gdb13/parquet/*.3D.parquet
- config_name: real-3d
  data_files: data/real/parquet/*.3D.parquet
---

# USearch Molecules

7'132'507'184 small molecules with 2D fingerprints, 3D conformers and shape descriptors, indexed for real-time similarity search.

Start with `example`: 2 million molecules drawn from all three sources, 4 GB, carrying everything the larger subsets do.

| Config    |     Molecules | Source                                   |
| :-------- | ------------: | :--------------------------------------- |
| `example` |     2'000'000 | drawn from the three below, not additive |
| `pubchem` |   115'627'267 | NCBI PubChem                             |
| `gdb13`   |   977'468'267 | University of Bern GDB13                 |
| `real`    | 6'039'411'650 | Enamine REAL                             |

Each subset has a `-3d` companion config holding three conformers per molecule.

Enamine REAL is the exception on this mirror: its conformers are here and still growing, while its fingerprint shards live on the S3 mirrors rather than the Hub.

## Loading

The shards are plain Parquet, and `pyarrow`, `pandas`, `polars` and `dask` read both families directly:

```python
import pyarrow.parquet as pq

molecules = pq.read_table("data/example/parquet/0000000000-0001000000.parquet")
geometry = pq.read_table("data/example/parquet/0000000000-0001000000.3D.parquet")
```

The `datasets` library loads the `-3d` configs, but not the fingerprint ones: MACCS, ECFP4, FCFP4 and PubChem are stored as `fixed_size_binary`, which has no `datasets` dtype equivalent.

```python
from datasets import load_dataset

geometry = load_dataset("unum-cloud/USearchMolecules", "example-3d", split="train")
```

Searching the fingerprints is what [USearch](https://github.com/unum-cloud/USearch) is for, rebuilding chemistry from a SMILES string is [RDKit](https://www.rdkit.org), and the Kabsch and Umeyama kernels for comparing conformers come from [NumKong](https://github.com/ashvardanian/NumKong).

## Columns

Fingerprint configs carry one row per molecule:

| Column    | Type          | Description                                             |
| :-------- | :------------ | :------------------------------------------------------ |
| `smiles`  | `utf8`        | Canonical graph: atoms, bonds, charges, stereochemistry |
| `maccs`   | `binary(21)`  | MACCS structural keys, 166 bits                         |
| `pubchem` | `binary(111)` | PubChem substructure fingerprint, 881 bits              |
| `ecfp4`   | `binary(256)` | Extended-connectivity fingerprint, radius 2, 2048 bits  |
| `fcfp4`   | `binary(256)` | Functional-class fingerprint, radius 2, 2048 bits       |

The `-3d` configs carry one row per conformer, three per molecule, lowest energy first:

| Column                                | Type             | Description                                                 |
| :------------------------------------ | :--------------- | :---------------------------------------------------------- |
| `input_shard`, `input_row`            | `utf8`, `uint64` | Join keys back to the fingerprint row                       |
| `smiles`                              | `utf8`           | Carried for convenience                                     |
| `conformer_index`                     | `uint8`          | Energy rank, 0 is lowest                                    |
| `status`                              | `uint8`          | 0 is success; other codes mark why geometry is absent       |
| `n_heavy_atoms`, `n_atoms`, `n_bonds` | `uint16`         | Counts, with and without hydrogens                          |
| `molecular_weight`                    | `float32`        | Exact mass in Daltons                                       |
| `conformer_coords`                    | `list<float16>`  | `3 * n_atoms`, row-major, centroid-centered                 |
| `conformer_energy`                    | `float32`        | MMFF94 energy in kcal/mol                                   |
| `usrcat`                              | `list<float16>`  | 60-dimensional shape descriptor; USR is its first 12 values |

Geometry columns are null wherever `status` is non-zero. Conformers come from ETKDG embedding over RDKit's experimental torsion preferences, followed by MMFF94 relaxation.

## Caveats

Conformer yield is not complete: PubChem reaches three conformers for 97.9 % of molecules, and the shortfall is concentrated above 100 atoms. A `success` status bounds the energy's finiteness rather than its magnitude, so filter on the gap to a molecule's own lowest conformer rather than on absolute energy.

Where a SMILES names more than one fragment, the geometry covers only the largest, while the `smiles` column keeps the whole string. Reproduce the choice with RDKit's `LargestFragmentChooser` under `preferOrganic`.

## More

Pre-built USearch indexes, mirror choices, the SMARTS catalogs and the full methodology live in the [GitHub repository](https://github.com/unum-science/USearchMolecules).

## Citation

```bibtex
@software{Vardanian_USearchMolecules,
  author = {Vardanian, Ash},
  title = {{USearchMolecules: A Multi-Modal Atlas of 7 Billion Small Molecules}},
  doi = {10.5281/zenodo.21613663},
  url = {https://github.com/unum-science/USearchMolecules},
  license = {Apache-2.0}
}
```
