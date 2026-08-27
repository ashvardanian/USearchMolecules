# SMARTS Catalog

A curated collection of __public, academically and commercially relevant SMARTS__ for chemoinformatics workloads on USearchMolecules' molecule corpus.

The catalog is organized by __use case__ rather than by source.
Each subdirectory holds one use case, and sources within a use case share a JSONL schema with three required fields — `smarts` or `smirks`, `name`, `source` — plus optional ones documented per-source.
Every record carries its own `license` field so the constraint travels with the data.

The starter sets prioritize small, well-vetted, openly-licensed collections, and each one names the scale it grows toward.

| Use case             | Records |    Target | Sources still to fold in                         |
| :------------------- | ------: | --------: | :----------------------------------------------- |
| `alerts/`            |   2'448 | at target | —                                                |
| `functional_groups/` |     198 |    1'000+ | Ertl IFG, DataWarrior, RDKit `Fragments`         |
| `pharmacophores/`    |      27 |       150 | halogen bonds, ring stacking, metal binders      |
| `reactions/`         |      12 |  500'000+ | USPTO via Lowe and ORDerly, Hartenfeller-58      |
| `markush/`           |       0 |  1M to 3M | MarkushGrapher over USPTO, EPO, JPO, KIPO, CNIPA |

## Use Cases

### `alerts/` — Structural Alerts

Reactive-group, PAINS, toxicophore and screening-deck filters.
Used as a substructure-screen layer on top of similarity search — "find ECFP4-similar molecules to X that aren't PAINS-flagged".
25 public sets from Lilly, GSK, BMS, Novartis, Pfizer, AbbVie, the PAINS family, NIH, ZINC, ChEMBL and Toxtree.
This is the most mature subfolder; see [`alerts/README.md`](alerts/README.md) for per-source provenance and complexity tables.

### `functional_groups/` — Descriptive Functional-Group Dictionaries

What is _in_ a molecule rather than what is _wrong_ with it, used for descriptor generation, scaffold classification and human-readable labeling.
The starter is 198 patterns from Toxtree's `toxtree-functional_groups` plugin under LGPL-2.1.
Reaching __1'000+__ distinct groups means folding in Ertl's IFG algorithm, DataWarrior and RDKit's `Fragments` vocabulary, where coverage matters more than precision.
See [`functional_groups/README.md`](functional_groups/README.md).

### `pharmacophores/` — 3D Feature Definitions

SMARTS that identify chemical _roles_ rather than _structures_ — donor, acceptor, aromatic centroid, hydrophobe, positive ionizable.
Used for 3D similarity search and pharmacophore-based virtual screening on the pre-generated conformers.
The starter is RDKit's `BaseFeatures.fdef`, 27 features across 8 families, BSD-3.
A __150__-feature base set adding halogen bonds, aromatic ring stacking and metal binders is what expands into Ph4FP triplet and quadruplet fingerprints, which run to millions of dimensions from a base set this small.
See [`pharmacophores/README.md`](pharmacophores/README.md).

### `reactions/` — Reaction Templates

SMIRKS transformations of the form `reactant >> product` with atom maps, used for retrosynthesis tree search, library enumeration and matched-molecular-pair analysis.
The starter is 12 hand-encoded named medchem reactions — Suzuki, Buchwald-Hartwig, amide coupling, click.
The __500'000+__ target comes from mining the USPTO patent corpus through Lowe extraction and ORDerly cleaning, supplemented by Hartenfeller's 58 parallel-synthesis rules and the Schneider BRAINS and NameRxn named-reaction taxonomies.
See [`reactions/README.md`](reactions/README.md).

### `markush/` — Patent Markush Structures

Generic claim structures with R-group enumerations, used for freedom-to-operate analysis, IP landscape mapping and patent-aware molecule classification.
This subfolder is a placeholder, and the starter is the IBM M2S benchmark.
Full-corpus extraction with MarkushGrapher over bulk USPTO, EPO, JPO, KIPO and CNIPA literature puts __1 to 3 million__ structures in reach, comparable in scope to the commercial CAS MARPAT corpus of roughly 1.1M records.
See [`markush/README.md`](markush/README.md).

### Future Subfolders

The same convention applies — new subfolder, new README, `.jsonl` files carrying the schema documented in that subfolder.

- `bioisosteres/` — transformation pairs from BIOSTER, SwissBioisostere and MMPDB for lead-optimization queries.
- `metabolism/` — site-of-oxidation rules from SMARTCyp-class predictors for ADMET screening.
- `scaffolds/` — privileged scaffolds and drug-class queries such as kinase pyrimidines and GPCR aminergics.

## Schema Conventions

Every record in every subfolder is one JSON object on a single line.
Three fields are required across all use cases:

- `smarts` — the SMARTS string.
  Reactions carry `smirks` instead, which holds `>>` and atom maps.
- `name` — human-readable label.
- `source` — short identifier matching the file basename.

Two fields are conventional but not strictly required:

- `citation` — short citation key.
- `license` — SPDX-style identifier, or `hand-encoded` for primary-source transcriptions.

Anything else is use-case-specific and documented in each subfolder's README.
