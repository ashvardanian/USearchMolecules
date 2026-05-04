# SMARTS catalog

A curated collection of __public, academically and commercially relevant SMARTS__ for chemoinformatics workloads on USearchMolecules' molecule corpus.

The catalog is organized by __use case__ rather than by source.
Each subdirectory holds one use case; sources within a use case share a JSONL schema with three required fields (`smarts` or `smirks`, `name`, `source`) plus optional ones documented per-source.
Every record carries its own `license` field so the constraint travels with the data.

The starter dataset prioritizes small, well-vetted, openly-licensed collections.
Larger corpora — USPTO reaction templates, patent Markush, etc. — will be folded in over time as they're extracted.

## Use cases

### `alerts/` — Structural alerts

Reactive-group, PAINS, toxicophore, and screening-deck filters.
Used as a substructure-screen layer on top of similarity search ("find ECFP4-similar molecules to X that aren't PAINS-flagged").
Sources include 25 public sets from Lilly, GSK, BMS, Novartis, Pfizer, AbbVie, the PAINS family, NIH, ZINC, ChEMBL, and Toxtree.
This is the most mature subfolder; see [`alerts/README.md`](alerts/README.md) for per-source provenance and complexity tables.

### `reactions/` — Reaction templates

SMIRKS transformations (`reactant >> product`) with atom maps, used for retrosynthesis tree search, library enumeration, and matched-molecular-pair analysis.
Starter: a hand-encoded set of named medchem reactions (Suzuki, Buchwald-Hartwig, amide coupling, click, ...).
Planned additions: Hartenfeller's 58 parallel-synthesis rules, USPTO templates extracted via Lowe + ORDerly cleaning.
See [`reactions/README.md`](reactions/README.md).

### `markush/` — Patent Markush structures

Generic claim structures with R-group enumerations.
Used for freedom-to-operate analysis, IP landscape mapping, and patent-aware molecule classification.
This subfolder is currently a placeholder; the planned starter is the IBM M2S benchmark, with a long-term goal of full-corpus extraction via MarkushGrapher on USPTO/EPO/JPO/CNIPA bulk data.
See [`markush/README.md`](markush/README.md).

### `pharmacophores/` — 3D feature definitions

SMARTS that identify chemical _roles_ rather than _structures_ — donor, acceptor, aromatic centroid, hydrophobe, positive ionizable, etc.
Used for 3D similarity search and pharmacophore-based virtual screening on USearchMolecules' pre-generated conformers.
Starter: RDKit's `BaseFeatures.fdef` (27 features across 8 families), BSD-3.
See [`pharmacophores/README.md`](pharmacophores/README.md).

### `functional_groups/` — Descriptive functional-group dictionaries

What's _in_ a molecule rather than what's _wrong_ with it.
Used for descriptor generation, scaffold classification, and human-readable labeling.
Starter: 198 patterns from Toxtree's `toxtree-functional_groups` plugin (LGPL-2.1).
See [`functional_groups/README.md`](functional_groups/README.md).

### Future subfolders

The same convention applies — new subfolder, new README, drop in `.jsonl` files with the schema documented in the subfolder README.
Plausible additions:

- `bioisosteres/` — transformation pairs from BIOSTER / SwissBioisostere / MMPDB for lead-optimization queries.
- `metabolism/` — site-of-oxidation rules from SMARTCyp-class predictors for ADMET screening.
- `scaffolds/` — privileged scaffolds and drug-class queries (kinase pyrimidines, GPCR aminergic, etc.).

## Schema conventions

Every record in every subfolder is one JSON object on a single line.
Three fields are required across all use cases:

- `smarts` — the SMARTS string. (Reactions use `smirks` instead, which contains `>>` and atom maps.)
- `name` — human-readable label.
- `source` — short identifier matching the file basename.

Two fields are conventional but not strictly required:

- `citation` — short citation key.
- `license` — SPDX-style identifier or `hand-encoded` for primary-source transcriptions.

Anything else is use-case-specific and documented in each subfolder's README.
