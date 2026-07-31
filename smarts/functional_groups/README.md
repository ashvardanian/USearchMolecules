# Functional-group dictionaries

Descriptive SMARTS that identify what's _in_ a molecule rather than what's _wrong_ with it.
Each pattern names a functional group (alcohol, ester, primary amide, sulfonamide, carbamate, ...) without an alert / reject judgment.

Used for:

- __Descriptor generation__ — a vector of "FG counts" per molecule is a classical chemoinformatics descriptor, complementary to fingerprint similarity.
- __Scaffold and class labeling__ — automated tagging of large libraries by chemical class for navigation and clustering.
- __Substructure-screen pre-filter for SAR queries__ — "show me ECFP4-similar molecules to X that contain a sulfonamide" reduces the ECFP4 candidate set cheaply.
- __Reaction-template applicability hints__ — coupling templates need an acid + an amine; querying for those FGs identifies plausible coupling partners across the corpus.

## Records

| File               | Patterns | Origin                                                                       | License  |
| ------------------ | -------: | ---------------------------------------------------------------------------- | -------- |
| `toxtree_fg.jsonl` |      198 | Toxtree `toxtree-functional_groups` plugin, extracted from Java rule sources | LGPL-2.1 |

## Schema

Standard core (`smarts`, `name`, `source`, `citation`, `license`) plus:

- `rule_id` — the originating rule's ID in the Toxtree source (`FG31_1`, `FG88`, ...).
- `rule_title` — the human-readable rule title from `setTitle(...)`.

## What's in the starter

The 198 Toxtree functional-group patterns cover the full catalog used by Toxtree's toxicology decision trees — alcohols, ethers, amines (primary / secondary / tertiary), amides, esters, carboxylic acids, phenols, anilines, halogenated variants, sulfur and phosphorus FGs, etc.
The patterns are mostly depth 0–1 with light recursion (~120 of 198 use `$()`); they're cheap to match and produce a clean per-atom FG annotation.

A note on overlap: roughly 40–50 of these patterns are __also__ present in the `alerts/` subfolder (specifically in `alerts/iss_ames.jsonl` and `alerts/benigni_bossa.jsonl`) because Toxtree's tox plugins reuse the FG plugin's vocabulary as building blocks.
The duplication is intentional in Toxtree — alerts are FG patterns plus toxicology semantics.
For a single-source view of "what FGs are present," use this file; for "is any FG in this molecule a known toxicophore," consult `alerts/`.

## Planned additions

- __Ertl IFG__ (Ertl 2017, _J. Cheminform._ 9, 36; reference impl in `rdkit/Contrib/IFG/ifg.py`). Algorithmic rather than pattern-based — IFG identifies functional groups dynamically from a small set of primitive SMARTS (atoms-double-bonded-to-heteroatom, acetal carbons, etc.) plus a graph-walk algorithm. To add this as a `.jsonl` we'd run IFG against a representative molecule corpus and emit the discovered FG SMARTS as records; the catalog would be empirical rather than canonical.
- __Curated medchem FG vocabularies__ (Bemis-Murcko-style scaffold types, drug-class privileged motifs).

## Matching cost

Functional groups are the cheapest catalog category — median pattern depth 0, OR-arity 0, no ring-closure heroics.
For a brute-force matcher, the 198 Toxtree patterns ought to apply across a 7B-molecule corpus in roughly the same wall-clock time as running 1–2 of the larger PAINS-C patterns alone.
This is the natural place to start when validating a matcher's correctness — the patterns are simple and the expected hit rates are well-known.
