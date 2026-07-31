# 3D pharmacophores

SMARTS that identify chemical _roles_ rather than _structures_ — donor, acceptor, aromatic centroid, hydrophobic group, positively or negatively ionizable, zinc-binder, etc.
The matching atom or atom-set is the "feature"; the matched 3D position becomes the input to pharmacophore-based virtual screening, 3D similarity (ROCS-flavored), and de-novo design.

Used for:

- __Pharmacophore-based virtual screening__ — define a query as a 3D arrangement of features (e.g. "donor at position A, aromatic centroid 3.5 Å away at position B, hydrophobe 5 Å further") and find molecules whose conformers match the geometry.
- __3D similarity__ — Ph4FP-style fingerprints encode the multiset of feature triplets / quadruplets in a molecule, providing a 3D-aware analog to ECFP.
- __De-novo design scoring__ — generative models score candidate molecules by how well their predicted conformers match a target pharmacophore.

## Records

| File | Features | Origin | License |
|---|---:|---|---|
| `rdkit_basefeatures.jsonl` | 27 | RDKit's built-in `BaseFeatures.fdef`, AtomType references expanded | BSD-3 |

## Schema

Standard core (`smarts`, `name`, `source`, `citation`, `license`) plus pharmacophore-specific fields:

- `family` — the feature category. RDKit BaseFeatures uses 8 families: `Donor`, `Acceptor`, `NegIonizable`, `PosIonizable`, `Aromatic`, `Hydrophobe`, `LumpedHydrophobe`, `ZnBinder`.

Future sources may add a `weights` field (per-atom contribution weights for centroid placement; RDKit's `.fdef` carries these but the dict-format extraction collapses them).

## What's in the starter

RDKit's default pharmacophore feature set, identical to the one MOE / Phase / OpenEye-flavored tools use as a baseline.
The 27 features cover:

- 1× `Donor` (single-atom H-bond donor)
- 1× `Acceptor` (single-atom H-bond acceptor)
- 1× `NegIonizable` (carboxylic / sulfonic / phosphonic acidic group)
- 4× `PosIonizable` (basic amines, imidazole, guanidine, quaternary nitrogen variants)
- 6× `Aromatic` (4-, 5-, 6-, 7-, 8-membered aromatic rings + bridged variants)
- 9× `Hydrophobe` / `LumpedHydrophobe` (chain hydrophobes, ring hydrophobes, lumped multi-atom hydrophobic groups)
- 5× `ZnBinder` (metal-chelating motifs for zinc-metalloenzymes — HDAC, MMP, etc.)

The SMARTS are unusually heavy on recursive `$()` predicates because feature definitions need to exclude many edge cases (a pyrrole nitrogen donor must exclude indole-like contexts, an acceptor oxygen must exclude N-oxide oxygens, etc.).
The single-atom donor SMARTS is one of the longest in the catalog.

## Planned additions

- __Per-source variants__ from MOE, Schrödinger Phase, or Astex's published feature definitions, when redistributable.
- __Halogen-bond donor / acceptor__ features (relevant for protein-ligand contacts but absent from RDKit's default set).
- __Pi-stacking__ features (face / edge aromatic interactions).

## How this maps onto USearchMolecules

The catalog is most useful __per-conformer__: for each conformer of each molecule in the corpus, run the pharmacophore SMARTS to record the atom-index and 3D position of each feature.
That precompute becomes the input to a Ph4FP-style 3D similarity index, complementary to the existing 2D ECFP4 / MACCS / FCFP4 / PubChem fingerprints.
Conformer generation is already part of the USearchMolecules pipeline; adding the 27-feature scan is a small marginal cost.
