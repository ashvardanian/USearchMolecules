# Reaction templates

SMIRKS transformations of the form `reactant_pattern >> product_pattern` with atom maps `:1`, `:2` ... that track which atoms in the reactant correspond to which atoms in the product.

Used for:

- __Retrosynthesis__ — apply the reverse direction of templates to a target molecule to enumerate plausible precursors. Tools like AiZynthFinder, IBM RXN, and ASKCOS lean on a few hundred-thousand templates per node of the search tree.
- __Library enumeration__ — apply the forward direction of a template to combinations of building blocks to generate virtual product space. Enamine REAL and similar build their 6B-molecule corpora this way.
- __Matched-molecular-pair / Free-Wilson analysis__ — define the "context" and "transformation" parts of a SAR change as SMIRKS.

## Records

| File | Reactions | Origin | License |
|---|---:|---|---|
| `named_reactions.jsonl` | 12 | Hand-encoded canonical medchem reactions, validated by `RunReactants` against a positive-control reactant pair | hand-encoded |

## Schema

Every record carries the standard core (`smirks`, `name`, `source`, `citation`, `license`) plus reaction-specific fields:

- `smirks` — the reaction SMARTS, with `>>` separator and atom maps.
- `rxn_class` — short class label (`amide_coupling`, `suzuki_miyaura`, `buchwald_hartwig`, ...) for grouping templates that perform the same transformation under different conditions.
- `test_reactants` — when present, an array of SMILES strings on which the template was verified to produce at least one product via RDKit's `RunReactants`. Acts as a regression test.

## What's in the starter

The 12 named reactions cover the most-run transformations in modern medicinal chemistry — amide coupling, Suzuki-Miyaura, Buchwald-Hartwig, Sonogashira, click (CuAAC), reductive amination, sulfonamide formation, esterification (Mitsunobu-flavored), Williamson ether, Heck, Wittig, ester hydrolysis, and Boc deprotection.
Each has a `test_reactants` field with input SMILES that exercise the template; the build script asserts these produce at least one product before the record lands in the file.

## Planned additions

Two larger corpora belong here next:

- __Hartenfeller's 58 parallel-synthesis rules__ from Hartenfeller et al. 2011 (_J. Chem. Inf. Model._ 51, 3093). Hand-curated, well-vetted, the most-cited public reaction template set. The original paper's supplementary information has the SMIRKS as a Word table; transcribing them is the most direct path. Public, redistributable.
- __USPTO templates__ extracted by Lowe (1.7M reactions, 50–500k unique templates depending on extraction radius). The cleaned ORDerly version (Wigh et al., _JCIM_ 2024, doi:10.1021/acs.jcim.4c00292) is the standard dataset. CC-BY-SA-4.0 via ORD; downstream restrictions follow from the share-alike clause.

The schema above is forward-compatible: USPTO templates will add `template_radius` (0 / 1 / 2) and `n_uses` (frequency in the corpus) fields; they don't need a different schema family.

## Matching cost

Reaction templates are typically depth ≤ 2 like alert SMARTS, but two characteristics differ:

- __Atom maps are required__ for the template to be usable. Matching a SMIRKS without atom-map perception silently produces wrong products.
- __Directional bonds__ (`/`, `\`) appear in templates that preserve _E_ / _Z_ geometry across the reaction. The starter set has none, but USPTO templates for olefin-handling reactions (Wittig, Heck, Stille of vinyl halides) frequently use them.

Otherwise the cost characteristics — recursive width, ring-bond annotations, multi-fragment patterns — match the structural-alert profile.
