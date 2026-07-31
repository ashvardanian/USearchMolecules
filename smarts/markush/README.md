# Patent Markush structures

Generic chemical structures from patent claims, with R-group placeholders and substituent enumerations.
A Markush structure represents a __class__ of molecules that the patent claim covers, not a single compound.

Used for:

- __Freedom-to-operate (FTO) analysis__ — for a given molecule, which patents claim it?
- __IP landscape mapping__ — for a given chemical class, who owns what?
- __Patent-aware similarity search__ — combine ECFP4 nearest-neighbors with "and excluded by no patent priority-dated before 2020."

This subfolder is currently a __placeholder__.
The catalog uses a different extraction path from the other use cases: Markush structures are drawn as 2D images in patents and accompanied by R-group definitions in claim text — neither component is directly available as SMARTS in any free public source.

## Why this is harder than alerts or reactions

The other use cases here have public, machine-readable starter sets.
For Markush, the starter set has to be _produced_:

- __Patent text__ is freely available as XML from USPTO (`bulkdata.uspto.gov`), EPO (`epo.org/searching-for-patents/data/bulk-data-sets.html`), WIPO PATENTSCOPE, and via Google Patents BigQuery — but the structure drawings are external TIFF / PNG files, and the R-group definitions are prose ("R₁ is selected from the group consisting of...").
- __Specific-compound extraction__ from patents is solved (SureChEMBL, PatCID, PubChem Patents) — but those projects __explicitly exclude__ Markush, because Markush requires joint reasoning about an image and the surrounding claim text.
- __Commercial Markush corpora__ exist (CAS MARPAT, ChemAxon JChem Markush, Reaxys Markush) but cannot be redistributed.

## Planned starter

__M2S benchmark__ from IBM's MarkushGrapher project (Morin et al., CVPR 2025, arXiv:2503.16096; repo at https://github.com/DS4SD/MarkushGrapher).
M2S is the first openly-annotated Markush set — a few hundred patterns from real USPTO / EPO / WIPO patents, MIT-licensed.
The MarkushGrapher repository ships sample images and a vocabulary, but the per-image SMARTS-style annotations live in the model's evaluation data (released alongside the model weights when those land).
Once the labels are downloadable, populating `m2s_benchmark.jsonl` is mechanical.

## Long-term direction

For full-corpus Markush extraction across USPTO + EPO + JPO + CNIPA + KIPO patents (estimated 1–5M Markush structures globally), the path is:

1. __Layout analysis__ — page-level extraction of figure regions from patent PDFs/TIFFs. IBM Deep Search and PatCID handle this end-to-end.
2. __Optical chemical-structure recognition with R-group awareness__ — MarkushGrapher specifically. Vision encoder + claim-text encoder → SMARTS-with-variables decoder.
3. __Validation__ — 70–80% structural accuracy is reported; a 20–30% error tail needs human review or a model ensemble.

A back-of-envelope on 8× RTX 6000 Blackwell put the OCSR step at ~55 hours wall-clock for ~10M Markush images, with the bottleneck being preprocessing and validation rather than inference.
This is feasible and would create the first public Markush corpus at MARPAT-comparable coverage.

## Schema (planned)

```json
{
  "smarts": "[*:R1]c1ccc(cc1)C(=O)N[*:R2]",
  "name": "Anilide-class Markush",
  "source": "m2s",
  "patent_id": "US20230012345A1",
  "claim_number": 1,
  "priority_date": "2021-06-15",
  "assignee": "Example Pharma Inc.",
  "r_groups": {
    "R1": ["[#6;X4]", "[#6;X3]:[#6]"],
    "R2": ["[#6;X4]", "[F,Cl,Br,I]"]
  },
  "license": "MIT (M2S benchmark)"
}
```

The two new fields beyond the standard core are `r_groups` (mapping R-label → array of allowed substituent SMARTS) and patent metadata (`patent_id`, `claim_number`, `priority_date`, `assignee`).

## Matching cost

Markush patterns are categorically more expensive than alerts or reactions: they're often __depth-3 or depth-4 recursive__ when the R-group expansions themselves contain nested SMARTS, and a single Markush can match by enumerating combinations of R-groups (each R-group with N alternatives → N¹ × N² × ... combinations).
This is exactly the "complex SMARTS that benefits from a brute-force matcher" workload we discussed — pure linear scan over the molecule corpus, no fingerprint pre-filter, but parallelizable across millions of patterns.
