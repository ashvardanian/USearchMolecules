"""Interactive web app for molecular similarity search with 3D visualization.

Search for similar molecules using SMILES notation and view results in an interactive
3D viewer. Uses USearch for fast approximate nearest neighbor search on molecular
fingerprints, with py3Dmol for WebGL-based molecular visualization.

Geometry comes from the dataset's own `*.3D.parquet` shards — ETKDG-embedded and
MMFF94-relaxed ahead of time — and is only re-derived in the browser session for a
query molecule that isn't in the corpus.

Run with:
    uv run streamlit run streamlit_app.py

Learn more:
- SMILES: https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system
- Streamlit: https://docs.streamlit.io
- 3Dmol.js: https://3dmol.csb.pitt.edu
"""

import numkong as nk
import numpy as np
import py3Dmol
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Geometry import Point3D
from usearch.eval import measure_seconds

from usearchmolecules.dataset import FingerprintedDataset, shape_mixed

TILES = 4  # One query molecule and its three nearest neighbours
MCS_TIMEOUT = 1  # Seconds before a common-substructure search gives up on a pair
# Fewer atoms superimpose more easily, so RMSD flatters a thin match: across a sample, pairs
# sharing six atoms or fewer aligned to 0.15 A against 1.07 A for twelve or more. Requiring the
# correspondence to cover half the query keeps the comparison between molecules that mostly overlap.
MCS_MIN_COVERAGE = 0.5

st.set_page_config(
    page_icon="⚗️",
    page_title="USearchMolecules",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.title("USearchMolecules")

max_results = st.sidebar.slider("Similar Molecules to Fetch", 1, 1000, 100)

st.sidebar.caption(
    "Only the top 3 are shown. More are fetched to catch collisions, since different molecules "
    "can share a MACCS or ECFP4 fingerprint."
)

# Stepped by 16 because a queue one slot deeper changes nothing a reader could notice.
expansion_search = st.sidebar.slider("Expansion Factor for Search", 16, 8192, 4096, step=16)

st.sidebar.caption(
    "Depth of the priority queue during HNSW traversal. Higher is more accurate; 1024 and above "
    "reaches over 99% on billion-scale collections."
)

rerank_depth = st.sidebar.slider("Candidates to Rerank by Shape", 0, 200, 50)

st.sidebar.caption(
    "Reorders candidates by Kabsch RMSD over the substructure they share with the query, so how a "
    "molecule sits in space breaks ties its fingerprint cannot. 0 ranks by fingerprint alone."
)


@st.cache_resource
def get_dataset():
    """Load and cache the molecular dataset with USearch index."""
    return FingerprintedDataset.open("data/example", shape=shape_mixed)


@st.cache_resource
def get_fragment_chooser():
    """The pipeline's own fragment choice, so rebuilt molecules match the stored coordinates."""
    return rdMolStandardize.LargestFragmentChooser(preferOrganic=True)


data = get_dataset()
# Outside `get_dataset`, or the cached loader would pin whatever the slider read on first run.
data.index.expansion_search = expansion_search


def stored_molecule(smiles: str, key: int) -> Chem.Mol | None:
    """Rebuild `smiles` around the conformer stored for `key`, or None when the shards hold none.

    Stored coordinates cover the largest fragment after `AddHs`, so the molecule has to be
    reduced the same way before its atoms will line up with them.
    """
    stored = data.conformer(key)
    if stored is None:
        return None
    coordinates, _energy = stored

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    if len(Chem.GetMolFrags(molecule)) > 1:
        molecule = get_fragment_chooser().choose(molecule)
        Chem.SanitizeMol(molecule)
    molecule = Chem.AddHs(molecule)
    if molecule.GetNumAtoms() != len(coordinates):
        return None

    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for atom, (x, y, z) in enumerate(np.asarray(coordinates, dtype=np.float64)):
        conformer.SetAtomPosition(atom, Point3D(x, y, z))
    molecule.AddConformer(conformer, assignId=True)
    return molecule


def embedded_molecule(smiles: str) -> Chem.Mol | None:
    """Fall back to a fresh ETKDG embedding for a molecule the corpus doesn't cover."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    molecule = Chem.AddHs(molecule)
    if AllChem.EmbedMolecule(molecule) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(molecule, maxIters=200)
    return molecule


def geometry(smiles: str, key: int | None) -> tuple[Chem.Mol | None, str]:
    """A renderable molecule for `smiles` plus a label saying where its coordinates came from."""
    if key is not None:
        molecule = stored_molecule(smiles, key)
        if molecule is not None:
            _coordinates, energy = data.conformer(key)
            return molecule, f"MMFF94 relaxed · {energy:.1f} kcal/mol"
    return embedded_molecule(smiles), "ETKDG · embedded in this session"


def shape_rmsd(query: Chem.Mol, hit: Chem.Mol) -> tuple[float, int] | None:
    """Kabsch RMSD in Angstroms and the atom count it was measured over, or None if incomparable.

    Both molecules must be hydrogen-free and carry a conformer. Kabsch consumes a one-to-one
    correspondence, which two different molecules do not have until a maximum common substructure
    supplies one; a pair sharing too little of the query is reported as incomparable rather than
    scored, because a short correspondence superimposes well no matter how unlike the molecules are.
    """
    # Three non-collinear points already fix a rotation, so that is the floor coverage cannot go under.
    floor = max(3, round(MCS_MIN_COVERAGE * query.GetNumAtoms()))
    common = rdFMCS.FindMCS(
        [query, hit],
        timeout=MCS_TIMEOUT,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
    )
    if common.canceled or common.numAtoms < floor:
        return None

    pattern = Chem.MolFromSmarts(common.smartsString)
    query_atoms = query.GetSubstructMatch(pattern)
    hit_atoms = hit.GetSubstructMatch(pattern)
    if not query_atoms or len(query_atoms) != len(hit_atoms):
        return None

    query_xyz = query.GetConformer().GetPositions()[list(query_atoms)]
    hit_xyz = hit.GetConformer().GetPositions()[list(hit_atoms)]
    alignment = nk.kabsch(
        np.ascontiguousarray(query_xyz, dtype=np.float16),
        np.ascontiguousarray(hit_xyz, dtype=np.float16),
    )
    return float(alignment.rmsd), len(query_atoms)


def interactive(molecule: Chem.Mol, width=300, height=300, background="white"):
    """Render an RDKit molecule as a standalone 3Dmol.js viewer, returned as HTML."""
    viewer = py3Dmol.view(width=width, height=height)

    # Pass styling settings to 3dmol.js `GLViewer` class
    # https://3dmol.csb.pitt.edu/doc/GLViewer.html#GLViewer-title
    viewer.addModel(Chem.MolToMolBlock(molecule), "mol")
    viewer.setStyle({"stick": {}})
    viewer.setBackgroundColor(background)
    viewer.zoomTo()
    return viewer._make_html()


if "query_smiles" not in st.session_state:
    # Seeded once. Re-rolling it every rerun would change the widget's identity and throw away
    # whatever the visitor typed, leaving the box searchable only by accident.
    st.session_state.query_smiles = data.random_smiles()

if st.sidebar.button("Random molecule"):
    st.session_state.query_smiles = data.random_smiles()

query_smiles = st.text_input("Enter a valid SMILES string", key="query_smiles")

# The same strict parse the fingerprinting path runs, so nothing slips through to a traceback.
if Chem.MolFromSmiles(query_smiles) is None:
    st.error("Provided SMILES isn't valid according to RDKit")
    st.stop()

seconds, results = measure_seconds(lambda: data.search(query_smiles, max_results))

results_smiles: list[str] = []
results_keys: list[int] = []
results_scores: list[float] = []
query_key: int | None = None
for key, smiles, distance in results:
    if smiles == query_smiles:
        query_key = key
    else:
        results_smiles.append(smiles)
        results_keys.append(key)
        results_scores.append(distance)

shown = min(TILES - 1, len(results_smiles))
st.success(f"Found {len(results)} similar molecules in {seconds:.3f} seconds. Showing top {shown}")
if shown == 0:
    st.warning("No neighbours left to show. Raise 'Similar Molecules to Fetch' in the sidebar.")
    st.stop()

query_molecule, query_provenance = geometry(query_smiles, query_key)

# Only the reranked window and the shown tiles are ever built, not the whole result list.
rendered: dict[int, tuple[Chem.Mol | None, str]] = {}


def rendered_geometry(index: int) -> tuple[Chem.Mol | None, str]:
    if index not in rendered:
        rendered[index] = geometry(results_smiles[index], results_keys[index])
    return rendered[index]


order = list(range(len(results_smiles)))
aligned: dict[int, tuple[float, int] | None] = {}
if rerank_depth and query_molecule is not None:
    query_heavy = Chem.RemoveHs(query_molecule)
    candidates = order[:rerank_depth]
    progress = st.progress(0.0, "Aligning candidates with Kabsch")
    for done, i in enumerate(candidates, start=1):
        molecule, _provenance = rendered_geometry(i)
        aligned[i] = shape_rmsd(query_heavy, Chem.RemoveHs(molecule)) if molecule is not None else None
        progress.progress(done / len(candidates), "Aligning candidates with Kabsch")
    progress.empty()
    # Incomparable pairs keep their fingerprint order behind everything that did align.
    # `.get(i, default)` would not do: such a pair stores None, which the default never replaces.
    order.sort(key=lambda i: (aligned.get(i) is None, (aligned.get(i) or (0.0,))[0]))

st.caption(
    "Each tile is labelled on the left with the geometry it draws, and on the right with its "
    "distance from the query — Tanimoto over fingerprints, where 0 is an identical fingerprint, "
    + ("and Kabsch RMSD over the atoms both molecules share." if rerank_depth else "which is also the ranking.")
)

color_light = "white"
color_dark = "#F0F2F6"
color_accent = "#FF4B4B"


def score_label(i: int) -> str:
    tanimoto = f"Tanimoto {results_scores[i]:.2f}"
    match = aligned.get(i)
    if match is None:
        return tanimoto
    rmsd, atoms = match
    return f"RMSD {rmsd:.2f} Å over {atoms} atoms · {tanimoto}"


tiles = [(query_smiles, None, "Query · in corpus" if query_key is not None else "Query · not in corpus")]
tiles += [(results_smiles[i], i, score_label(i)) for i in order[:shown]]

tiles_html = ""
for position, (smiles, index, score) in enumerate(tiles):
    molecule, provenance = (query_molecule, query_provenance) if index is None else rendered_geometry(index)
    background = color_light if position % 3 == 0 else color_dark
    body = interactive(molecule, background=background) if molecule else "<p>No renderable geometry</p>"
    tiles_html += f"""
        <div class="tile" style="background-color: {background}">
            <h3>{smiles}</h3>
            {body}
            <div class="footer">
                <span class="badge">{provenance}</span>
                <span class="badge score">{score}</span>
            </div>
        </div>
    """

# The following snippet defines a 2x2 table grid, that fills all of the screen width and all of
# vertical space after the `st.text_input` search bar and subsequent results summary.
# The top-left and bottom-right tiles have lighter gray background, while the top-right and
# bottom-left have darker one. Every tile contains a horizontally-centered title, a footer row
# naming its geometry and score, and a large canvas for the interactive 3D render of the molecule.
components.html(
    f"""
    <style>
        .tile {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
            font-family: 'Arial', sans-serif;
        }}
        .footer {{
            position: absolute;
            bottom: 10px;
            left: 12px;
            right: 12px;
            display: flex;
            justify-content: space-between;
            gap: 8px;
        }}
        .badge {{
            background-color: white;
            border-radius: 12px;
            padding: 4px 10px;
            box-shadow: 0px 0px 5px rgba(0,0,0,0.2);
            font-size: 12px;
            color: #666;
            white-space: nowrap;
        }}
        .score {{
            color: {color_accent};
            font-weight: bold;
        }}
        .tiles-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            height: 800px;
            width: 100%;
        }}
        h3 {{
            text-align: center;
            overflow-wrap: break-word;
            padding: 0 10px;
            max-width: 90%;
        }}
    </style>

    <div class="tiles-container">{tiles_html}</div>
    """,
    height=800,
)
