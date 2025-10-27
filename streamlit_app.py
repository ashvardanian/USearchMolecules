import py3Dmol
import streamlit as st
import streamlit.components.v1 as components

from rdkit import Chem
from rdkit.Chem import AllChem

from usearch.eval import measure_seconds

from usearch_molecules.dataset import FingerprintedDataset, shape_mixed

st.set_page_config(
    page_icon="⚗️",
    page_title="USearch Molecules",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.title("USearch Molecules")

max_results = st.sidebar.select_slider(
    "Similar Molecules to Fetch", (1, 10, 100, 1000), 100
)

st.sidebar.markdown(
    """
Only top-3 results are shown in this layout, but more elements are fetched to check for collisions,
as MACCS and ECFP4 representation of different molecules may still be identical.
"""
)

expansion_search = st.sidebar.select_slider(
    "Expansion Factor for Search", (64, 512, 4096), 4096
)

st.sidebar.markdown(
    """
Expansion factor controls the depth of the priority-queue used during HNSW graph traversals.
Higher value means higher accuracy of search. 1024 or higher is recommended to reach over 99% accuracy
on billion-scale molecule collections.
"""
)


@st.cache_resource
def get_dataset():
    data = FingerprintedDataset.open("data/example", shape=shape_mixed)
    data.index.expansion_search = expansion_search
    return data


data = get_dataset()


def interactive(
    smiles,
    width=300,
    height=300,
    background="white",
    style="stick",
    surface=False,
    opacity=0.5,
):
    molecule = Chem.MolFromSmiles(smiles)
    molecule = Chem.AddHs(molecule)
    AllChem.EmbedMolecule(molecule)
    AllChem.MMFFOptimizeMolecule(molecule, maxIters=200)

    assert style in ("line", "stick", "sphere", "carton")
    molecule_block = Chem.MolToMolBlock(molecule)
    viewer = py3Dmol.view(width=width, height=height)

    # Pass styling settings to 3dmol.js `GLViewer` class
    # https://3dmol.csb.pitt.edu/doc/GLViewer.html#GLViewer-title
    viewer.addModel(molecule_block, "mol")
    viewer.setStyle({style: {}})
    viewer.setBackgroundColor(background)
    if surface:
        viewer.addSurface(py3Dmol.SAS, {"opacity": opacity})
    viewer.zoomTo()
    html = viewer._make_html()
    return html


query_smiles = st.text_input("Enter a valid SMILES string", data.random_smiles())

if Chem.MolFromSmiles(query_smiles, sanitize=False) is None:
    st.error("Provided SMILES isn't valid according to RDKit")
else:
    seconds, results = measure_seconds(
        lambda: data.search(
            query_smiles,
            max_results,
            log=st.progress,
        )
    )
    st.success(
        f"Found {len(results)} similar molecules in {seconds:.3f} seconds. Showing top 3 results"
    )

    # Remove the match from results
    results_smiles: list[str] = []
    results_scores: list[float] = []
    found_query = False
    for key, smiles, distance in results:
        if smiles == query_smiles:
            found_query = True
        else:
            results_smiles.append(smiles)
            results_scores.append(distance)

    color_light = "white"
    color_dark = "#F0F2F6"
    color_accent = "#FF4B4B"

    # Prepare data for our tiled layout
    html_query_top_left = interactive(query_smiles, background=color_light)
    html_result_top_right = interactive(results_smiles[0], background=color_dark)
    html_result_bottom_left = interactive(results_smiles[1], background=color_dark)
    html_result_bottom_right = interactive(results_smiles[2], background=color_light)

    title_query_top_left = query_smiles
    title_result_top_right = str(results_smiles[0])
    title_result_bottom_left = str(results_smiles[1])
    title_result_bottom_right = str(results_smiles[2])

    score_query_top_left = "✅" if found_query else "❌"
    score_result_top_right = f"{results_scores[0]:.2f}"
    score_result_bottom_left = f"{results_scores[1]:.2f}"
    score_result_bottom_right = f"{results_scores[2]:.2f}"

    # The following `tiles_html` HTML snippet defines a 2x2 table grid,
    # that fills all of the screen width and all of vertical space after
    # the `st.text_input` search bar and subsequent `st.markdown` results summary.
    # The top-left and bottom-right tiles have lighter gray background, while
    # the top-right and bottom-left have darker one.
    # Every tile contains a horizontally-centered title, a score bubble in the top-right corner,
    # and a large canvas for the interactive 3D render of the molecule.
    tiles_html = f"""
    <style>
        .tile {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 0;
            border-radius: 0;
            position: relative;
            overflow: hidden; 
            font-family: 'Arial', sans-serif; 
        }}
        .light-gray-bg {{
            background-color: {color_light};
        }}
        .dark-gray-bg {{
            background-color: {color_dark};
        }}
        .score-bubble {{
            position: absolute;
            bottom: 10px;
            right: 10px;
            color: {color_accent}; 
            background-color: white;
            border-radius: 50%;
            padding: 5px 10px;
            box-shadow: 0px 0px 5px rgba(0,0,0,0.2);
            font-weight: bold;
        }}
        .tile iframe {{
            width: 100%;
            height: 100%;
            border: none; 
        }}
        .tiles-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 0; 
            height: 80vh;
            width: 100%; 
        }}
        h3 {{
            text-align: center;
            overflow-wrap: break-word; 
            padding: 0 10px; 
            max-width: 90%;
        }}
    </style>

    <div class="tiles-container">
        <div class="tile light-gray-bg">
            <div class="score-bubble">{score_query_top_left}</div>
            <h3>{title_query_top_left}</h3>
            {html_query_top_left}
        </div>
        <div class="tile dark-gray-bg">
            <div class="score-bubble">{score_result_top_right}</div>
            <h3>{title_result_top_right}</h3>
            {html_result_top_right}
        </div>
        <div class="tile dark-gray-bg">
            <div class="score-bubble">{score_result_bottom_left}</div>
            <h3>{title_result_bottom_left}</h3>
            {html_result_bottom_left}
        </div>
        <div class="tile light-gray-bg">
            <div class="score-bubble">{score_result_bottom_right}</div>
            <h3>{title_result_bottom_right}</h3>
            {html_result_bottom_right}
        </div>
    </div>
    """

    components.html(tiles_html, height=800)
