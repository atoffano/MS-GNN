import torch
import networkx as nx
import matplotlib.pyplot as plt
import os
import pymol2
import logging
from src.utils.helpers import timeit

import os
import numpy as np
import requests
from typing import Dict, Sequence, Tuple


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


UNIPROT_JSON_URL = "https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
PDB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
ALPHAFOLD_URL = "https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.pdb"


def _download_pdb(uniprot_id: str, dest_path: str) -> bool:
    try:
        response = requests.get(
            UNIPROT_JSON_URL.format(uniprot_id=uniprot_id), timeout=15
        )
        response.raise_for_status()
    except requests.RequestException:
        return False

    data = response.json()
    for ref in data.get("uniProtKBCrossReferences", []):
        if ref.get("database") != "PDB":
            continue
        pdb_id = ref.get("id")
        if not pdb_id:
            continue
        try:
            pdb_resp = requests.get(PDB_DOWNLOAD_URL.format(pdb_id=pdb_id), timeout=15)
            pdb_resp.raise_for_status()
        except requests.RequestException:
            continue
        with open(dest_path, "wb") as handle:
            handle.write(pdb_resp.content)
        return True
    return False


def _download_alphafold(uniprot_id: str, dest_path: str) -> bool:
    try:
        response = requests.get(ALPHAFOLD_URL.format(uniprot_id=uniprot_id), timeout=15)
        response.raise_for_status()
    except requests.RequestException:
        return False

    with open(dest_path, "wb") as handle:
        handle.write(response.content)
    return True


def ensure_structure(uniprot_id: str, out_dir: str) -> str:
    """
    Download a structure for the UniProt ID, preferring PDB (RCSB) entries,
    falling back to AlphaFold if no experimental structure exists.
    Returns the local PDB path.
    """
    os.makedirs(out_dir, exist_ok=True)
    pdb_path = os.path.join(out_dir, f"{uniprot_id}.pdb")
    if os.path.exists(pdb_path):
        logger.info(f"Got PDB structure for {uniprot_id} from local cache")
        return pdb_path

    if _download_pdb(uniprot_id, pdb_path):
        logger.info(f"Downloaded PDB structure for {uniprot_id} from RCSB")
        return pdb_path
    if _download_alphafold(uniprot_id, pdb_path):
        logger.info(f"Downloaded PDB structure for {uniprot_id} from AlphaFold")
        return pdb_path
    raise FileNotFoundError(
        f"No structure available for UniProt ID {uniprot_id} from PDB or AlphaFold"
    )


def render_structure_colormap(
    pdb_path: str,
    residue_scores: Sequence[Tuple[int, float]],
    image_path: str,
    *,
    colormap: str = "viridis",
    title: str | None = None,
    normalize: bool = True,
) -> None:
    """
    Color a structure by residue scores via PyMOL and save to image_path.
    residue_scores: iterable of (1-based residue index, score).
    """
    if pymol2 is None:
        raise RuntimeError(
            "pymol2 not installed. Install pymol-open-source>=2.5 and pymol2."
        )

    if not residue_scores:
        logger.warning(f"No residue scores provided for {pdb_path}, skipping rendering")
        return

    res_idx, scores = zip(*residue_scores)
    res_idx = np.asarray(res_idx, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32)
    if normalize:
        lo, hi = float(scores.min()), float(scores.max())
        if hi - lo < 1e-9:
            hi = lo + 1e-6
    else:
        lo, hi = None, None

    os.makedirs(os.path.dirname(image_path), exist_ok=True)

    with pymol2.PyMOL() as pymol:
        cmd = pymol.cmd
        cmd.reinitialize()
        cmd.set("fetch_path", os.path.dirname(pdb_path))
        cmd.load(pdb_path, "prot")
        cmd.alter("prot", "b=0.0")
        for idx, value in zip(res_idx, scores):
            cmd.alter(f"prot and resi {int(idx)}", f"b={float(value)}")
        cmd.rebuild()
        if normalize:
            cmd.spectrum("b", colormap, "prot", minimum=lo, maximum=hi)
        else:
            cmd.spectrum("b", colormap, "prot")
        if title:
            cmd.set_title("title", state=0, text=title)
        cmd.set("ray_opaque_background", 0)
        cmd.bg_color("white")
        cmd.orient("prot")
        cmd.png(image_path, width=1600, height=1200, dpi=300, ray=1)
    logger.info(f"Saved structure rendering to {image_path}")


@timeit
def plot_systemic_explanation(path, hetero_explanation, dataset):
    """
    Plot explanations from a heterogeneous explainer using the SwissProtDataset
    for protein name resolution and protein ID mapping.
    Produces a plot per batch protein, normalized per subgraph, labeling nodes with their full names.
    Simplified: assume all nodes in the returned batch are involved.
    """
    batch_proteins = hetero_explanation.batch["protein"].detach().cpu()
    edge_mask = (
        hetero_explanation[("protein", "aligned_with", "protein")]["edge_mask"]
        .detach()
        .cpu()
    )
    edge_index = (
        hetero_explanation[("protein", "aligned_with", "protein")]["edge_index"]
        .detach()
        .cpu()
    )

    # edge_mask_z = zscore(edge_mask)
    edge_mask_z = edge_mask

    src_local, dst_local = edge_index[0], edge_index[1]

    G = nx.Graph()
    for local_idx, global_idx in enumerate(batch_proteins["n_id"].tolist()):
        G.add_node(
            local_idx,
            global_id=global_idx,
            label=dataset.idx_to_protein.get(global_idx, str(global_idx)),
        )
    for local_idx in range(edge_index.size(1)):
        G.add_edge(
            src_local[local_idx].item(),
            dst_local[local_idx].item(),
            color=float(edge_mask_z[local_idx].item()),
        )

    edge_colors = [data for (_, _, data) in G.edges(data="color")]
    labels = {n: G.nodes[n]["label"] for n in G.nodes()}

    pos = nx.spring_layout(G, seed=42)
    nodes = nx.draw_networkx_nodes(G, pos)
    edges = nx.draw_networkx_edges(
        G, pos, edge_color=edge_colors, edge_cmap=plt.cm.viridis
    )
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

    protein = dataset.idx_to_protein[batch_proteins["n_id"][0].item()]
    plt.title(f"Protein-Protein Explanation: {protein}")
    plt.colorbar(nodes, label="Node color (edge z-score mean)")
    plt.colorbar(edges, label="Edge z-score")
    filename = f"{path}/explanations/{protein}_system_explanation.png"
    os.makedirs(f"{path}/explanations", exist_ok=True)
    plt.savefig(filename)
    plt.show()
    plt.close()


@timeit
def plot_protein_explanation(path, hetero_explanation, dataset):
    """
    Generate one amino-acid-to-protein explanation plot per protein node in the batch
    as a scatter plot ordered by residue, coloring points by edge-mask z-score.
    """
    edge_mask = (
        hetero_explanation[("aa", "belongs_to", "protein")]["edge_mask"].detach().cpu()
    )
    edge_index = (
        hetero_explanation[("aa", "belongs_to", "protein")]["edge_index"].detach().cpu()
    )

    edge_mask_z = edge_mask

    # edge_mask_z = zscore(edge_mask)
    # edge_mask_z = torch.log1p(edge_mask_z)

    src_local, dst_local = edge_index[0], edge_index[1]

    protein_batch = hetero_explanation.batch["protein"].detach().cpu()
    protein_global_ids = protein_batch["n_id"].tolist()
    root_global = int(protein_global_ids[0])
    root_label = dataset.idx_to_protein.get(root_global, str(root_global))

    explanations_dir = os.path.join(path, "explanations")
    os.makedirs(explanations_dir, exist_ok=True)

    for dst_val in torch.unique(dst_local, sorted=True).tolist():
        mask = dst_local == dst_val
        if torch.count_nonzero(mask) == 0:
            continue

        aa_indices = src_local[mask]
        edge_importance_z = edge_mask_z[mask].view(-1)

        target_idx = int(dst_val)
        target_global = int(protein_global_ids[target_idx])
        target_label = dataset.idx_to_protein.get(target_global, str(target_global))

        sort_idx = torch.argsort(aa_indices)
        aa_sorted = aa_indices[sort_idx]
        edge_z_sorted = edge_importance_z[sort_idx]

        x_positions = torch.arange(len(aa_sorted), dtype=torch.float32)

        plt.figure(figsize=(8, 4))
        scatter = plt.scatter(
            x_positions.numpy(),
            edge_z_sorted.numpy(),
            c=edge_z_sorted.numpy(),
            cmap=plt.cm.viridis,
        )
        plt.colorbar(scatter, label="Edge z-score")
        plt.xlabel("Residue")
        plt.ylabel("Edge Importance")
        plt.title(f"AA-Protein Explanation: {target_label}")
        plt.tight_layout()

        filename = os.path.join(
            explanations_dir,
            f"{root_label}_{target_label}_aa_explanation.png",
        )
        plt.savefig(filename)
        plt.show()
        plt.close()


def _mean_attention(attention_weights: torch.Tensor) -> torch.Tensor:
    """Averages attention weights over all heads if multiple heads are present."""
    if attention_weights.dim() > 1:
        return attention_weights.mean(dim=-1)
    return attention_weights.view(-1)


def analyze_attention_captum_correlation(
    output_dir: str,
    dataset,
    batch,
    attentions,
    hetero_explanation,
    *,
    layer_to_plot: int = 2,
) -> None:
    key = ("aa", "belongs_to", "protein")

    captum_edge_index = hetero_explanation[key]["edge_index"].detach().cpu()
    captum_scores = hetero_explanation[key]["edge_mask"].detach().cpu().view(-1)
    edge_to_captum = {
        (int(captum_edge_index[0, i]), int(captum_edge_index[1, i])): float(
            captum_scores[i].item()
        )
        for i in range(captum_edge_index.size(1))
    }

    scatter_data = None
    for layer_idx, layer_attention in enumerate(attentions, start=1):
        if layer_attention is None or key not in layer_attention:
            continue
        edge_index, attn_weights = layer_attention[key]
        edge_index = edge_index.detach().cpu()
        attn_vals = _mean_attention(attn_weights.detach().cpu())

        shared_attn, shared_captum = [], []
        for i in range(edge_index.size(1)):
            edge = (int(edge_index[0, i]), int(edge_index[1, i]))
            captum_val = edge_to_captum.get(edge)
            if captum_val is None:
                continue
            shared_attn.append(float(attn_vals[i].item()))
            shared_captum.append(captum_val)

        if len(shared_attn) < 2:
            continue
        attn_arr = np.asarray(shared_attn, dtype=np.float32)
        captum_arr = np.asarray(shared_captum, dtype=np.float32)
        # zscore normalization
        attn_arr = (attn_arr - attn_arr.mean()) / (attn_arr.std() + 1e-9)
        captum_arr = (captum_arr - captum_arr.mean()) / (captum_arr.std() + 1e-9)

        if (
            np.std(attn_arr) < 1e-12
            or np.std(captum_arr) < 1e-12
            or np.isnan(attn_arr).any()
            or np.isnan(captum_arr).any()
        ):
            corr_val = float("nan")
        else:
            corr_val = float(np.corrcoef(attn_arr, captum_arr)[0, 1])

        logger.info(
            f"Pearson correlation between attention layer {layer_idx} and Captum: {corr_val:.4f}"
        )

        if layer_idx == layer_to_plot:
            scatter_data = (attn_arr, captum_arr)

    if scatter_data is None:
        return

    attn_arr, captum_arr = scatter_data
    explanations_dir = os.path.join(output_dir, "explanations")
    os.makedirs(explanations_dir, exist_ok=True)
    root_global = int(batch["protein"].n_id.detach().cpu()[0].item())
    root_label = dataset.idx_to_protein.get(root_global, str(root_global))

    plt.figure(figsize=(6, 5))
    plt.scatter(attn_arr, captum_arr, alpha=0.6)
    # Add regression line in red
    m, b = np.polyfit(attn_arr, captum_arr, 1)
    plt.plot(attn_arr, m * attn_arr + b, color="red")
    plt.xlabel(f"Attention Layer {layer_to_plot}")
    plt.ylabel("Captum Score")
    plt.title(f"Attention vs Captum (Layer {layer_to_plot}) – {root_label}")
    plt.tight_layout()
    plot_path = os.path.join(
        explanations_dir,
        f"{root_label}_attn_layer{layer_to_plot}_captum_scatter.png",
    )
    plt.savefig(plot_path)
    plt.close()


def plot_systemic_attention(path, layer_attention, dataset, batch, layer_idx):
    key = ("protein", "aligned_with", "protein")
    if layer_attention is None or key not in layer_attention:
        return
    edge_index, attn_weights = layer_attention[key]
    edge_index = edge_index.detach().cpu()
    attn_values = _mean_attention(attn_weights.detach().cpu())

    protein_ids = batch["protein"].n_id.detach().cpu().tolist()
    G = nx.Graph()
    for local_idx, global_idx in enumerate(protein_ids):
        G.add_node(
            local_idx,
            global_id=global_idx,
            label=dataset.idx_to_protein.get(global_idx, str(global_idx)),
        )

    src_local, dst_local = edge_index[0], edge_index[1]
    for idx in range(edge_index.size(1)):
        G.add_edge(
            src_local[idx].item(),
            dst_local[idx].item(),
            color=float(attn_values[idx].item()),
        )

    edge_colors = [data for (_, _, data) in G.edges(data="color")]
    labels = {n: G.nodes[n]["label"] for n in G.nodes()}

    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, cmap=plt.cm.viridis)
    edges = nx.draw_networkx_edges(
        G, pos, edge_color=edge_colors, edge_cmap=plt.cm.viridis
    )
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

    root_label = dataset.idx_to_protein.get(protein_ids[0], str(protein_ids[0]))
    plt.title(f"Protein-Protein Attention (Layer {layer_idx}): {root_label}")
    plt.colorbar(edges, label="Attention weight")
    filename = f"{path}/explanations/{root_label}_system_attention_layer{layer_idx}.png"
    os.makedirs(f"{path}/explanations", exist_ok=True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()


def plot_protein_attention(path, layer_attention, dataset, batch, layer_idx):
    key = ("aa", "belongs_to", "protein")
    if layer_attention is None or key not in layer_attention:
        return
    edge_index, attn_weights = layer_attention[key]
    edge_index = edge_index.detach().cpu()
    attn_values = _mean_attention(attn_weights.detach().cpu())

    src_local, dst_local = edge_index[0], edge_index[1]
    protein_ids = batch["protein"].n_id.detach().cpu().tolist()
    root_label = dataset.idx_to_protein.get(protein_ids[0], str(protein_ids[0]))

    explanations_dir = os.path.join(path, "explanations")
    os.makedirs(explanations_dir, exist_ok=True)

    for dst_val in torch.unique(dst_local, sorted=True).tolist():
        mask = dst_local == dst_val
        if torch.count_nonzero(mask) == 0:
            continue

        aa_indices = src_local[mask]
        attention_sorted = attn_values[mask][torch.argsort(aa_indices)]

        target_idx = int(dst_val)
        target_global = int(protein_ids[target_idx])
        target_label = dataset.idx_to_protein.get(target_global, str(target_global))

        x_positions = torch.arange(len(aa_indices), dtype=torch.float32)

        plt.figure(figsize=(8, 4))
        scatter = plt.scatter(
            x_positions.numpy(),
            attention_sorted.numpy(),
            c=attention_sorted.numpy(),
            cmap=plt.cm.viridis,
        )
        plt.colorbar(scatter, label="Attention weight")
        plt.xlabel("Residue")
        plt.ylabel("Attention")
        plt.title(f"AA-Protein Attention (Layer {layer_idx}): {target_label}")
        plt.tight_layout()

        filename = os.path.join(
            explanations_dir,
            f"{root_label}_{target_label}_aa_attention_layer{layer_idx}.png",
        )
        plt.savefig(filename)
        plt.show()
        plt.close()
