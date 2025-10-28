"""Visualization utilities for model interpretability and analysis.

This module provides comprehensive visualization tools for understanding model
predictions and internal representations, including:
- Attention weight visualization on protein structures
- Attribution score plotting
- Network graph visualizations
- PyMOL integration for 3D structure rendering
- Correlation analysis between attention and attribution methods
"""

import logging
import os
import shutil
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import requests
import torch

from src.utils.helpers import timeit

try:
    import pymol2
except ImportError:  # pragma: no cover
    pymol2 = None


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
    Get structure for the UniProt ID, checking local caches before downloading.
    Priority:
    1. Check out_dir cache
    2. Check data/swissprot/2024_01/alphafold_pdb
    3. Check data/swissprot/2024_01/esmfold_pdb
    4. Download from PDB (RCSB)
    5. Download from AlphaFold

    Returns the local PDB path in out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    pdb_path = os.path.join(out_dir, f"{uniprot_id}.pdb")

    # Check if already in output directory cache
    if os.path.exists(pdb_path):
        logger.info(f"Got PDB structure for {uniprot_id} from output directory cache")
        return pdb_path

    # Get base directory relative to this file (src/utils/visualize.py)
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_dir = os.path.join(
        current_file_dir, "..", "..", "data", "swissprot", "2024_01"
    )

    # Check local data directories
    local_cache_dirs = [
        os.path.join(base_data_dir, "alphafold_pdb"),
        os.path.join(base_data_dir, "esmfold_pdb"),
    ]

    for cache_dir in local_cache_dirs:
        cache_path = os.path.join(cache_dir, f"{uniprot_id}.pdb")
        if os.path.exists(cache_path):
            shutil.copy2(cache_path, pdb_path)
            logger.info(
                f"Got PDB structure for {uniprot_id} from local cache: {cache_dir}"
            )
            return pdb_path

    # Try downloading from PDB
    if _download_pdb(uniprot_id, pdb_path):
        logger.info(f"Downloaded PDB structure for {uniprot_id} from RCSB")
        return pdb_path

    # Fall back to AlphaFold download
    if _download_alphafold(uniprot_id, pdb_path):
        logger.info(f"Downloaded PDB structure for {uniprot_id} from AlphaFold")
        return pdb_path

    raise FileNotFoundError(
        f"No structure available for UniProt ID {uniprot_id} from local caches, PDB, or AlphaFold"
    )


@dataclass
class ProteinPlotContext:
    root_label: str
    root_global: int
    root_dir: str
    neighbor_dir: str
    protein_ids: list[int]
    labels: dict[int, str]


_PLOT_SEED = 42


def _draw_weighted_protein_graph(
    context: ProteinPlotContext,
    edge_index: torch.Tensor,
    weights: torch.Tensor,
    *,
    title: str,
    colorbar_label: str,
    filename_suffix: str,
) -> None:
    edge_index = edge_index.detach().cpu()
    weights = weights.detach().cpu().view(-1)

    G = nx.Graph()
    for local_idx, label in context.labels.items():
        G.add_node(local_idx, label=label)

    src, dst = edge_index
    for idx in range(edge_index.size(1)):
        G.add_edge(
            int(src[idx]),
            int(dst[idx]),
            color=float(weights[idx].item()),
        )

    pos = nx.spring_layout(G, seed=_PLOT_SEED)
    nx.draw_networkx_nodes(G, pos, node_color="#d9d9d9")

    edges = None
    if G.number_of_edges():
        edge_colors = [data for (_, _, data) in G.edges(data="color")]
        edges = nx.draw_networkx_edges(
            G,
            pos,
            edge_color=edge_colors,
            edge_cmap=plt.cm.viridis,
        )
    nx.draw_networkx_labels(G, pos, labels=context.labels, font_size=9)
    plt.title(title)
    if edges is not None:
        plt.colorbar(edges, label=colorbar_label)
    plt.tight_layout()

    filename = os.path.join(
        context.root_dir,
        f"{context.root_label}_{filename_suffix}.png",
    )
    plt.savefig(filename)
    plt.close()


def build_plot_context(base_path: str, dataset, batch) -> ProteinPlotContext:
    protein_ids = batch["protein"].n_id.detach().cpu().tolist()
    root_global = int(protein_ids[0])
    labels = {
        idx: dataset.idx_to_protein.get(global_id, str(global_id))
        for idx, global_id in enumerate(protein_ids)
    }
    root_label = labels[0]
    root_dir = os.path.join(base_path, "explanations", root_label)
    neighbor_dir = os.path.join(root_dir, "neighbors")
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(neighbor_dir, exist_ok=True)
    return ProteinPlotContext(
        root_label=root_label,
        root_global=root_global,
        root_dir=root_dir,
        neighbor_dir=neighbor_dir,
        protein_ids=protein_ids,
        labels=labels,
    )


def resolve_plot_target(context: ProteinPlotContext, local_idx: int) -> tuple[str, str]:
    is_root = context.protein_ids[local_idx] == context.root_global
    target_label = context.labels[local_idx]
    if is_root:
        return context.root_dir, context.root_label
    return context.neighbor_dir, f"{context.root_label}_{target_label}"


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
def plot_systemic_explanation(
    path: str,
    hetero_explanation,
    dataset,
    title_suffix: Optional[str] = None,
    go_term: Optional[str] = None,
):
    """Plot protein-protein explanation graph.

    Args:
        path: Output directory
        hetero_explanation: Explanation object
        dataset: Dataset object
        title_suffix: Optional suffix to add to plot title (e.g., "GO: term_name")
        go_term: Optional GO term ID for subdirectory organization
    """
    context = build_plot_context(path, dataset, hetero_explanation.batch)
    key = ("protein", "aligned_with", "protein")
    edge_mask = hetero_explanation[key]["edge_mask"].detach().cpu().view(-1)
    edge_index = hetero_explanation[key]["edge_index"].detach().cpu()

    title = f"Protein-Protein Explanation: {context.root_label}"
    if title_suffix:
        title += f" ({title_suffix})"

    # Determine output directory
    if go_term:
        go_subdir = go_term.replace(":", "_")
        output_dir = os.path.join(context.root_dir, go_subdir)
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(
            output_dir, f"{context.root_label}_system_explanation.png"
        )
    else:
        filename = os.path.join(
            context.root_dir, f"{context.root_label}_system_explanation.png"
        )

    # Draw graph
    edge_index_cpu = edge_index.detach().cpu()
    edge_mask_cpu = edge_mask.detach().cpu().view(-1)

    G = nx.Graph()
    for local_idx, label in context.labels.items():
        G.add_node(local_idx, label=label)

    src, dst = edge_index_cpu
    for idx in range(edge_index_cpu.size(1)):
        G.add_edge(
            int(src[idx]),
            int(dst[idx]),
            color=float(edge_mask_cpu[idx].item()),
        )

    pos = nx.spring_layout(G, seed=_PLOT_SEED)
    nx.draw_networkx_nodes(G, pos, node_color="#d9d9d9")

    edges = None
    if G.number_of_edges():
        edge_colors = [data for (_, _, data) in G.edges(data="color")]
        edges = nx.draw_networkx_edges(
            G,
            pos,
            edge_color=edge_colors,
            edge_cmap=plt.cm.viridis,
        )
    nx.draw_networkx_labels(G, pos, labels=context.labels, font_size=9)
    plt.title(title)
    if edges is not None:
        plt.colorbar(edges, label="Edge importance")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


@timeit
def plot_protein_explanation(
    path: str,
    hetero_explanation,
    dataset,
    title_suffix: Optional[str] = None,
    go_term: Optional[str] = None,
):
    """Plot amino acid to protein explanation.

    Args:
        path: Output directory
        hetero_explanation: Explanation object
        dataset: Dataset object
        title_suffix: Optional suffix to add to plot title (e.g., "GO: term_name")
        go_term: Optional GO term ID for subdirectory organization
    """
    context = build_plot_context(path, dataset, hetero_explanation.batch)
    edge_mask = (
        hetero_explanation[("aa", "belongs_to", "protein")]["edge_mask"].detach().cpu()
    )
    edge_index = (
        hetero_explanation[("aa", "belongs_to", "protein")]["edge_index"].detach().cpu()
    )

    src_local, dst_local = edge_index[0], edge_index[1]

    protein_batch = hetero_explanation.batch["protein"].detach().cpu()
    protein_global_ids = protein_batch["n_id"].tolist()

    for dst_val in torch.unique(dst_local, sorted=True).tolist():
        mask = dst_local == dst_val
        if torch.count_nonzero(mask) == 0:
            continue

        aa_indices = src_local[mask]
        edge_importance_z = edge_mask[mask].view(-1)

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
        plt.colorbar(scatter, label="Edge attribution")
        plt.xlabel("Residue")
        plt.ylabel("Edge Importance")

        title = f"AA-Protein Explanation: {target_label}"
        if title_suffix:
            title += f" ({title_suffix})"
        plt.title(title)

        plt.tight_layout()

        out_dir, prefix = resolve_plot_target(context, target_idx)

        # Determine output path based on GO term
        if go_term:
            go_subdir = go_term.replace(":", "_")
            out_dir = os.path.join(out_dir, go_subdir)
            os.makedirs(out_dir, exist_ok=True)

        filename = os.path.join(out_dir, f"{prefix}_aa_explanation.png")
        plt.savefig(filename)
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
    context = build_plot_context(output_dir, dataset, batch)
    key = ("aa", "belongs_to", "protein")

    captum_edge_index = hetero_explanation[key]["edge_index"].detach().cpu()
    captum_scores = hetero_explanation[key]["edge_mask"].detach().cpu()
    seed_mask = captum_edge_index[1] == 0  # analyze edges to root protein only
    captum_edge_index = captum_edge_index[:, seed_mask]
    captum_scores = captum_scores[seed_mask]

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
        edge_index = edge_index[:, seed_mask]  # edges to root protein only
        attn_vals = attn_vals[seed_mask]

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
        # min max normalization
        attn_arr = (attn_arr - attn_arr.min()) / (
            attn_arr.max() - attn_arr.min() + 1e-9
        )
        captum_arr = (captum_arr - captum_arr.min()) / (
            captum_arr.max() - captum_arr.min() + 1e-9
        )

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
    plot_path = os.path.join(
        context.root_dir,
        f"{context.root_label}_attn_layer{layer_to_plot}_captum_scatter.png",
    )
    plt.figure(figsize=(6, 5))
    plt.scatter(attn_arr, captum_arr, alpha=0.6)
    # Add regression line in red
    m, b = np.polyfit(attn_arr, captum_arr, 1)
    plt.plot(attn_arr, m * attn_arr + b, color="red")
    plt.xlabel(f"Attention Layer {layer_to_plot}")
    plt.ylabel("Captum Score")
    plt.title(f"Attention vs Captum (Layer {layer_to_plot}) – {context.root_label}")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def plot_systemic_attention(path, layer_attention, dataset, batch, layer_idx):
    context = build_plot_context(path, dataset, batch)
    key = ("protein", "aligned_with", "protein")
    if layer_attention is None or key not in layer_attention:
        return

    edge_index, attn_weights = layer_attention[key]
    edge_index = edge_index.detach().cpu()
    attn_values = _mean_attention(attn_weights).detach().cpu()

    _draw_weighted_protein_graph(
        context,
        edge_index,
        attn_values,
        title=f"Protein-Protein Attention (Layer {layer_idx}): {context.root_label}",
        colorbar_label="Attention weight",
        filename_suffix=f"system_attention_layer{layer_idx}",
    )


def plot_protein_attention(path, layer_attention, dataset, batch, layer_idx):
    context = build_plot_context(path, dataset, batch)
    key = ("aa", "belongs_to", "protein")
    if layer_attention is None or key not in layer_attention:
        return
    edge_index, attn_weights = layer_attention[key]
    edge_index = edge_index.detach().cpu()
    attn_values = _mean_attention(attn_weights.detach().cpu())

    src_local, dst_local = edge_index[0], edge_index[1]
    protein_ids = batch["protein"].n_id.detach().cpu().tolist()

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

        out_dir, prefix = resolve_plot_target(context, target_idx)
        filename = os.path.join(
            out_dir,
            f"{prefix}_aa_attention_layer{layer_idx}.png",
        )
        plt.savefig(filename)
        plt.close()
