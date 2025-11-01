"""Visualization utilities for model interpretability and analysis."""

import logging
import os
import shutil
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from src.utils.constants import (
    RANDOM_SEED,
)
from src.utils.api import download_alphafold, download_pdb
from src.utils.helpers import timeit

try:
    import pymol2
except ImportError:
    pymol2 = None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ProteinPlotContext:
    """Context for protein visualization."""

    root_label: str
    root_global: int
    root_dir: str
    neighbor_dir: str
    protein_ids: list[int]
    labels: dict[int, str]


def build_plot_context(base_path: str, dataset, batch) -> ProteinPlotContext:
    """Build plotting context from batch."""
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


def resolve_plot_target(
    context: ProteinPlotContext, local_idx: int, go_term: Optional[str] = None
) -> tuple[str, str]:
    """Resolve output directory and filename prefix for a protein."""
    is_root = context.protein_ids[local_idx] == context.root_global
    target_label = context.labels[local_idx]

    base_dir = context.root_dir if is_root else context.neighbor_dir
    prefix = context.root_label if is_root else f"{context.root_label}_{target_label}"

    if go_term:
        go_subdir = go_term.replace(":", "_")
        output_dir = os.path.join(base_dir, "per-term", go_subdir)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir, prefix

    return base_dir, prefix


def ensure_structure(uniprot_id: str, out_dir: str) -> str:
    """Get structure for UniProt ID, checking caches before downloading.

    Priority: output cache -> alphafold_pdb -> esmfold_pdb -> RCSB PDB -> AlphaFold download
    """
    os.makedirs(out_dir, exist_ok=True)
    pdb_path = os.path.join(out_dir, f"{uniprot_id}.pdb")

    if os.path.exists(pdb_path):
        logger.info(f"Got PDB structure for {uniprot_id} from output directory cache")
        return pdb_path

    # Check local data directories
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_dir = os.path.join(
        current_file_dir, "..", "..", "data", "swissprot", "2024_01"
    )

    for cache_name in ["alphafold_pdb", "esmfold_pdb"]:
        cache_path = os.path.join(base_data_dir, cache_name, f"{uniprot_id}.pdb")
        if os.path.exists(cache_path):
            shutil.copy2(cache_path, pdb_path)
            logger.info(f"Got PDB structure for {uniprot_id} from {cache_name}")
            return pdb_path

    # Download from remote sources
    if download_pdb(uniprot_id, pdb_path):
        logger.info(f"Downloaded PDB structure for {uniprot_id} from RCSB")
        return pdb_path

    if download_alphafold(uniprot_id, pdb_path):
        logger.info(f"Downloaded PDB structure for {uniprot_id} from AlphaFold")
        return pdb_path

    raise FileNotFoundError(f"No structure available for {uniprot_id}")


# Structure rendering
def render_structure_colormap(
    pdb_path: str,
    residue_scores: Sequence[Tuple[int, float]],
    image_path: str,
    *,
    colormap: str = "viridis",
    title: str | None = None,
    normalize: bool = True,
) -> None:
    """Color a structure by residue scores via PyMOL."""
    if pymol2 is None:
        raise RuntimeError("pymol2 not installed")

    if not residue_scores:
        logger.warning(f"No residue scores for {pdb_path}, skipping")
        return

    res_idx, scores = zip(*residue_scores)
    res_idx = np.asarray(res_idx, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32)

    lo, hi = (float(scores.min()), float(scores.max())) if normalize else (None, None)
    if normalize and hi - lo < 1e-9:
        hi = lo + 1e-6

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


# Plotting utilities
def _save_plot(
    context: ProteinPlotContext, filename_suffix: str, go_term: Optional[str] = None
):
    """Save current plot to appropriate directory."""
    if go_term:
        go_subdir = go_term.replace(":", "_")
        output_dir = os.path.join(context.root_dir, "per-term", go_subdir)
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(
            output_dir, f"{context.root_label}_{filename_suffix}.png"
        )
    else:
        filename = os.path.join(
            context.root_dir, f"{context.root_label}_{filename_suffix}.png"
        )

    plt.savefig(filename)
    plt.close()


def _plot_protein_network(
    context: ProteinPlotContext,
    edge_index: torch.Tensor,
    weights: torch.Tensor,
    title: str,
    colorbar_label: str,
    filename_suffix: str,
    go_term: Optional[str] = None,
):
    """Create and save protein network graph."""
    G = nx.Graph()
    for local_idx, label in context.labels.items():
        G.add_node(local_idx, label=label)

    src, dst = edge_index[0], edge_index[1]
    for i in range(edge_index.size(1)):
        G.add_edge(int(src[i]), int(dst[i]), color=float(weights[i].item()))

    pos = nx.spring_layout(G, seed=RANDOM_SEED)
    nx.draw_networkx_nodes(G, pos, node_color="#d9d9d9")

    edges = None
    if G.number_of_edges():
        edge_colors = [data for (_, _, data) in G.edges(data="color")]
        edges = nx.draw_networkx_edges(
            G, pos, edge_color=edge_colors, edge_cmap=plt.cm.viridis
        )

    nx.draw_networkx_labels(G, pos, labels=context.labels, font_size=9)
    plt.title(title)

    if edges is not None:
        plt.colorbar(edges, label=colorbar_label)

    plt.tight_layout()
    _save_plot(context, filename_suffix, go_term)


def _mean_attention(attention_weights: torch.Tensor) -> torch.Tensor:
    """Average attention weights over all heads."""
    return (
        attention_weights.mean(dim=-1)
        if attention_weights.dim() > 1
        else attention_weights.view(-1)
    )


# Public plotting functions
@timeit
def plot_systemic_explanation(
    path, hetero_explanation, dataset, title_suffix=None, go_term=None
):
    """Plot protein-protein explanation graph."""
    context = build_plot_context(path, dataset, hetero_explanation.batch)
    key = ("protein", "aligned_with", "protein")

    edge_index = hetero_explanation[key]["edge_index"].detach().cpu()
    edge_mask = hetero_explanation[key]["edge_mask"].detach().cpu().view(-1)

    title = f"Protein-Protein Explanation: {context.root_label}"
    if title_suffix:
        title += f" ({title_suffix})"

    _plot_protein_network(
        context,
        edge_index,
        edge_mask,
        title,
        "Edge importance",
        "system_explanation",
        go_term,
    )


def plot_systemic_attention(path, layer_attention, dataset, batch, layer_idx):
    """Plot protein-protein attention graph."""
    context = build_plot_context(path, dataset, batch)
    key = ("protein", "aligned_with", "protein")

    if layer_attention is None or key not in layer_attention:
        return

    edge_index, attn_weights = layer_attention[key]
    edge_index = edge_index.detach().cpu()
    attn_values = _mean_attention(attn_weights).detach().cpu()

    title = f"Protein-Protein Attention (Layer {layer_idx}): {context.root_label}"
    _plot_protein_network(
        context,
        edge_index,
        attn_values,
        title,
        "Attention weight",
        f"system_attention_layer{layer_idx}",
    )


@timeit
def plot_protein_explanation(
    path: str,
    hetero_explanation,
    dataset,
    title_suffix: Optional[str] = None,
    go_term: Optional[str] = None,
):
    """Plot amino acid to protein explanation."""
    context = build_plot_context(path, dataset, hetero_explanation.batch)
    key = ("aa", "belongs_to", "protein")

    edge_mask = hetero_explanation[key]["edge_mask"].detach().cpu()
    edge_index = hetero_explanation[key]["edge_index"].detach().cpu()

    src_local, dst_local = edge_index[0], edge_index[1]
    protein_global_ids = (
        hetero_explanation.batch["protein"]["n_id"].detach().cpu().tolist()
    )

    for dst_val in torch.unique(dst_local, sorted=True).tolist():
        mask = dst_local == dst_val
        if not torch.any(mask):
            continue

        aa_indices = src_local[mask]
        edge_importance = edge_mask[mask].view(-1)

        target_idx = int(dst_val)
        target_global = int(protein_global_ids[target_idx])
        target_label = dataset.idx_to_protein.get(target_global, str(target_global))

        # Sort by AA index
        sort_idx = torch.argsort(aa_indices)
        edge_z_sorted = edge_importance[sort_idx]

        plt.figure(figsize=(8, 4))
        x_positions = torch.arange(len(edge_z_sorted), dtype=torch.float32)
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

        out_dir, prefix = resolve_plot_target(context, target_idx, go_term=go_term)
        filename = os.path.join(out_dir, f"{prefix}_aa_explanation.png")
        plt.savefig(filename)
        plt.close()


def plot_protein_attention(
    path, layer_attention, dataset, batch, layer_idx, go_term=None
):
    """Plot amino acid to protein attention weights."""
    context = build_plot_context(path, dataset, batch)
    key = ("aa", "belongs_to", "protein")

    if layer_attention is None or key not in layer_attention:
        return

    edge_index, attn_weights = layer_attention[key]
    edge_index = edge_index.detach().cpu()
    attn_values = _mean_attention(attn_weights.detach().cpu())

    src_local, dst_local = edge_index[0], edge_index[1]
    protein_ids = batch["protein"].n_id.detach().cpu().tolist()

    for dst_val in torch.unique(dst_local, sorted=True).tolist():
        mask = dst_local == dst_val
        if not torch.any(mask):
            continue

        aa_indices = src_local[mask]
        attention_sorted = attn_values[mask][torch.argsort(aa_indices)]

        target_idx = int(dst_val)
        target_global = int(protein_ids[target_idx])
        target_label = dataset.idx_to_protein.get(target_global, str(target_global))

        plt.figure(figsize=(8, 4))
        x_positions = torch.arange(len(aa_indices), dtype=torch.float32)
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

        out_dir, prefix = resolve_plot_target(context, target_idx, go_term=go_term)
        filename = os.path.join(out_dir, f"{prefix}_aa_attention_layer{layer_idx}.png")
        plt.savefig(filename)
        plt.close()


def analyze_attention_captum_correlation(
    output_dir: str,
    dataset,
    batch,
    attentions,
    hetero_explanation,
    *,
    layer_to_plot: int = 2,
) -> None:
    """Analyze correlation between attention and Captum scores."""
    context = build_plot_context(output_dir, dataset, batch)
    key = ("aa", "belongs_to", "protein")

    captum_edge_index = hetero_explanation[key]["edge_index"].detach().cpu()
    captum_scores = hetero_explanation[key]["edge_mask"].detach().cpu()

    # Analyze only root protein edges
    seed_mask = captum_edge_index[1] == 0
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
        edge_index = edge_index.detach().cpu()[:, seed_mask]
        attn_vals = _mean_attention(attn_weights.detach().cpu())[seed_mask]

        # Match edges
        shared_attn, shared_captum = [], []
        for i in range(edge_index.size(1)):
            edge = (int(edge_index[0, i]), int(edge_index[1, i]))
            if edge in edge_to_captum:
                shared_attn.append(float(attn_vals[i].item()))
                shared_captum.append(edge_to_captum[edge])

        if len(shared_attn) < 2:
            continue

        # Normalize
        attn_arr = np.asarray(shared_attn, dtype=np.float32)
        captum_arr = np.asarray(shared_captum, dtype=np.float32)
        attn_arr = (attn_arr - attn_arr.min()) / (
            attn_arr.max() - attn_arr.min() + 1e-9
        )
        captum_arr = (captum_arr - captum_arr.min()) / (
            captum_arr.max() - captum_arr.min() + 1e-9
        )

        # Compute correlation
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
            f"Pearson correlation (layer {layer_idx} vs Captum): {corr_val:.4f}"
        )

        if layer_idx == layer_to_plot:
            scatter_data = (attn_arr, captum_arr)

    if scatter_data is None:
        return

    # Plot scatter
    attn_arr, captum_arr = scatter_data
    plot_path = os.path.join(
        context.root_dir,
        f"{context.root_label}_attn_layer{layer_to_plot}_captum_scatter.png",
    )

    plt.figure(figsize=(6, 5))
    plt.scatter(attn_arr, captum_arr, alpha=0.6)
    m, b = np.polyfit(attn_arr, captum_arr, 1)
    plt.plot(attn_arr, m * attn_arr + b, color="red")
    plt.xlabel(f"Attention Layer {layer_to_plot}")
    plt.ylabel("Captum Score")
    plt.title(f"Attention vs Captum (Layer {layer_to_plot}) – {context.root_label}")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
