"""Visualization utilities for model interpretability and analysis."""

import logging
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import subprocess
import tempfile


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys

import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_scatter import scatter

from src.utils.constants import RANDOM_SEED
from src.utils.api import download_alphafold, download_alphafold_cif, download_pdb

try:
    import pymol2
except ImportError:
    pymol2 = None

logger = logging.getLogger(__name__)


def build_protein_score_map(
    edge_index: torch.Tensor, edge_values: torch.Tensor
) -> Dict[int, Dict[int, float]]:
    """Convert edge scores to per-protein residue score maps.

    Args:
        edge_index: Edge index tensor of shape [2, num_edges] where
                   edge_index[0] are source (AA) indices and
                   edge_index[1] are destination (protein) indices.
        edge_values: Score tensor of shape [num_edges] or [num_edges, 1].

    Returns:
        Dict mapping protein_local_idx to {aa_local_idx: score}.
    """
    src_local, dst_local = edge_index[0], edge_index[1]
    flat_values = edge_values.view(-1)

    protein_data = {}
    for dst_val in torch.unique(dst_local, sorted=True).tolist():
        mask = dst_local == dst_val
        if not torch.any(mask):
            continue
        aa_indices = src_local[mask].tolist()
        values = flat_values[mask].tolist()
        protein_data[dst_val] = dict(zip(aa_indices, values))

    return protein_data





@dataclass
class ProteinPlotContext:
    """Context for protein visualization."""

    seed_label: str
    seed_global: int
    seed_dir: str
    neighbor_dir: str
    protein_ids: List[int]
    labels: Dict[int, str]


def build_plot_context(base_path: str, dataset, batch) -> ProteinPlotContext:
    """Build plotting context from batch."""
    protein_ids = batch["protein"].n_id.detach().cpu().tolist()
    seed_global = int(protein_ids[0])
    labels = {
        idx: dataset.idx_to_protein.get(global_id, str(global_id))
        for idx, global_id in enumerate(protein_ids)
    }
    seed_label = labels[0]
    seed_dir = os.path.join(base_path, "explanations", seed_label)
    neighbor_dir = os.path.join(seed_dir, "neighbors")
    os.makedirs(seed_dir, exist_ok=True)
    os.makedirs(neighbor_dir, exist_ok=True)

    return ProteinPlotContext(
        seed_label=seed_label,
        seed_global=seed_global,
        seed_dir=seed_dir,
        neighbor_dir=neighbor_dir,
        protein_ids=protein_ids,
        labels=labels,
    )


def ensure_structure(uniprot_id: str, out_dir: str, global_dir: Optional[str] = None) -> str:
    """Get structure for UniProt ID, checking caches before downloading.

    Priority: global_dir .cif -> global_dir .pdb -> out_dir .cif -> out_dir .pdb
              -> AlphaFold CIF download (to global_dir or out_dir) -> AlphaFold PDB download
    Returns the path to the structure file (.cif preferred over .pdb).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Check global model directory first
    search_dirs = [d for d in [global_dir, out_dir] if d]
    for search_dir in search_dirs:
        cif_path = os.path.join(search_dir, f"{uniprot_id}.cif")
        if os.path.exists(cif_path):
            logger.info(f"Found CIF structure for {uniprot_id} in {search_dir}")
            return cif_path
        pdb_path = os.path.join(search_dir, f"{uniprot_id}.pdb")
        if os.path.exists(pdb_path):
            logger.info(f"Found PDB structure for {uniprot_id} in {search_dir}")
            return pdb_path

    # Download to global_dir if provided, otherwise out_dir
    download_dir = global_dir if global_dir else out_dir
    os.makedirs(download_dir, exist_ok=True)
    cif_path = os.path.join(download_dir, f"{uniprot_id}.cif")
    pdb_path = os.path.join(download_dir, f"{uniprot_id}.pdb")

    # Check local data directories

    # Download from remote sources — try CIF first, then fall back to PDB
    if download_alphafold_cif(uniprot_id, cif_path):
        logger.info(f"Downloaded CIF structure for {uniprot_id} from AlphaFold")
        return cif_path


    if download_alphafold(uniprot_id, pdb_path):
        logger.info(f"Downloaded PDB structure for {uniprot_id} from AlphaFold")
        return pdb_path

    raise FileNotFoundError(f"No structure available for {uniprot_id}")


def adjust_colormap(cmap, luminance_factor=1.0, saturation_factor=1.0, n_colors=256):
    """
    Adjust the luminance and saturation of a colormap.

    Parameters:
    -----------
    cmap : matplotlib.colors.Colormap
        The input colormap to adjust.
    luminance_factor : float
        Factor to scale the luminance (value) of the colors.
    saturation_factor : float
        Factor to scale the saturation of the colors.
    n_colors : int
        Number of colors in the adjusted colormap.

    Returns:
    --------
    new_cmap : matplotlib.colors.LinearSegmentedColormap
        The adjusted colormap.
    """
    # Sample the colormap
    colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]

    # Convert RGBA to RGB and adjust luminance and saturation
    adjusted_colors = []
    for color in colors:
        r, g, b = color[:3]  # Ignore alpha channel
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        s = min(1.0, s * saturation_factor)  # Scale saturation
        v = min(1.0, v * luminance_factor)  # Scale luminance
        adjusted_colors.append(colorsys.hsv_to_rgb(h, s, v))

    # Create a new colormap
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        "adjusted_cmap", adjusted_colors, N=n_colors
    )
    return new_cmap


def apply_spectrum(cmd, cmap, scores, ca_resi, n_colors=256):
    """
    Generates a custom cmap and applies it to a PyMOL scene.
    """
    cmap = adjust_colormap(cmap, luminance_factor=1, saturation_factor=4)
    sampled_colors = [cmap(i / (n_colors - 1))[:3] for i in range(n_colors)]
    rgb_colors = [tuple(color) for color in sampled_colors]
    names = [mcolors.to_hex(color) for color in sampled_colors]

    for name, color in zip(names, rgb_colors):
        cmd.set_color(name, list(color))
    # Apply spectrum to CA atoms based on scores
    cmd.spectrum(
        expression="b",
        palette=" ".join(names),
        selection=f"resi {'+'.join(str(r) for r in ca_resi)}",
        minimum=float(np.min(scores)),
        maximum=float(np.max(scores)),
    )
    # Color everything else gray, including resi lacking scores
    # cmd.spectrum(
    #     expression="b",
    #     palette="gray70 gray70",
    #     selection=f"not resi {'+'.join(str(r) for r in ca_resi)}",
    #     minimum=0,
    #     maximum=0,
    # )


def render_scene(
    structure_path: str,
    residue_scores: Sequence[Tuple[int, float]],
    image_path: str,
    *,
    title: str | None = None,
) -> None:
    """Color a structure by residue scores via PyMOL. Supports both .pdb and .cif files."""

    if not residue_scores:
        logger.warning(f"No residue scores for {structure_path}, skipping")
        return

    # Build lookup from PDB residue number to score
    score_map = dict(residue_scores)
    score_values = np.asarray([s for _, s in residue_scores], dtype=np.float32)

    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    logger.info(f"Rendering structure {structure_path}")

    with pymol2.PyMOL() as pymol:
        cmd = pymol.cmd
        cmd.reinitialize()
        cmd.load(structure_path, "prot")
        cmd.alter("prot", "b=0.0")
        ca_resi = [int(atom.resi) for atom in cmd.get_model("prot and name CA").atom]
        scored_resi = []
        for resi_nb in ca_resi:
            if resi_nb in score_map:
                cmd.alter(f"prot and resi {resi_nb}", f"b={score_map[resi_nb]}")
                scored_resi.append(resi_nb)
        if not scored_resi:
            logger.warning(
                f"No residue scores matched CA atoms in {structure_path}, skipping rendering."
            )
            return
        if len(scored_resi) != len(score_map):
            logger.warning(
                f"Matched {len(scored_resi)}/{len(score_map)} scored residues to CA atoms in {structure_path}."
            )
        apply_spectrum(cmd, plt.cm.Spectral_r, score_values, scored_resi)
        # Color residues without scores gray
        unscored_sel = f"not resi {'+'.join(str(r) for r in scored_resi)}"
        cmd.color("gray70", selection=unscored_sel)
        cmd.color("gray70", selection=f"HETATM")

        if title:
            cmd.set_title("title", state=0, text=title)
        cmd.set("spec_reflect", 0)
        cmd.set("ray_shadows", 0)
        cmd.set("ray_opaque_background", 0)
        cmd.bg_color("black")
        cmd.orient("prot")
        cmd.png(image_path, width=1600, height=1200, dpi=300, ray=1)
        cmd.save(image_path.replace(".png", ".pse"))
        logger.info(f"Saved structure rendering to {image_path}")


def _save_plot(
    context: ProteinPlotContext,
    filename_suffix: str,
    local_idx: int = 0,
    go_term: Optional[str] = None,
):
    """Save current plot to appropriate directory."""
    is_seed = context.protein_ids[local_idx] == context.seed_global
    base_dir = context.seed_dir if is_seed else context.neighbor_dir
    prefix = (
        context.seed_label
        if is_seed
        else f"{context.seed_label}_{context.labels[local_idx]}"
    )

    if go_term:
        go_subdir = go_term.replace(":", "_")
        output_dir = os.path.join(base_dir, "per-term", go_subdir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = base_dir

    filename = os.path.join(output_dir, f"{prefix}_{filename_suffix}.png")
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

    # Remove isolated nodes & associated labels
    G.remove_nodes_from(list(nx.isolates(G)))
    filtered_labels = {
        idx: label for idx, label in context.labels.items() if idx in set(G.nodes())
    }

    pos = nx.spring_layout(G, seed=RANDOM_SEED)
    nx.draw_networkx_nodes(G, pos, node_color="#d9d9d9")

    edges = None
    if G.number_of_edges():
        edge_colors = [data for (_, _, data) in G.edges(data="color")]
        edges = nx.draw_networkx_edges(
            G, pos, edge_color=edge_colors, edge_cmap=plt.cm.Spectral_r
        )
    nx.draw_networkx_labels(G, pos, labels=filtered_labels, font_size=9)
    plt.title(title)

    if edges is not None:
        plt.colorbar(edges, label=colorbar_label)

    plt.tight_layout()
    _save_plot(context, filename_suffix, go_term=go_term)


def _plot_aa_to_protein_scatter(
    context: ProteinPlotContext,
    edge_index: torch.Tensor,
    edge_values: torch.Tensor,
    title_template: str,
    ylabel: str,
    filename_suffix: str,
    go_term: Optional[str] = None,
    plot_neighbors: bool = True,
):
    """Per-protein scatter plot for AA→protein edges (attention or attribution)."""
    src_local, dst_local = edge_index[0], edge_index[1]

    for dst_val in torch.unique(dst_local, sorted=True).tolist():
        target_idx = int(dst_val)
        is_seed = context.protein_ids[target_idx] == context.seed_global

        if not plot_neighbors and not is_seed:
            continue

        mask = dst_local == dst_val
        if not torch.any(mask):
            continue

        # Select the right residue nodes and edge values for the currently targeted protein
        aa_indices = src_local[mask]
        values = edge_values[mask].view(-1)
        sort_idx = torch.argsort(aa_indices)
        values_sorted = values[sort_idx]

        target_idx = int(dst_val)
        target_label = context.labels[target_idx]

        plt.figure(figsize=(8, 4))
        x_positions = torch.arange(len(values_sorted), dtype=torch.float32)
        scatter = plt.scatter(
            x_positions.numpy(),
            values_sorted.numpy(),
            c=values_sorted.numpy(),
            cmap=plt.cm.Spectral_r,
        )
        plt.colorbar(scatter, label=ylabel)
        plt.xlabel("Residue")
        plt.ylabel(ylabel)
        plt.title(title_template.format(protein=target_label))
        plt.tight_layout()
        _save_plot(context, filename_suffix, target_idx, go_term)




#     return aligned_seqs


# def plot_protein_explanation_msa(
#     path: str,
#     hetero_explanation,
#     dataset,
#     batch,
#     title_suffix: Optional[str] = None,
#     go_term: Optional[str] = None,
#     aligned_seqs: Optional[List[str]] = None,
# ):
#     """Plot amino acid to protein explanation aligned by MSA."""
#     if aligned_seqs is None:
#         aligned_seqs = perform_msa_from_batch(batch)
#     if aligned_seqs is None:
#         return

#     context = build_plot_context(path, dataset, hetero_explanation.batch)
#     key = ("aa", "belongs_to", "protein")

#     title = f"AA-Protein Explanation (MSA-aligned): {context.seed_label}"
#     if title_suffix:
#         title += f" ({title_suffix})"

#     _plot_aa_to_protein_msa(
#         context,
#         hetero_explanation[key]["edge_index"].detach().cpu(),
#         hetero_explanation[key]["edge_mask"].detach().cpu(),
#         aligned_seqs,
#         title,
#         "Normalized Edge Importance",
#         "aa_explanation_msa",
#         go_term,
#     )


# def plot_protein_attention_msa(
#     path: str,
#     layer_attention,
#     dataset,
#     batch,
#     layer_idx: int,
#     go_term: Optional[str] = None,
#     aligned_seqs: Optional[List[str]] = None,
# ):
#     """Plot amino acid to protein attention weights aligned by MSA."""
#     if aligned_seqs is None:
#         aligned_seqs = perform_msa_from_batch(batch)
#     if aligned_seqs is None:
#         return

#     context = build_plot_context(path, dataset, batch)
#     key = ("aa", "belongs_to", "protein")

#     if layer_attention is None or key not in layer_attention:
#         return

#     edge_index, attn_weights = layer_attention[key]
#     mean_attn = attn_weights.mean(dim=-1).detach().cpu()
#     edge_index_cpu = edge_index.detach().cpu()

#     _plot_aa_to_protein_msa(
#         context,
#         edge_index_cpu,
#         mean_attn,
#         aligned_seqs,
#         f"AA-Protein Attention (Layer {layer_idx}, MSA-aligned): {context.seed_label}",
#         "Normalized Attention Weight",
#         f"aa_attention_layer{layer_idx}_msa",
#         go_term,
#         center_on_uniform=True,
#     )

#     _plot_msa_alignment_violin(
#         context,
#         edge_index_cpu,
#         mean_attn,
#         aligned_seqs,
#         f"Aligned vs Unaligned Attention (Layer {layer_idx}): {context.seed_label}",
#         f"aa_attention_layer{layer_idx}_msa_violin",
#         go_term,
#     )


# def plot_attn_seed_vs_neighbor_scatter(
#     path: str,
#     layer_attention,
#     dataset,
#     batch,
#     layer_idx: int,
#     go_term: Optional[str] = None,
#     aligned_seqs: Optional[List[str]] = None,
# ):
#     """Plot scatter of seed vs neighbor attention for aligned residues."""
#     if aligned_seqs is None:
#         aligned_seqs = perform_msa_from_batch(batch)
#     if aligned_seqs is None:
#         return

#     context = build_plot_context(path, dataset, batch)
#     key = ("aa", "belongs_to", "protein")

#     if layer_attention is None or key not in layer_attention:
#         return

#     edge_index, attn_weights = layer_attention[key]

#     _plot_attn_seed_vs_neighbor_scatter(
#         context,
#         edge_index.detach().cpu(),
#         attn_weights.mean(dim=-1).detach().cpu(),
#         aligned_seqs,
#         f"Seed vs Neighbor Normalized Attention (Layer {layer_idx}): {context.seed_label}",
#         f"attn_seed_vs_neighbor_layer{layer_idx}",
#         go_term,
#     )


def plot_attn_stringdb_vs_aligned_scatter(
    path: str,
    layer_attention,
    dataset,
    batch,
    layer_idx: int,
    go_term: Optional[str] = None,
):
    """Plot scatter of StringDB vs Aligned attention for common neighbors."""
    context = build_plot_context(path, dataset, batch)

    # Identify keys
    stringdb_key = None
    aligned_key = None

    if layer_attention is None:
        return

    for key in layer_attention.keys():
        if not isinstance(key, tuple) or len(key) != 3:
            continue
        src, rel, dst = key
        if src == "protein" and dst == "protein":
            if "stringdb" in rel:
                stringdb_key = key
            elif "aligned" in rel:
                aligned_key = key

    if not stringdb_key or not aligned_key:
        return

    # Helper to extract weights for edges pointing to seed (index 0)
    def get_weights(key):
        edge_index, weights = layer_attention[key]
        weights = weights.mean(dim=-1).detach().cpu()
        edge_index = edge_index.detach().cpu()

        # Assuming seed is at index 0
        mask = edge_index[1] == 0
        neighbors = edge_index[0][mask]
        w = weights[mask]
        return {int(n): float(v) for n, v in zip(neighbors, w)}

    w_stringdb = get_weights(stringdb_key)
    w_aligned = get_weights(aligned_key)

    common = sorted(list(set(w_stringdb.keys()) & set(w_aligned.keys())))
    if len(common) < 2:
        return

    x_vals = [w_stringdb[n] for n in common]
    y_vals = [w_aligned[n] for n in common]

    # Correlation
    if np.std(x_vals) > 1e-9 and np.std(y_vals) > 1e-9:
        corr = np.corrcoef(x_vals, y_vals)[0, 1]
    else:
        corr = 0.0

    plt.figure(figsize=(6, 6))
    plt.scatter(x_vals, y_vals, alpha=0.7, c="blue", edgecolors="k")

    # Add trendline
    if len(x_vals) > 1:
        try:
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(x_vals), max(x_vals), 100)
            plt.plot(x_range, p(x_range), "r--", alpha=0.5)
        except Exception:
            pass

    plt.xlabel(f"Attention ({stringdb_key[1]})")
    plt.ylabel(f"Attention ({aligned_key[1]})")
    plt.title(f"Layer {layer_idx} Attention Correlation\n{context.seed_label}")
    plt.legend([f"Pearson R = {corr:.3f}"], loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    _save_plot(context, f"attn_stringdb_vs_aligned_layer{layer_idx}", go_term=go_term)


def plot_systemic_explanation(
    path, hetero_explanation, dataset, title_suffix=None, go_term=None
):
    """Plot protein-protein explanation graph."""
    context = build_plot_context(path, dataset, hetero_explanation.batch)

    plotted = False
    for edge_type in hetero_explanation.edge_types:
        src, rel, dst = edge_type
        if src == "protein" and dst == "protein":
            if edge_type not in hetero_explanation:
                continue

            expl_data = hetero_explanation[edge_type]
            if (
                not hasattr(expl_data, "edge_index")
                or expl_data.edge_index.numel() == 0
            ):
                continue

            edge_index = expl_data.edge_index.detach().cpu()
            edge_mask = expl_data.edge_mask.detach().cpu().view(-1)

            title = f"Protein-Protein Explanation ({rel}): {context.seed_label}"
            if title_suffix:
                title += f" ({title_suffix})"

            _plot_protein_network(
                context,
                edge_index,
                edge_mask,
                title,
                "Edge importance",
                f"system_explanation_{rel}",
                go_term,
            )
            plotted = True

    if not plotted:
        logger.info("No systemic edges found to plot.")


def plot_systemic_attention(path, layer_attention, dataset, batch, layer_idx):
    """Plot protein-protein attention graph."""
    context = build_plot_context(path, dataset, batch)

    if layer_attention is None:
        return

    plotted = False
    for edge_type, (edge_index, attn_weights) in layer_attention.items():
        if not isinstance(edge_type, tuple) or len(edge_type) != 3:
            continue

        src, rel, dst = edge_type
        if src == "protein" and dst == "protein":
            edge_index = edge_index.detach().cpu()
            if edge_index.numel() == 0:
                continue

            attn_values = attn_weights.mean(dim=-1).detach().cpu()

            # Filter out self-loops for non-seed proteins (seed is at index 0 when batch_size = 1)
            src_idx, dst_idx = edge_index[0], edge_index[1]
            mask = ~((src_idx == dst_idx) & (src_idx != 0))
            edge_index = edge_index[:, mask]
            attn_values = attn_values[mask]

            if edge_index.numel() == 0:
                continue

            title = f"Protein-Protein Attention (Layer {layer_idx}, {rel}): {context.seed_label}"
            _plot_protein_network(
                context,
                edge_index,
                attn_values,
                title,
                "Attention Weight",
                f"system_attention_layer{layer_idx}_{rel}",
            )
            plotted = True

    if not plotted:
        logger.info(f"No systemic attention found for layer {layer_idx}.")


def plot_protein_explanation(
    path: str,
    hetero_explanation,
    dataset,
    title_suffix: Optional[str] = None,
    go_term: Optional[str] = None,
    plot_neighbors: bool = True,
):
    """Plot amino acid to protein explanation."""
    context = build_plot_context(path, dataset, hetero_explanation.batch)
    key = ("aa", "belongs_to", "protein")

    title = f"AA-Protein Explanation: {{protein}}"
    if title_suffix:
        title += f" ({title_suffix})"

    _plot_aa_to_protein_scatter(
        context,
        hetero_explanation[key]["edge_index"].detach().cpu(),
        hetero_explanation[key]["edge_mask"].detach().cpu(),
        title,
        "Edge Importance",
        "aa_explanation",
        go_term,
        plot_neighbors=plot_neighbors,
    )


def plot_protein_attention(
    path: str,
    layer_attention,
    dataset,
    batch,
    layer_idx: int,
    go_term: Optional[str] = None,
    plot_neighbors: bool = True,
):
    """Plot amino acid to protein attention weights."""
    context = build_plot_context(path, dataset, batch)

    if layer_attention is None:
        return

    # Plot AA -> Protein attention.
    for key in [
        ("aa", "belongs_to", "protein"),
    ]:
        if key not in layer_attention:
            continue

        edge_index, attn_weights = layer_attention[key]
        _plot_aa_to_protein_scatter(
            context,
            edge_index.detach().cpu(),
            attn_weights.mean(dim=-1).detach().cpu(),
            f"AA-Protein Attention ({key[1]}, Layer {layer_idx}): {{protein}}",
            "Attention weight",
            f"aa_attention_{key[1]}_layer{layer_idx}",
            go_term,
            plot_neighbors=plot_neighbors,
        )

    # Plot AA->AA attention for residues linked to seed protein, if available
    aa_close_key = ("aa", "close_to", "aa")
    if aa_close_key not in layer_attention:
        return
    # Get AA indices linked to seed protein (index 0) through belongs_to edges
    edge_index, _ = layer_attention[("aa", "belongs_to", "protein")]
    aa_src, protein_dst = edge_index[0], edge_index[1]
    seed_mask = protein_dst == 0
    aa_idx = aa_src[seed_mask]
    edge_index, attn_weights = layer_attention[aa_close_key]
    # Get attention weights for edge_index where both source and target are in aa_idx
    aa_src, aa_dst = edge_index[0], edge_index[1]
    aa_mask = torch.isin(aa_src, aa_idx) & torch.isin(aa_dst, aa_idx)
    aa_edge_index = edge_index[:, aa_mask].detach().cpu()
    aa_attn_weights = attn_weights[aa_mask].detach().cpu()

    if aa_attn_weights.dim() > 1:
        aa_attn_weights = aa_attn_weights.mean(dim=-1)

    # Aggregate AA->AA attention per source AA node with degree smoothing
    node_sum_attn = scatter(aa_attn_weights, aa_edge_index[0], dim=0, reduce="sum")
    node_degree = scatter(
        torch.ones_like(aa_attn_weights), aa_edge_index[0], dim=0, reduce="sum"
    )
    smoothing_factor = node_degree.mean().item() if node_degree.numel() > 0 else 1.0
    node_avg_attn = node_sum_attn / (node_degree + smoothing_factor)

    _plot_aa_to_aa_scatter(
        context,
        aa_edge_index,
        node_avg_attn.flatten(),
        f"AA→AA Attention (Layer {layer_idx}): {{protein}}",
        "Mean Attention Weight",
        f"aa_aa_attention_layer{layer_idx}",
        go_term,
    )


def _plot_aa_to_aa_scatter(
    context: ProteinPlotContext,
    aa_edge_index: torch.Tensor,
    node_avg_attn: torch.Tensor,
    title_template: str,
    ylabel: str,
    filename_suffix: str,
    go_term: Optional[str] = None,
):
    """Scatter plot of per-residue average AA→AA attention for the seed protein."""
    if aa_edge_index.numel() == 0:
        logger.warning("No AA→AA edges found, skipping scatter plot.")
        return

    unique_src = aa_edge_index[0].unique().cpu()

    plt.figure(figsize=(10, 4))
    sc = plt.scatter(
        unique_src.numpy(),
        node_avg_attn.numpy(),
        c=node_avg_attn.numpy(),
        cmap=plt.cm.Spectral_r,
        s=15,
        alpha=0.8,
    )
    plt.colorbar(sc, label=ylabel)
    plt.xlabel("Residue (sorted by AA index)")
    plt.ylabel(ylabel)
    plt.title(title_template.format(protein=context.seed_label))
    plt.tight_layout()
    _save_plot(context, filename_suffix, local_idx=0, go_term=go_term)


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

    # Analyze only seed protein edges
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
        edge_index = edge_index.detach().cpu()
        attn_vals = attn_weights.mean(dim=-1).detach().cpu()

        # Compute seed mask from this layer's own edge index
        layer_seed_mask = edge_index[1] == 0
        edge_index = edge_index[:, layer_seed_mask]
        attn_vals = attn_vals[layer_seed_mask]

        # Match edges
        shared_attn, shared_captum = [], []
        for i in range(edge_index.size(1)):
            edge = (int(edge_index[0, i]), int(edge_index[1, i]))
            if edge in edge_to_captum:
                shared_attn.append(float(attn_vals[i].item()))
                shared_captum.append(edge_to_captum[edge])

        if len(shared_attn) < 2:
            continue

        attn_arr = np.asarray(shared_attn, dtype=np.float32)
        captum_arr = np.asarray(shared_captum, dtype=np.float32)

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

        if layer_idx == layer_to_plot:
            scatter_data = (attn_arr, captum_arr, corr_val)

    if scatter_data is None:
        return

    # Plot scatter
    attn_arr, captum_arr, corr_val = scatter_data
    plot_path = os.path.join(
        context.seed_dir,
        f"{context.seed_label}_attn_layer{layer_to_plot}_captum_scatter.png",
    )

    plt.figure(figsize=(6, 5))
    plt.scatter(attn_arr, captum_arr, alpha=0.6, label=f"Pearson R = {corr_val:.3f}")
    m, b = np.polyfit(attn_arr, captum_arr, 1)
    plt.plot(attn_arr, m * attn_arr + b, color="red")
    plt.xlabel(f"Attention Layer {layer_to_plot}")
    plt.ylabel("Captum Score")
    plt.title(f"Attention vs Captum (Layer {layer_to_plot}) – {context.seed_label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
