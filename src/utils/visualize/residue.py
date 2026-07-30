"""AA-level (residue) scatter and bar plots."""

import logging
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import rankdata
from torch_scatter import scatter

from src.utils.visualize.context import ProteinPlotContext, build_plot_context, save_plot

logger = logging.getLogger(__name__)


def _rank_values(values: torch.Tensor) -> torch.Tensor:
    """Return 1-based ranks of *values* as a float tensor (ties → average rank)."""
    arr = values.detach().cpu().numpy().astype(np.float64)
    ranks = rankdata(arr, method="average")
    return torch.from_numpy(ranks).float()


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

        # Select residue nodes and edge values for the target protein, sorted by AA index
        aa_indices = src_local[mask]
        values = edge_values[mask].view(-1)
        sort_idx = torch.argsort(aa_indices)
        values_sorted = values[sort_idx]

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
        save_plot(context, filename_suffix, target_idx, go_term)


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
    # Index node_avg_attn at the unique source positions to avoid dimension mismatch
    y_values = node_avg_attn[unique_src]

    plt.figure(figsize=(10, 4))
    x_positions = torch.arange(len(unique_src), dtype=torch.float32)
    sc = plt.scatter(
        x_positions.numpy(),
        y_values.numpy(),
        c=y_values.numpy(),
        cmap=plt.cm.Spectral_r,
        s=15,
        alpha=0.8,
    )
    plt.colorbar(sc, label=ylabel)
    plt.xlabel("Residue (sorted by AA index)")
    plt.ylabel(ylabel)
    plt.title(title_template.format(protein=context.seed_label))
    plt.tight_layout()
    save_plot(context, filename_suffix, local_idx=0, go_term=go_term)


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

    title = "AA-Protein Explanation: {protein}"
    if title_suffix:
        title += f" ({title_suffix})"

    _plot_aa_to_protein_scatter(
        context,
        hetero_explanation[key]["edge_index"].detach().cpu(),
        hetero_explanation[key]["edge_mask"].detach().cpu(),
        title,
        "Edge Importance",
        "scatter_aa_captum",
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

    # Plot AA -> Protein attention
    belongs_key = ("aa", "belongs_to", "protein")
    if belongs_key in layer_attention:
        edge_index, attn_weights = layer_attention[belongs_key]
        _plot_aa_to_protein_scatter(
            context,
            edge_index.detach().cpu(),
            attn_weights.mean(dim=-1).detach().cpu(),
            f"AA-Protein Attention (Layer {layer_idx}): {{protein}}",
            "Attention weight",
            f"scatter_aa_attention_L{layer_idx}",
            go_term,
            plot_neighbors=plot_neighbors,
        )

    # Plot AA->AA attention for residues linked to seed protein, if available
    aa_close_key = ("aa", "close_to", "aa")
    if aa_close_key not in layer_attention or belongs_key not in layer_attention:
        return

    # Get AA indices linked to seed protein (index 0) through belongs_to edges
    edge_index, _ = layer_attention[belongs_key]
    aa_src, protein_dst = edge_index[0], edge_index[1]
    seed_mask = protein_dst == 0
    aa_idx = aa_src[seed_mask]

    edge_index, attn_weights = layer_attention[aa_close_key]
    # Get attention weights for edges where both source and target belong to seed protein
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
        f"scatter_aa_aa_attention_L{layer_idx}",
        go_term,
    )


def plot_protein_attention_rank(
    path: str,
    layer_attention,
    dataset,
    batch,
    layer_idx: int,
    go_term: Optional[str] = None,
    plot_neighbors: bool = True,
):
    """Plot amino acid to protein attention *ranks* (ranked within edge type)."""
    context = build_plot_context(path, dataset, batch)

    if layer_attention is None:
        return

    # Plot AA -> Protein attention ranks
    belongs_key = ("aa", "belongs_to", "protein")
    if belongs_key in layer_attention:
        edge_index, attn_weights = layer_attention[belongs_key]
        attn_mean = attn_weights.mean(dim=-1).detach().cpu()
        rank_values = _rank_values(attn_mean)
        _plot_aa_to_protein_scatter(
            context,
            edge_index.detach().cpu(),
            rank_values,
            f"AA-Protein Attention Rank (Layer {layer_idx}): {{protein}}",
            "Attention Rank",
            f"scatter_aa_attention_rank_L{layer_idx}",
            go_term,
            plot_neighbors=plot_neighbors,
        )

    # Plot AA->AA attention ranks for residues linked to seed protein
    aa_close_key = ("aa", "close_to", "aa")
    if aa_close_key not in layer_attention or belongs_key not in layer_attention:
        return

    edge_index, _ = layer_attention[belongs_key]
    aa_src, protein_dst = edge_index[0], edge_index[1]
    seed_mask = protein_dst == 0
    aa_idx = aa_src[seed_mask]

    edge_index, attn_weights = layer_attention[aa_close_key]
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

    # Rank the aggregated per-AA values within this edge type
    rank_values = _rank_values(node_avg_attn.flatten())

    _plot_aa_to_aa_scatter(
        context,
        aa_edge_index,
        rank_values,
        f"AA→AA Attention Rank (Layer {layer_idx}): {{protein}}",
        "Mean Attention Rank",
        f"scatter_aa_aa_attention_rank_L{layer_idx}",
        go_term,
    )


def plot_protein_explanation_rank(
    path: str,
    hetero_explanation,
    dataset,
    title_suffix: Optional[str] = None,
    go_term: Optional[str] = None,
    plot_neighbors: bool = True,
):
    """Plot amino acid to protein Captum explanation *ranks* (ranked within edge type)."""
    context = build_plot_context(path, dataset, hetero_explanation.batch)
    key = ("aa", "belongs_to", "protein")

    edge_mask = hetero_explanation[key]["edge_mask"].detach().cpu()
    rank_values = _rank_values(edge_mask)

    title = "AA-Protein Explanation Rank: {protein}"
    if title_suffix:
        title += f" ({title_suffix})"

    _plot_aa_to_protein_scatter(
        context,
        hetero_explanation[key]["edge_index"].detach().cpu(),
        rank_values,
        title,
        "Edge Importance Rank",
        "scatter_aa_captum_rank",
        go_term,
        plot_neighbors=plot_neighbors,
    )
