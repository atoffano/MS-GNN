"""Protein-protein (systemic/network-level) graph plots."""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import torch

from src.utils.constants import RANDOM_SEED
from src.utils.visualize.context import ProteinPlotContext, build_plot_context, save_plot

logger = logging.getLogger(__name__)


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
    save_plot(context, filename_suffix, go_term=go_term)


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
                f"network_captum_{rel}",
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
                f"network_attention_L{layer_idx}_{rel}",
            )
            plotted = True

    if not plotted:
        logger.info(f"No systemic attention found for layer {layer_idx}.")
