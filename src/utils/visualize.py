from typing import Any, Dict, List, Optional
import torch


def visualize_graph_via_networkx(
    hetero_explanation,
    path: Optional[str] = None,
    cutoff_edge: float = 0.0,
    cutoff_node: float = 0.0,
    node_labels: Optional[Dict[str, List[str]]] = None,
) -> Any:
    import matplotlib.pyplot as plt
    import networkx as nx

    plots = {}
    for edge_type in hetero_explanation.edge_types:
        src_type, _, dst_type = edge_type
        edge_index = hetero_explanation[edge_type].edge_index
        edge_mask = hetero_explanation[edge_type].edge_mask

        # Filter edges based on cutoff
        mask = edge_mask >= cutoff_edge
        edge_index = edge_index[:, mask]
        edge_mask = edge_mask[mask]

        # Get node masks and compute importance (sum over features)
        src_node_mask = hetero_explanation[src_type].node_mask
        dst_node_mask = hetero_explanation[dst_type].node_mask
        src_node_importance = src_node_mask.sum(dim=1)
        dst_node_importance = dst_node_mask.sum(dim=1)

        # Filter nodes based on cutoff
        src_nodes = edge_index[0].unique()
        dst_nodes = edge_index[1].unique()
        src_keep = src_nodes[src_node_importance[src_nodes] >= cutoff_node]
        dst_keep = dst_nodes[dst_node_importance[dst_nodes] >= cutoff_node]

        # Further filter edges to only include kept nodes
        src_mask = torch.isin(edge_index[0], src_keep)
        dst_mask = torch.isin(edge_index[1], dst_keep)
        keep_mask = src_mask & dst_mask
        edge_index = edge_index[:, keep_mask]
        edge_mask = edge_mask[keep_mask]

        # Get all nodes after filtering
        all_nodes = torch.cat([edge_index[0], edge_index[1]]).unique()

        g = nx.DiGraph()
        node_size = 800

        # Create node to label mapping and add nodes
        node_to_label = {}
        for node in all_nodes.tolist():
            if node in edge_index[0]:
                nt = src_type
            else:
                nt = dst_type
            labels = node_labels.get(nt, None) if node_labels else None
            label = str(node) if labels is None else labels[node]
            node_to_label[node] = label
            importance = hetero_explanation[nt].node_mask[node].sum().item()
            g.add_node(label, importance=importance)

        # Add edges
        for i, (src, dst) in enumerate(edge_index.t().tolist()):
            w = edge_mask[i].item()
            g.add_edge(node_to_label[src], node_to_label[dst], weight=w)

        # Plot
        fig, ax = plt.subplots()
        pos = nx.spring_layout(g)

        # Node colors based on importance
        node_colors = [g.nodes[n]["importance"] for n in g.nodes]
        if node_colors:
            min_imp = min(node_colors)
            max_imp = max(node_colors)
            if max_imp > min_imp:
                node_colors = [(c - min_imp) / (max_imp - min_imp) for c in node_colors]
            else:
                node_colors = [0.5] * len(node_colors)

        # Edge colors and widths based on weight
        edge_colors = [g.edges[e]["weight"] for e in g.edges]
        edge_widths = [w * 10 for w in edge_colors]  # Scale width
        if edge_colors:
            min_w = min(edge_colors)
            max_w = max(edge_colors)
            if max_w > min_w:
                edge_colors = [(c - min_w) / (max_w - min_w) for c in edge_colors]
            else:
                edge_colors = [0.5] * len(edge_colors)

        # Draw
        nx.draw_networkx_nodes(
            g,
            pos,
            node_size=node_size,
            node_color=node_colors,
            cmap=plt.cm.viridis,
            ax=ax,
        )
        nx.draw_networkx_edges(
            g, pos, edge_color=edge_colors, width=edge_widths, arrowstyle="->", ax=ax
        )
        nx.draw_networkx_labels(g, pos, font_size=10, ax=ax)

        if path is not None:
            suffix = "_".join(edge_type)
            plt.savefig(f"{path}_{suffix}.png")
        else:
            plt.show()
        plt.close()
        plots[edge_type] = fig

    return plots


def plot_aa_edge_histogram(
    hetero_explanation,
    edge_type: tuple = ("aa", "aa2protein", "protein"),
    path: Optional[str] = None,
    aa_labels: Optional[List[str]] = None,
) -> Any:
    """
    Scatter plot for all 'aa' nodes where:
      - x axis: aa node (label if provided, otherwise index)
      - y axis: summed node-feature importance for that aa node

    Returns dict with figure, values (per-aa summed importance) and indices.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Resolve node mask for 'aa' and compute summed importance per node
    aa_node_mask = hetero_explanation["aa"].node_mask  # shape [num_aa, num_features]
    aa_importance = aa_node_mask.sum(dim=1).detach().cpu().numpy()  # shape [num_aa]

    num_aa = aa_importance.shape[0]
    indices = np.arange(num_aa)

    # Labels for x axis
    if aa_labels is not None:
        labels = [str(l) for l in aa_labels]
        # If label list shorter/longer than nodes, fallback to indices for missing ones
        if len(labels) < num_aa:
            labels = labels + [str(i) for i in range(len(labels), num_aa)]
    else:
        labels = [str(i) for i in range(num_aa)]

    # Create scatter plot
    figsize = (max(6, int(num_aa * 0.15)), 4)  # scale width with number of nodes
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(indices, aa_importance, color="C0", s=20)
    ax.set_xlabel("aa node")
    ax.set_ylabel("Summed node feature importance")
    ax.set_title(f"Per-aa summed importance (edge_type={edge_type})")

    # X ticks: show all if few nodes, otherwise rotate and reduce fontsize
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, rotation=90, fontsize=6 if num_aa > 30 else 8)

    plt.tight_layout()

    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()

    return {"fig": fig, "values": aa_importance, "indices": indices.tolist()}
