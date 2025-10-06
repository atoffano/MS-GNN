import torch
import networkx as nx
import matplotlib.pyplot as plt
import os
from src.utils.helpers import timeit


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
    nodes = nx.draw_networkx_nodes(G, pos, cmap=plt.cm.viridis)
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
