import torch
import yaml
import logging
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import os
from torch_geometric.explain import Explainer, CaptumExplainer
from torch_geometric.loader import NeighborLoader
from src.data.dataloading import SwissProtDataset, make_batch_transform
from src.models.gnn_model import ProteinGNN
from src.utils.helpers import timeit

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model_and_config(model_path, device):
    """Load the pretrained model and configuration."""
    # Load config
    with open(f"{model_path}/cfg.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create dataset to get vocab sizes
    dataset = SwissProtDataset(config)

    # Initialize model
    model = ProteinGNN(config, dataset)

    # Load pretrained weights
    state_dict = torch.load(
        f"{model_path}/model.pth", map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded model from {model_path}")
    logger.info(f"Model moved to device: {device}")

    return config, model, dataset


def get_loader(dataset, protein_names=None):
    """Get a batch for explanation generation."""
    # Create mask for specific proteins
    protein_indices = [dataset.protein_to_idx[pid] for pid in protein_names]
    mask = torch.zeros(len(dataset.proteins), dtype=torch.bool)
    mask[protein_indices] = True

    loader = NeighborLoader(
        dataset.data,
        num_neighbors={("protein", "aligned_with", "protein"): [-1]},
        batch_size=1,
        input_nodes=("protein", mask),
        transform=make_batch_transform(dataset, mode="predict"),
        shuffle=False,
        num_workers=0,
    )

    return loader


@timeit
def generate_explanations(model, batch, device):
    """Generate explanations using the model and batch."""
    batch = batch.to(device)

    # Create explainer
    explainer = Explainer(
        model,
        algorithm=CaptumExplainer("IntegratedGradients"),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        ),
    )

    # Generate explanations
    logger.info("Generating explanations...")
    hetero_explanation = explainer(
        batch.x_dict,
        batch.edge_index_dict,
        batch=batch,
        target=None,
        index=torch.arange(batch["protein"].batch_size),
    )

    logger.info(
        f"Generated explanations in {hetero_explanation.available_explanations}"
    )

    return hetero_explanation


@timeit
def plot_systemic_explanation(path, hetero_explanation, dataset, batch):
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
    node_mask = hetero_explanation["protein"]["node_mask"].detach().cpu()

    # Flatten node_mask if multidimensional (keep one score per protein node)
    if node_mask.dim() > 1:
        node_mask_flat = node_mask.sum(dim=1)
    else:
        node_mask_flat = node_mask

    # Local indices inside the batch for edges (these index into node_mask_flat)
    src_local, dst_local = edge_index[0], edge_index[1]

    # Normalize node mask across all batch nodes
    n_min, n_max = node_mask_flat.min(), node_mask_flat.max()
    node_mask_norm = (node_mask_flat - n_min) / (n_max - n_min + 1e-8)

    # Normalize edge mask across edges
    e_min, e_max = edge_mask.min(), edge_mask.max()
    edge_mask_norm = (edge_mask - e_min) / (e_max - e_min + 1e-8)

    # Construct NetworkX graph using local node ids (0..N-1) and local edge indices
    G = nx.Graph()
    for local_idx, global_idx in enumerate(batch_proteins["n_id"].tolist()):
        G.add_node(
            local_idx,
            global_id=global_idx,
            label=dataset.idx_to_protein.get(global_idx, str(global_idx)),
            color=float(node_mask_norm[local_idx].item()),
        )

    # Add edges using local indices and corresponding normalized edge importance
    for k in range(edge_index.shape[1]):
        s = int(src_local[k].item())
        d = int(dst_local[k].item())
        G.add_edge(s, d, color=float(edge_mask_norm[k].item()))

    # Prepare colors and labels using the local node ids
    node_colors = [G.nodes[n]["color"] for n in G.nodes()]
    edge_colors = [data for (_, _, data) in G.edges(data="color")]

    # Use textual labels (mapped from local->global->name) for plotting
    labels = {n: G.nodes[n]["label"] for n in G.nodes()}

    pos = nx.spring_layout(G, seed=42)
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.viridis)
    edges = nx.draw_networkx_edges(
        G, pos, edge_color=edge_colors, edge_cmap=plt.cm.viridis
    )
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

    protein = dataset.idx_to_protein[batch_proteins["n_id"][0].item()]
    plt.title(f"Protein-Protein Explanation: {protein}")
    plt.colorbar(nodes, label="Node mask")
    plt.colorbar(edges, label="Edge mask")
    filename = f"{path}/explanations/{protein}_system_explanation.png"
    os.makedirs(f"{path}/explanations", exist_ok=True)
    plt.savefig(filename)
    plt.show()
    plt.close()


@timeit
def plot_protein_explanation(path, hetero_explanation, dataset, batch):
    """
    Generate one amino-acid-to-protein explanation plot per protein node in the batch,
    normalizing node and edge importances per subgraph and naming outputs with dataset labels.
    """
    edge_mask = (
        hetero_explanation[("aa", "belongs_to", "protein")]["edge_mask"].detach().cpu()
    )
    edge_index = (
        hetero_explanation[("aa", "belongs_to", "protein")]["edge_index"].detach().cpu()
    )
    node_mask = hetero_explanation["aa"]["node_mask"].detach().cpu()

    if node_mask.dim() > 1:
        node_mask_flat = node_mask.sum(dim=1)
    else:
        node_mask_flat = node_mask

    src_local, dst_local = edge_index[0], edge_index[1]
    unique_targets = torch.unique(dst_local, sorted=True)

    protein_batch = hetero_explanation.batch["protein"].detach().cpu()
    protein_global_ids = protein_batch["n_id"].tolist()
    root_global = int(protein_global_ids[0])
    root_label = dataset.idx_to_protein.get(root_global, str(root_global))

    for dst_val in unique_targets.tolist():
        mask = dst_local == dst_val
        if torch.count_nonzero(mask) == 0:
            continue

        aa_indices = src_local[mask]
        edge_importance = edge_mask[mask]
        aa_importance = node_mask_flat[aa_indices]

        node_min = aa_importance.min()
        node_max = aa_importance.max()
        node_norm = (aa_importance - node_min) / (node_max - node_min + 1e-8)

        edge_min = edge_importance.min()
        edge_max = edge_importance.max()
        edge_norm = (edge_importance - edge_min) / (edge_max - edge_min + 1e-8)

        target_idx = int(dst_val)
        if target_idx >= len(protein_global_ids):
            continue
        target_global = int(protein_global_ids[target_idx])
        target_label = dataset.idx_to_protein.get(target_global, str(target_global))

        G = nx.Graph()
        protein_node_id = ("protein", target_idx)
        protein_color = float(node_norm.max().item()) if node_norm.numel() > 0 else 0.0
        G.add_node(protein_node_id, label=target_label, color=protein_color)

        for aa_local, n_score, e_score in zip(
            aa_indices.tolist(), node_norm.tolist(), edge_norm.tolist()
        ):
            aa_node_id = ("aa", aa_local)
            G.add_node(aa_node_id, label=f"AA_{aa_local}", color=float(n_score))
            G.add_edge(aa_node_id, protein_node_id, color=float(e_score))

        node_colors = [attrs["color"] for _, attrs in G.nodes(data=True)]
        edge_colors = [color for _, _, color in G.edges(data="color")]
        labels = {node: attrs["label"] for node, attrs in G.nodes(data=True)}

        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)
        nodes = nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, cmap=plt.cm.viridis
        )
        edges = nx.draw_networkx_edges(
            G, pos, edge_color=edge_colors, edge_cmap=plt.cm.plasma
        )
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

        plt.title(f"AA-Protein Explanation: {target_label}")
        plt.colorbar(nodes, label="AA node mask")
        plt.colorbar(edges, label="Edge mask")
        plt.tight_layout()
        filename = f"{path}/explanations/{root_label}_{target_label}_aa_explanation.png"
        os.makedirs(f"{path}/explanations", exist_ok=True)
        plt.savefig(filename)
        plt.show()
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate explanations for pretrained protein function prediction model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to pretrained model weights (.pth file)",
    )
    parser.add_argument(
        "--proteins",
        nargs="*",
        default=None,
        help="Specific protein indices to explain (if None, uses test proteins)",
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model and config
    config, model, dataset = load_model_and_config(args.model_path, device)
    loader = get_loader(dataset, args.proteins)
    # Get batch for explanation
    for batch in loader:
        # Generate explanations
        hetero_explanation = generate_explanations(model, batch, device)
        # Plot protein-protein explanation
        plot_systemic_explanation(args.model_path, hetero_explanation, dataset, batch)
        plot_protein_explanation(args.model_path, hetero_explanation, dataset, batch)
        # create_visualizations(hetero_explanation, batch, args.model_path)

    logger.info(
        f"Explanation generation completed! Results saved to: {args.model_path}"
    )


if __name__ == "__main__":
    main()
