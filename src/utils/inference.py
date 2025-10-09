import torch
import yaml
import logging
import argparse
import numpy as np
import os
from typing import Callable, Dict, List, Sequence, Tuple
from torch_geometric.explain import (
    Explainer,
    CaptumExplainer,
)
from torch_geometric.loader import NeighborLoader
from src.data.dataloading import (
    SwissProtDataset,
    make_batch_transform,
)
from src.models.gnn_model import ProteinGNN
from src.utils.visualize import (
    plot_systemic_explanation,
    plot_protein_explanation,
    plot_systemic_attention,
    plot_protein_attention,
    ensure_structure,
    render_structure_colormap,
    analyze_attention_captum_correlation,
    build_plot_context,
    resolve_plot_target,
)
from src.utils.helpers import timeit

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _edge_scores_to_residues(
    edge_index: torch.Tensor,
    scores: torch.Tensor,
    *,
    target_nodes: int,
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Convert AA→protein edge scores into per-residue lists.
    Returns {protein_local_idx: [(resi_1_based, score), ...]}.
    """
    residue_dict: Dict[int, List[Tuple[int, float]]] = {}
    aa_idx = edge_index[0]
    prot_idx = edge_index[1]
    flat_scores = scores.view(-1)

    for protein_local in range(target_nodes):
        mask = prot_idx == protein_local
        if not torch.any(mask):
            continue
        aa_local = aa_idx[mask]
        vals = flat_scores[mask]
        order = torch.argsort(aa_local)
        aa_sorted = aa_local[order] + 1  # convert to 1-based indexing
        vals_sorted = vals[order]
        residue_dict[protein_local] = list(
            zip(aa_sorted.tolist(), vals_sorted.tolist())
        )
    return residue_dict


def _render_residue_structures(
    context,
    dataset,
    protein_ids: Sequence[int],
    residue_scores: Dict[int, List[Tuple[int, float]]],
    *,
    suffix: str,
    colormap: str,
    title_factory: Callable[[str], str],
) -> None:
    for local_idx, residues in residue_scores.items():
        if not residues:
            continue
        uniprot_id = dataset.idx_to_protein[protein_ids[local_idx]]
        out_dir, prefix = resolve_plot_target(context, local_idx)
        pdb_path = ensure_structure(uniprot_id, out_dir)
        image_path = os.path.join(out_dir, f"{prefix}_{suffix}.png")
        render_structure_colormap(
            pdb_path,
            residues,
            image_path,
            colormap=colormap,
            title=title_factory(uniprot_id),
        )


def export_layer_attention_3d(
    output_dir: str,
    dataset,
    batch,
    layer_idx: int,
    layer_attention,
) -> None:
    key = ("aa", "belongs_to", "protein")
    if layer_attention is None or key not in layer_attention:
        return

    edge_index, attn_weights = layer_attention[key]
    residue_scores = _edge_scores_to_residues(
        edge_index.detach().cpu(),
        attn_weights.detach().cpu(),
        target_nodes=batch["protein"].batch_size,
    )

    context = build_plot_context(output_dir, dataset, batch)
    title_factory = lambda uid, idx=layer_idx: f"{uid} – Attention L{idx}"
    _render_residue_structures(
        context,
        dataset,
        context.protein_ids,
        residue_scores,
        suffix=f"attention_layer{layer_idx}",
        colormap="rainbow",
        title_factory=title_factory,
    )


def export_captum_3d(
    output_dir: str,
    dataset,
    batch,
    hetero_explanation,
) -> None:
    logger.info("Exporting Captum explanations to 3D renderings...")
    key = ("aa", "belongs_to", "protein")

    edge_index = hetero_explanation[key]["edge_index"].detach().cpu()
    edge_scores = hetero_explanation[key]["edge_mask"].detach().cpu()
    residue_scores = _edge_scores_to_residues(
        edge_index,
        edge_scores,
        target_nodes=batch["protein"].batch_size,
    )

    context = build_plot_context(output_dir, dataset, batch)
    _render_residue_structures(
        context,
        dataset,
        context.protein_ids,
        residue_scores,
        suffix="captum",
        colormap="rainbow",
        title_factory=lambda uid: f"{uid} – Captum",
    )


def load_model_and_config(model_path, device):
    """Load the pretrained model and configuration."""
    # Load config
    with open(f"{model_path}/cfg.yaml", "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loading dataset..")
    dataset = SwissProtDataset(config)
    logger.info(f"Loaded dataset.\n Loading model..")
    model = ProteinGNN(config, dataset)
    state_dict = torch.load(
        f"{model_path}/model.pth", map_location=device, weights_only=True
    )
    for key in list(state_dict.keys()):
        if key.startswith("_orig_mod."):
            state_dict[key.replace("_orig_mod.", "")] = state_dict.pop(key)
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


def export_attention_artifacts(output_dir, dataset, batch, attentions):
    if attentions is None:
        return
    for idx, layer_attention in enumerate(attentions, start=1):
        plot_systemic_attention(output_dir, layer_attention, dataset, batch, idx)
        plot_protein_attention(output_dir, layer_attention, dataset, batch, idx)
        export_layer_attention_3d(output_dir, dataset, batch, idx, layer_attention)


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model and config
    _, model, dataset = load_model_and_config(args.model_path, device)
    loader = get_loader(dataset, args.proteins)

    for batch in loader:
        batch = batch.to(device)
        preds, attn = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch=batch,
            return_attention_weights=True,
        )

        logger.info("Generating explanations..")
        hetero_explanation = generate_explanations(model, batch, device)

        logger.info("Plotting explanations..")
        plot_systemic_explanation(args.model_path, hetero_explanation, dataset)
        plot_protein_explanation(args.model_path, hetero_explanation, dataset)

        export_captum_3d(args.model_path, dataset, batch, hetero_explanation)
        export_attention_artifacts(args.model_path, dataset, batch, attn)
        analyze_attention_captum_correlation(
            args.model_path, dataset, batch, attn, hetero_explanation
        )

    logger.info(
        f"Explanation generation completed! Results saved to: {args.model_path}"
    )


if __name__ == "__main__":
    main()
