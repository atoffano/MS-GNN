"""Model inference and interpretability utilities.

This module provides tools for running inference with trained models and generating
explanations for predictions using attention mechanisms and Captum attribution methods.
It includes functionality for:
- Model inference on protein data
- Attention weight analysis
- Captum-based attribution (IntegratedGradients, GradientShap)
- Correlation analysis between attention and attribution scores
"""

import torch
import yaml
import logging
import argparse
import os
from typing import Callable, Dict, List, Sequence, Tuple, Optional
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

SUPPORTED_CAPTUM_METHODS = [
    "IntegratedGradients",
    "Saliency",
    "InputXGradient",
    "Deconvolution",
    "ShapleyValueSampling",
    "GuidedBackprop",
]


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
    structure_cache: Optional[Dict[str, str]] = None,
    go_term: Optional[str] = None,
) -> None:
    """Render residue structures with optional structure caching.

    Args:
        context: Plot context
        dataset: Dataset object
        protein_ids: List of protein IDs
        residue_scores: Dict of residue scores per protein
        suffix: Filename suffix
        colormap: Colormap name
        title_factory: Function to generate title from UniProt ID
        structure_cache: Optional dict mapping UniProt ID to PDB file path
        go_term: Optional GO term ID for subdirectory organization
    """
    for local_idx, residues in residue_scores.items():
        if not residues:
            continue
        uniprot_id = dataset.idx_to_protein[protein_ids[local_idx]]
        out_dir, prefix = resolve_plot_target(context, local_idx)

        # Create GO term subdirectory if specified
        if go_term:
            go_subdir = go_term.replace(":", "_")
            out_dir = os.path.join(out_dir, go_subdir)
            os.makedirs(out_dir, exist_ok=True)

        # Use cached structure path if available, otherwise download
        if structure_cache and uniprot_id in structure_cache:
            pdb_path = structure_cache[uniprot_id]
        else:
            pdb_id = (
                dataset.pid_mapping[uniprot_id] if dataset.uses_entryid else uniprot_id
            )
            pdb_path = ensure_structure(pdb_id, out_dir)
            if structure_cache is not None:
                structure_cache[uniprot_id] = pdb_path

        image_path = os.path.join(out_dir, f"{prefix}_{suffix}.png")
        render_structure_colormap(
            pdb_path,
            residues,
            image_path,
            colormap=colormap,
            title=title_factory(uniprot_id),
        )


def preload_structures(dataset, batch, base_output_dir: str) -> Dict[str, str]:
    """Preload all required protein structures for the batch.

    Args:
        dataset: Dataset object
        batch: Data batch containing protein IDs
        base_output_dir: Base directory for structure files

    Returns:
        Dict mapping UniProt ID to PDB file path
    """
    logger.info("Preloading protein structures...")
    structure_cache = {}
    protein_ids = batch["protein"].n_id.detach().cpu().tolist()

    for protein_global_id in protein_ids:
        uniprot_id = dataset.idx_to_protein[protein_global_id]
        pdb_id = dataset.pid_mapping[uniprot_id] if dataset.uses_entryid else uniprot_id

        # Determine output directory
        context = build_plot_context(base_output_dir, dataset, batch)
        out_dir = context.root_dir

        try:
            pdb_path = ensure_structure(pdb_id, out_dir)
            structure_cache[uniprot_id] = pdb_path
            logger.debug(f"Cached structure for {uniprot_id}: {pdb_path}")
        except FileNotFoundError as e:
            logger.warning(f"Could not preload structure for {uniprot_id}: {e}")

    logger.info(f"Preloaded {len(structure_cache)} protein structures")
    return structure_cache


def export_layer_attention_3d(
    output_dir: str,
    dataset,
    batch,
    layer_idx: int,
    layer_attention,
    structure_cache: Optional[Dict[str, str]] = None,
    go_term: Optional[str] = None,
) -> None:
    """Export layer attention to 3D renderings.

    Args:
        output_dir: Output directory
        dataset: Dataset object
        batch: Data batch
        layer_idx: Layer index
        layer_attention: Attention weights for this layer
        structure_cache: Optional dict of cached structure paths
        go_term: Optional GO term ID for subdirectory organization
    """
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
        structure_cache=structure_cache,
        go_term=go_term,
    )


def export_captum_3d(
    output_dir: str,
    dataset,
    batch,
    hetero_explanation,
    go_term: Optional[str] = None,
    structure_cache: Optional[Dict[str, str]] = None,
) -> None:
    """Export Captum explanations to 3D renderings.

    Args:
        output_dir: Base output directory
        dataset: Dataset object
        batch: Data batch
        hetero_explanation: Explanation object
        go_term: Optional GO term ID to include in filename and subdirectory
        structure_cache: Optional dict of cached structure paths
    """
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
    suffix = f"captum_{go_term.replace(':', '_')}" if go_term else "captum"
    title_factory = lambda uid, term=go_term: (
        f"{uid} – Captum ({term})" if term else f"{uid} – Captum"
    )
    _render_residue_structures(
        context,
        dataset,
        context.protein_ids,
        residue_scores,
        suffix=suffix,
        colormap="rainbow",
        title_factory=title_factory,
        structure_cache=structure_cache,
        go_term=go_term,
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


def build_go_term_mapping(dataset) -> Dict[str, int]:
    """Build a complete mapping from GO term IDs to indices.

    Args:
        dataset: Dataset object with go_vocab_info

    Returns:
        Dict mapping GO term ID to index
        e.g., {'GO:0000001': 0, 'GO:0000002': 1, ...}
    """
    go_term_mapping = {}

    for _, vocab_info in dataset.go_vocab_info.items():
        for go_term, idx in vocab_info["go_to_idx"].items():
            go_term_mapping[go_term] = idx

    logger.info(f"Built GO term mapping with {len(go_term_mapping)} terms")
    return go_term_mapping


@timeit
def generate_explanations(
    model,
    batch,
    device,
    captum_method: str = "IntegratedGradients",
    explanation_type: str = "model",
    target_go_idx: Optional[int] = None,
):
    """Generate explanations using the model and batch.

    Args:
        model: Trained model
        batch: Data batch
        device: Computation device
        captum_method: Captum attribution method to use
        explanation_type: Either "model" (global) or "phenomenon" (target-specific)
        target_go_idx: Optional GO term index for phenomenon explanations
    """
    batch = batch.to(device)

    # Create explainer
    explainer = Explainer(
        model,
        algorithm=CaptumExplainer(captum_method),
        explanation_type=explanation_type,
        node_mask_type=None,
        edge_mask_type="object",
        model_config=dict(
            mode="binary_classification",
            task_level="node",
            return_type="probs",
        ),
    )

    logger.info(f"Generating {explanation_type} explanations with {captum_method}...")

    if explanation_type == "model":
        # Global model explanation
        hetero_explanation = explainer(
            batch.x_dict,
            batch.edge_index_dict,
            batch=batch,
        )
    else:
        # Phenomenon explanation for specific GO term
        if target_go_idx is None:
            raise ValueError(
                "target_go_idx must be provided for phenomenon explanations"
            )

        logger.info(f"Computing attributions for GO term index {target_go_idx}")
        target = torch.zeros(
            batch["protein"].y.size(1), dtype=torch.long, device=device
        )
        target[target_go_idx] = 1
        hetero_explanation = explainer(
            batch.x_dict,
            batch.edge_index_dict,
            batch=batch,
            target=target,
            index=None,
        )

    # Normalize edge masks
    for key in [
        ("aa", "belongs_to", "protein"),
        ("protein", "aligned_with", "protein"),
    ]:
        if key in hetero_explanation:
            hetero_explanation[key]["edge_mask"] = hetero_explanation[key][
                "edge_mask"
            ].abs()
            # Apply min max scaling
            em = hetero_explanation[key]["edge_mask"]
            em = (em - em.min()) / (em.max() - em.min() + 1e-8)
            hetero_explanation[key]["edge_mask"] = em

    return hetero_explanation


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
    parser.add_argument(
        "--go_terms",
        nargs="*",
        default=None,
        help="Specific GO terms to explain (e.g., GO:0000001 GO:0000002)",
    )
    parser.add_argument(
        "--captum_method",
        type=str,
        default="IntegratedGradients",
        choices=SUPPORTED_CAPTUM_METHODS,
        help="Captum attribution method to use",
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    _, model, dataset = load_model_and_config(args.model_path, device)

    # Build complete GO term mapping at start
    go_term_mapping = build_go_term_mapping(dataset)

    loader = get_loader(dataset, args.proteins)

    for batch in loader:
        batch = batch.to(device)

        # Preload all structures once for this batch
        structure_cache = preload_structures(dataset, batch, args.model_path)

        preds, attn = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch=batch,
            return_attention_weights=True,
        )

        # Always generate global model explanations
        logger.info("=" * 60)
        logger.info("Generating global model explanations...")
        logger.info("=" * 60)
        hetero_explanation = generate_explanations(
            model,
            batch,
            device,
            captum_method=args.captum_method,
            explanation_type="model",
        )

        logger.info("Plotting global explanations...")
        plot_systemic_explanation(args.model_path, hetero_explanation, dataset)
        plot_protein_explanation(args.model_path, hetero_explanation, dataset)
        export_captum_3d(
            args.model_path,
            dataset,
            batch,
            hetero_explanation,
            structure_cache=structure_cache,
        )

        # Generate GO term-specific explanations if requested
        if args.go_terms:
            logger.info("=" * 60)
            logger.info("Generating GO term-specific explanations...")
            logger.info("=" * 60)

            for go_term in args.go_terms:
                if go_term not in go_term_mapping:
                    logger.warning(
                        f"GO term {go_term} not found in vocabulary, skipping."
                    )
                    continue

                go_idx = go_term_mapping[go_term]
                logger.info(
                    f"\n--- Explaining GO term: {go_term} (index: {go_idx}) ---"
                )

                hetero_explanation_go = generate_explanations(
                    model,
                    batch,
                    device,
                    captum_method=args.captum_method,
                    explanation_type="phenomenon",
                    target_go_idx=go_idx,
                )

                logger.info(f"Plotting explanations for {go_term}...")

                # Export to protein-specific GO term subdirectories
                plot_systemic_explanation(
                    args.model_path, hetero_explanation_go, dataset
                )
                plot_protein_explanation(
                    args.model_path, hetero_explanation_go, dataset
                )
                export_captum_3d(
                    args.model_path,
                    dataset,
                    batch,
                    hetero_explanation_go,
                    go_term=go_term,
                    structure_cache=structure_cache,
                )

        # Export attention artifacts (same for all explanations)
        export_attention_artifacts(
            args.model_path, dataset, batch, attn, structure_cache
        )
        analyze_attention_captum_correlation(
            args.model_path, dataset, batch, attn, hetero_explanation
        )

    logger.info(
        f"Explanation generation completed! Results saved to: {args.model_path}"
    )


def export_attention_artifacts(
    output_dir, dataset, batch, attentions, structure_cache=None
):
    """Export attention artifacts with optional structure caching.

    Args:
        output_dir: Output directory
        dataset: Dataset object
        batch: Data batch
        attentions: List of attention weights per layer
        structure_cache: Optional dict of cached structure paths
    """
    if attentions is None:
        return
    for idx, layer_attention in enumerate(attentions, start=1):
        plot_systemic_attention(output_dir, layer_attention, dataset, batch, idx)
        plot_protein_attention(output_dir, layer_attention, dataset, batch, idx)
        export_layer_attention_3d(
            output_dir, dataset, batch, idx, layer_attention, structure_cache
        )


if __name__ == "__main__":
    main()
