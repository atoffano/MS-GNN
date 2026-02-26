"""Utilities for rendering protein structures with attribution scores."""

import logging
import os
from typing import Dict, List, Optional, Tuple
import torch
from torch_scatter import scatter
from src.utils.visualize import (
    build_plot_context,
    build_protein_score_map,
    render_scene,
    ensure_structure,
)

logger = logging.getLogger(__name__)


def _edge_scores_to_residues(
    edge_index: torch.Tensor, scores: torch.Tensor
) -> Dict[int, List[Tuple[int, float]]]:
    """Convert AA→protein edge scores into per-residue lists.

    Returns:
        Dict mapping protein_local_idx to [(residue_1_based, score), ...]
    """
    protein_score_map = build_protein_score_map(edge_index, scores)
    residue_dict = {}
    for protein_idx, aa_to_score in protein_score_map.items():
        # Convert to 1-based indexing
        residue_dict[protein_idx] = [
            (aa_idx + 1, score) for aa_idx, score in aa_to_score.items()
        ]
    return residue_dict


def _render_structures(
    context,
    dataset,
    protein_ids: List[int],
    residue_scores: Dict[int, List[Tuple[int, float]]],
    suffix: str,
    title_prefix: str,
    structure_cache: Optional[Dict[str, str]] = None,
    go_term: Optional[str] = None,
    plot_neighbors: bool = True,
):
    """Render residue structures with optional caching."""
    for local_idx, residues in residue_scores.items():
        if not residues:
            continue

        # seed protein or neighbor ?
        is_seed = context.protein_ids[local_idx] == context.seed_global
        if not plot_neighbors and not is_seed:
            continue

        uniprot_id = dataset.idx_to_protein[protein_ids[local_idx]]

        # Get structure path with caching
        if structure_cache and uniprot_id in structure_cache:
            pdb_path = structure_cache[uniprot_id]
        else:
            pdb_id = (
                dataset.pid_mapping.get(uniprot_id, uniprot_id)
                if dataset.uses_entryid
                else uniprot_id
            )
            try:
                pdb_path = ensure_structure(pdb_id, context.seed_dir)
            except FileNotFoundError as e:
                logger.warning(f"Skipping 3D rendering for {uniprot_id}: {e}")
                continue
            if structure_cache is not None:
                structure_cache[uniprot_id] = pdb_path

        # Resolve output path - Neighbor or seed protein ?
        is_seed = local_idx == 0
        base_dir = context.seed_dir if is_seed else context.neighbor_dir
        prefix = (
            context.seed_label
            if is_seed
            else f"{context.seed_label}_{context.labels[local_idx]}"
        )

        if go_term:
            out_dir = os.path.join(base_dir, "per-term", go_term.replace(":", "_"))
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = base_dir

        image_path = f"{out_dir}/{prefix}_{suffix}.png"
        render_scene(
            pdb_path, residues, image_path, title=f"{uniprot_id} – {title_prefix}"
        )


def export_layer_attention_3d(
    output_dir: str,
    dataset,
    batch,
    layer_idx: int,
    layer_attention,
    structure_cache: Optional[Dict[str, str]] = None,
    go_term: Optional[str] = None,
    plot_neighbors: bool = True,
) -> None:
    """Export layer attention to 3D structure renderings."""
    keys = [
        ("aa", "belongs_to", "protein"),
    ]
    if layer_attention is None:
        return

    selected_key = next((k for k in keys if k in layer_attention), None)
    if selected_key is None:
        return

    edge_index, attn_weights = layer_attention[selected_key]
    residue_scores = _edge_scores_to_residues(
        edge_index.detach().cpu(),
        attn_weights.detach().cpu(),
    )

    context = build_plot_context(output_dir, dataset, batch)
    _render_structures(
        context,
        dataset,
        context.protein_ids,
        residue_scores,
        suffix=f"attention_layer{layer_idx}",
        title_prefix=f"Attention L{layer_idx}",
        structure_cache=structure_cache,
        go_term=go_term,
        plot_neighbors=plot_neighbors,
    )

    # Render AA -> AA attention
    aa_key = ("aa", "close_to", "aa")
    if aa_key in layer_attention:
        aa_edge_index, aa_attn_weights = layer_attention[aa_key]
        aa_edge_index = aa_edge_index.detach().cpu()
        aa_attn_weights = aa_attn_weights.detach().cpu()

        if aa_attn_weights.dim() > 1:
            aa_attn_weights = aa_attn_weights.mean(dim=-1)

        # Aggregate AA->AA attention per source AA node with degree smoothing
        node_sum_attn = scatter(aa_attn_weights, aa_edge_index[0], dim=0, reduce="sum")
        node_degree = scatter(
            torch.ones_like(aa_attn_weights), aa_edge_index[0], dim=0, reduce="sum"
        )
        smoothing_factor = node_degree.mean().item() if node_degree.numel() > 0 else 1.0
        node_avg_attn = node_sum_attn / (node_degree + smoothing_factor)

        # Map AA nodes to their respective proteins using the belongs_to edge_index
        belongs_to_idx = edge_index.detach().cpu()
        if selected_key[1] == "rev_belongs_to":
            belongs_to_idx = belongs_to_idx.flip(0)

        aa_to_protein = {
            int(aa): int(prot) for aa, prot in zip(belongs_to_idx[0], belongs_to_idx[1])
        }

        aa_residue_scores = {}
        for aa_idx, score in enumerate(node_avg_attn.tolist()):
            if score > 0 and aa_idx in aa_to_protein:
                prot_idx = aa_to_protein[aa_idx]
                if prot_idx not in aa_residue_scores:
                    aa_residue_scores[prot_idx] = []
                aa_residue_scores[prot_idx].append((aa_idx + 1, score))

        _render_structures(
            context,
            dataset,
            context.protein_ids,
            aa_residue_scores,
            suffix=f"aa_aa_attention_layer{layer_idx}",
            title_prefix=f"AA-AA Attention L{layer_idx}",
            structure_cache=structure_cache,
            go_term=go_term,
            plot_neighbors=plot_neighbors,
        )


def export_captum_3d(
    output_dir: str,
    dataset,
    batch,
    hetero_explanation,
    go_term: Optional[str] = None,
    structure_cache: Optional[Dict[str, str]] = None,
    plot_neighbors: bool = True,
) -> None:
    """Export Captum explanations to 3D structure renderings."""
    logger.info("Exporting Captum explanations to 3D renderings...")
    key = ("aa", "belongs_to", "protein")

    edge_index = hetero_explanation[key]["edge_index"].detach().cpu()
    edge_scores = hetero_explanation[key]["edge_mask"].detach().cpu()

    residue_scores = _edge_scores_to_residues(edge_index, edge_scores)

    context = build_plot_context(output_dir, dataset, batch)
    suffix = f"captum_{go_term.replace(':', '_')}" if go_term else "captum"
    title_prefix = f"Captum ({go_term})" if go_term else "Captum"

    _render_structures(
        context,
        dataset,
        context.protein_ids,
        residue_scores,
        suffix=suffix,
        title_prefix=title_prefix,
        structure_cache=structure_cache,
        go_term=go_term,
        plot_neighbors=plot_neighbors,
    )
