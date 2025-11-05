"""Utilities for rendering protein structures with attribution scores."""

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch

from src.utils.visualize import (
    build_plot_context,
    render_structure_colormap,
    ensure_structure,
)

logger = logging.getLogger(__name__)


def _edge_scores_to_residues(
    edge_index: torch.Tensor, scores: torch.Tensor, target_nodes: int
) -> Dict[int, List[Tuple[int, float]]]:
    """Convert AA→protein edge scores into per-residue lists.

    Returns:
        Dict mapping protein_local_idx to [(residue_1_based, score), ...]
    """
    residue_dict = {}
    aa_idx, prot_idx = edge_index[0], edge_index[1]
    flat_scores = scores.view(-1)

    for protein_local in range(target_nodes):
        mask = prot_idx == protein_local
        if not torch.any(mask):
            continue

        aa_local = aa_idx[mask]
        vals = flat_scores[mask]

        # Sort by residue index
        order = torch.argsort(aa_local)
        residue_dict[protein_local] = list(
            zip((aa_local[order] + 1).tolist(), vals[order].tolist())
        )

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
):
    """Render residue structures with optional caching."""
    for local_idx, residues in residue_scores.items():
        if not residues:
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
            pdb_path = ensure_structure(pdb_id, context.root_dir)
            if structure_cache is not None:
                structure_cache[uniprot_id] = pdb_path

        # Resolve output path
        is_root = local_idx == 0
        base_dir = context.root_dir if is_root else context.neighbor_dir
        prefix = (
            context.root_label
            if is_root
            else f"{context.root_label}_{context.labels[local_idx]}"
        )

        if go_term:
            out_dir = os.path.join(base_dir, "per-term", go_term.replace(":", "_"))
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = base_dir

        image_path = f"{out_dir}/{prefix}_{suffix}.png"
        render_structure_colormap(
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
) -> None:
    """Export layer attention to 3D structure renderings."""
    key = ("aa", "belongs_to", "protein")
    if layer_attention is None or key not in layer_attention:
        return

    edge_index, attn_weights = layer_attention[key]
    residue_scores = _edge_scores_to_residues(
        edge_index.detach().cpu(),
        attn_weights.detach().cpu(),
        batch["protein"].batch_size,
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
    )


def export_captum_3d(
    output_dir: str,
    dataset,
    batch,
    hetero_explanation,
    go_term: Optional[str] = None,
    structure_cache: Optional[Dict[str, str]] = None,
) -> None:
    """Export Captum explanations to 3D structure renderings."""
    logger.info("Exporting Captum explanations to 3D renderings...")
    key = ("aa", "belongs_to", "protein")

    edge_index = hetero_explanation[key]["edge_index"].detach().cpu()
    edge_scores = hetero_explanation[key]["edge_mask"].detach().cpu()

    residue_scores = _edge_scores_to_residues(
        edge_index, edge_scores, batch["protein"].batch_size
    )

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
    )