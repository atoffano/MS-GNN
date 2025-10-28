"""Utilities for rendering protein structures with attribution scores."""

import logging
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from src.utils.visualize import (
    build_plot_context,
    resolve_plot_target,
    render_structure_colormap,
    ensure_structure,
)

logger = logging.getLogger(__name__)


def _edge_scores_to_residues(
    edge_index: torch.Tensor,
    scores: torch.Tensor,
    *,
    target_nodes: int,
) -> Dict[int, List[Tuple[int, float]]]:
    """Convert AA→protein edge scores into per-residue lists.

    Returns:
        Dict mapping protein_local_idx to [(residue_1_based, score), ...]
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
        aa_sorted = aa_local[order] + 1  # 1-based indexing
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
    """Render residue structures with optional structure caching."""
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

        # Get structure path from cache or download
        if structure_cache and uniprot_id in structure_cache:
            pdb_path = structure_cache[uniprot_id]
        else:
            pdb_id = (
                dataset.pid_mapping[uniprot_id] if dataset.uses_entryid else uniprot_id
            )
            pdb_path = ensure_structure(pdb_id, out_dir)
            if structure_cache is not None:
                structure_cache[uniprot_id] = pdb_path

        # Render structure
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
    """Export Captum explanations to 3D structure renderings."""
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
