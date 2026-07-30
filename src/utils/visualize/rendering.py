"""PyMOL-based 3D structure rendering with score overlays."""

import colorsys
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import rankdata
from torch_scatter import scatter

from src.utils.visualize.context import build_plot_context, ensure_structure
from src.utils.visualize.residue import build_protein_score_map

try:
    import pymol2
except ImportError:
    pymol2 = None

logger = logging.getLogger(__name__)


def adjust_colormap(cmap, luminance_factor=1.0, saturation_factor=1.0, n_colors=256):
    """Adjust the luminance and saturation of a colormap.

    Args:
        cmap: The input colormap to adjust.
        luminance_factor: Factor to scale the luminance (value) of the colors.
        saturation_factor: Factor to scale the saturation of the colors.
        n_colors: Number of colors in the adjusted colormap.

    Returns:
        The adjusted colormap.
    """
    colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]
    adjusted_colors = []
    for color in colors:
        r, g, b = color[:3]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        s = min(1.0, s * saturation_factor)
        v = min(1.0, v * luminance_factor)
        adjusted_colors.append(colorsys.hsv_to_rgb(h, s, v))

    return mcolors.LinearSegmentedColormap.from_list(
        "adjusted_cmap", adjusted_colors, N=n_colors
    )


def _apply_spectrum(cmd, cmap, scores, ca_resi, n_colors=256):
    """Generate a custom cmap and apply it to a PyMOL scene."""
    cmap = adjust_colormap(cmap, luminance_factor=1, saturation_factor=4)
    sampled_colors = [cmap(i / (n_colors - 1))[:3] for i in range(n_colors)]
    rgb_colors = [tuple(color) for color in sampled_colors]
    names = [mcolors.to_hex(color) for color in sampled_colors]

    for name, color in zip(names, rgb_colors):
        cmd.set_color(name, list(color))

    cmd.spectrum(
        expression="b",
        palette=" ".join(names),
        selection=f"resi {'+'.join(str(r) for r in ca_resi)}",
        minimum=float(np.min(scores)),
        maximum=float(np.max(scores)),
    )


def render_scene(
    structure_path: str,
    residue_scores: Sequence[Tuple[int, float]],
    image_path: str,
    *,
    title: str | None = None,
) -> None:
    """Color a structure by residue scores via PyMOL. Supports both .pdb and .cif files.

    Args:
        structure_path: Path to the structure file.
        residue_scores: Sequence of (residue_number_1based, score) tuples.
        image_path: Path to save the rendered image.
        title: Optional title for the rendering.
    """
    if not residue_scores:
        logger.warning(f"No residue scores for {structure_path}, skipping")
        return

    # Build lookup from PDB residue number to score
    score_map = dict(residue_scores)
    score_values = np.asarray([s for _, s in residue_scores], dtype=np.float32)

    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    logger.info(f"Rendering structure {structure_path}")

    with pymol2.PyMOL() as pymol:
        cmd = pymol.cmd
        cmd.reinitialize()
        cmd.load(structure_path, "prot")
        cmd.alter("prot", "b=0.0")
        ca_resi = [int(atom.resi) for atom in cmd.get_model("prot and name CA").atom]
        scored_resi = []
        for resi_nb in ca_resi:
            if resi_nb in score_map:
                cmd.alter(f"prot and resi {resi_nb}", f"b={score_map[resi_nb]}")
                scored_resi.append(resi_nb)
        if not scored_resi:
            logger.warning(
                f"No residue scores matched CA atoms in {structure_path}, skipping rendering."
            )
            return
        if len(scored_resi) != len(score_map):
            logger.warning(
                f"Matched {len(scored_resi)}/{len(score_map)} scored residues to CA atoms in {structure_path}."
            )
        _apply_spectrum(cmd, plt.cm.Spectral_r, score_values, scored_resi)
        # Color residues without scores gray
        unscored_sel = f"not resi {'+'.join(str(r) for r in scored_resi)}"
        cmd.color("gray70", selection=unscored_sel)
        cmd.color("gray70", selection="HETATM")

        if title:
            cmd.set_title("title", state=0, text=title)
        cmd.set("spec_reflect", 0)
        cmd.set("ray_shadows", 0)
        cmd.set("ray_opaque_background", 0)
        cmd.bg_color("black")
        cmd.orient("prot")
        cmd.png(image_path, width=1600, height=1200, dpi=300, ray=1)
        cmd.save(image_path.replace(".png", ".pse"))
        logger.info(f"Saved structure rendering to {image_path}")


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
    global_dir: Optional[str] = None,
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
            structure_path = structure_cache[uniprot_id]
        else:
            pdb_id = (
                dataset.pid_mapping.get(uniprot_id, uniprot_id)
                if dataset.uses_entryid
                else uniprot_id
            )
            try:
                structure_path = ensure_structure(pdb_id, context.seed_dir, global_dir=global_dir)
            except FileNotFoundError as e:
                logger.warning(f"Skipping 3D rendering for {uniprot_id}: {e}")
                continue
            if structure_cache is not None:
                structure_cache[uniprot_id] = structure_path

        # Resolve output path - Neighbor or seed protein ?
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
            structure_path, residues, image_path, title=f"{uniprot_id} – {title_prefix}"
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
    global_dir: Optional[str] = None,
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
        suffix=f"scene3d_aa_raw_attention_L{layer_idx}",
        title_prefix=f"Attention L{layer_idx}",
        structure_cache=structure_cache,
        go_term=go_term,
        plot_neighbors=plot_neighbors,
        global_dir=output_dir,
    )

    # Render AA -> AA attention
    aa_key = ("aa", "close_to", "aa")
    if aa_key in layer_attention:
        aa_edge_index, aa_attn_weights = layer_attention[aa_key]
        aa_edge_index = aa_edge_index.detach().cpu()
        aa_attn_weights = aa_attn_weights.detach().cpu()

        if aa_attn_weights.dim() > 1:
            aa_attn_weights = aa_attn_weights.mean(dim=-1)

        # Aggregate AA->AA attention per source AA node
        node_sum_attn = scatter(aa_attn_weights, aa_edge_index[0], dim=0, reduce="sum")

        # Map AA nodes to their respective proteins using the belongs_to edge_index
        belongs_to_idx = edge_index.detach().cpu()
        if selected_key[1] == "rev_belongs_to":
            belongs_to_idx = belongs_to_idx.flip(0)

        aa_to_protein = {
            int(aa): int(prot) for aa, prot in zip(belongs_to_idx[0], belongs_to_idx[1])
        }

        # Build per-protein AA offset so we can convert global batch AA index
        # to a local 1-based residue position within each protein.
        protein_aa_offset = {}  # protein_local_idx -> first global AA index
        for aa_global, prot_local in sorted(aa_to_protein.items()):
            if prot_local not in protein_aa_offset:
                protein_aa_offset[prot_local] = aa_global

        aa_residue_scores = {}
        for aa_idx, score in enumerate(node_sum_attn.tolist()):
            if score > 0 and aa_idx in aa_to_protein:
                prot_idx = aa_to_protein[aa_idx]
                if prot_idx not in aa_residue_scores:
                    aa_residue_scores[prot_idx] = []
                # Convert global batch AA index to local 1-based residue position
                local_residue = aa_idx - protein_aa_offset[prot_idx] + 1
                aa_residue_scores[prot_idx].append((local_residue, score))

        _render_structures(
            context,
            dataset,
            context.protein_ids,
            aa_residue_scores,
            suffix=f"scene3d_aa_aa_raw_attention_L{layer_idx}",
            title_prefix=f"AA-AA Attention L{layer_idx}",
            structure_cache=structure_cache,
            go_term=go_term,
            plot_neighbors=plot_neighbors,
            global_dir=output_dir,
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
    suffix = f"scene3d_raw_captum_{go_term.replace(':', '_')}" if go_term else "scene3d_raw_captum"
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
        global_dir=output_dir,
    )


def _rank_values(values: torch.Tensor) -> torch.Tensor:
    """Return 1-based ranks of *values* as a float tensor (ties → average rank)."""
    arr = values.detach().cpu().numpy().astype(np.float64)
    ranks = rankdata(arr, method="average")
    return torch.from_numpy(ranks).float()


def export_captum_3d_rank(
    output_dir: str,
    dataset,
    batch,
    hetero_explanation,
    go_term: Optional[str] = None,
    structure_cache: Optional[Dict[str, str]] = None,
    plot_neighbors: bool = True,
) -> None:
    """Export Captum explanations to 3D renderings using per-edge-type *ranks*.

    Identical to :func:`export_captum_3d` but replaces raw attribution scores
    with their 1-based ranks within the ``(aa, belongs_to, protein)`` edge
    type.  The resulting colormap therefore reflects relative importance rather
    than absolute magnitude.
    """
    logger.info("Exporting ranked Captum explanations to 3D renderings...")
    key = ("aa", "belongs_to", "protein")

    edge_index = hetero_explanation[key]["edge_index"].detach().cpu()
    edge_scores = hetero_explanation[key]["edge_mask"].detach().cpu()

    # Replace scores with their ranks within this edge type
    rank_scores = _rank_values(edge_scores)

    residue_scores = _edge_scores_to_residues(edge_index, rank_scores)

    context = build_plot_context(output_dir, dataset, batch)
    suffix = (
        f"scene3d_rank_captum_{go_term.replace(':', '_')}"
        if go_term
        else "scene3d_rank_captum"
    )
    title_prefix = f"Captum Rank ({go_term})" if go_term else "Captum Rank"

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
        global_dir=output_dir,
    )


def export_layer_attention_3d_rank(
    output_dir: str,
    dataset,
    batch,
    layer_idx: int,
    layer_attention,
    structure_cache: Optional[Dict[str, str]] = None,
    go_term: Optional[str] = None,
    plot_neighbors: bool = True,
    global_dir: Optional[str] = None,
) -> None:
    """Export layer attention to 3D structure renderings using per-edge-type *ranks*.

    Identical to :func:`export_layer_attention_3d` but replaces raw attention
    scores with their 1-based ranks within each edge type.
    """
    keys = [
        ("aa", "belongs_to", "protein"),
    ]
    if layer_attention is None:
        return

    selected_key = next((k for k in keys if k in layer_attention), None)
    if selected_key is None:
        return

    edge_index, attn_weights = layer_attention[selected_key]
    attn_cpu = attn_weights.detach().cpu()
    # Rank within this edge type
    rank_scores = _rank_values(attn_cpu)
    residue_scores = _edge_scores_to_residues(
        edge_index.detach().cpu(),
        rank_scores,
    )

    context = build_plot_context(output_dir, dataset, batch)
    _render_structures(
        context,
        dataset,
        context.protein_ids,
        residue_scores,
        suffix=f"scene3d_aa_rank_attention_L{layer_idx}",
        title_prefix=f"Attention Rank L{layer_idx}",
        structure_cache=structure_cache,
        go_term=go_term,
        plot_neighbors=plot_neighbors,
        global_dir=output_dir,
    )

    # Render AA -> AA attention ranks
    aa_key = ("aa", "close_to", "aa")
    if aa_key in layer_attention:
        aa_edge_index, aa_attn_weights = layer_attention[aa_key]
        aa_edge_index = aa_edge_index.detach().cpu()
        aa_attn_weights = aa_attn_weights.detach().cpu()

        if aa_attn_weights.dim() > 1:
            aa_attn_weights = aa_attn_weights.mean(dim=-1)

        # Aggregate AA->AA attention per source AA node
        node_sum_attn = scatter(aa_attn_weights, aa_edge_index[0], dim=0, reduce="sum")

        # Rank the aggregated per-node values
        node_sum_ranked = _rank_values(node_sum_attn)

        # Map AA nodes to their respective proteins using the belongs_to edge_index
        belongs_to_idx = edge_index.detach().cpu()
        if selected_key[1] == "rev_belongs_to":
            belongs_to_idx = belongs_to_idx.flip(0)

        aa_to_protein = {
            int(aa): int(prot) for aa, prot in zip(belongs_to_idx[0], belongs_to_idx[1])
        }

        # Build per-protein AA offset
        protein_aa_offset = {}
        for aa_global, prot_local in sorted(aa_to_protein.items()):
            if prot_local not in protein_aa_offset:
                protein_aa_offset[prot_local] = aa_global

        aa_residue_scores = {}
        for aa_idx, score in enumerate(node_sum_ranked.tolist()):
            if score > 0 and aa_idx in aa_to_protein:
                prot_idx = aa_to_protein[aa_idx]
                if prot_idx not in aa_residue_scores:
                    aa_residue_scores[prot_idx] = []
                local_residue = aa_idx - protein_aa_offset[prot_idx] + 1
                aa_residue_scores[prot_idx].append((local_residue, score))

        _render_structures(
            context,
            dataset,
            context.protein_ids,
            aa_residue_scores,
            suffix=f"scene3d_aa_aa_rank_attention_L{layer_idx}",
            title_prefix=f"AA-AA Attention Rank L{layer_idx}",
            structure_cache=structure_cache,
            go_term=go_term,
            plot_neighbors=plot_neighbors,
            global_dir=output_dir,
        )

