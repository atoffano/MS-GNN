"""Utilities for rendering protein structures with attribution scores."""

import logging
import os
from typing import Dict, List, Optional, Tuple
import subprocess
import tempfile

import torch
from src.utils.constants import MUSCLE_EXECUTABLE
from src.utils.visualize import (
    build_plot_context,
    build_protein_score_map,
    render_structure_colormap,
    ensure_structure,
)

logger = logging.getLogger(__name__)


def _protein_scores_to_residue_list(
    protein_score_map: Dict[int, Dict[int, float]]
) -> Dict[int, List[Tuple[int, float]]]:
    """Convert protein score map to sorted residue lists with 1-based indexing.

    Args:
        protein_score_map: Dict mapping protein_idx to {aa_idx: score}.

    Returns:
        Dict mapping protein_idx to [(residue_1_based, score), ...] sorted by residue.
    """
    residue_dict = {}
    for protein_idx, aa_to_score in protein_score_map.items():
        if not aa_to_score:
            continue
        # Sort by AA index and convert to 1-based
        sorted_items = sorted(aa_to_score.items(), key=lambda x: x[0])
        residue_dict[protein_idx] = [(aa_idx + 1, score) for aa_idx, score in sorted_items]
    return residue_dict


def _edge_scores_to_residues(
    edge_index: torch.Tensor, scores: torch.Tensor
) -> Dict[int, List[Tuple[int, float]]]:
    """Convert AA→protein edge scores into per-residue lists.

    Returns:
        Dict mapping protein_local_idx to [(residue_1_based, score), ...]
    """
    protein_score_map = build_protein_score_map(edge_index, scores)
    return _protein_scores_to_residue_list(protein_score_map)


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
            pdb_path = ensure_structure(pdb_id, context.seed_dir)
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
        render_structure_colormap(
            pdb_path, residues, image_path, title=f"{uniprot_id} – {title_prefix}"
        )


def _perform_msa(sequences: List[str], labels: List[str]):
    """Perform multiple sequence alignment using MUSCLE via subprocess.

    Returns:
        aligned_seqs: List of aligned sequences with gaps
        alignment_mappings: List of dicts mapping original_aa_idx -> aligned_position for each protein
    """
    if len(sequences) < 2:
        # No alignment needed for single sequence
        return [sequences[0]], [{i: i for i in range(len(sequences[0]))}]

    logger.info("Performing MSA alignment for %d sequences", len(sequences))

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".fasta", delete=False
    ) as input_fasta:
        input_path = input_fasta.name
        for label, seq in zip(labels, sequences):
            input_fasta.write(f">{label}\n{seq}\n")

    output_path = input_path.replace(".fasta", "_aligned.fasta")

    try:
        cmd = [str(MUSCLE_EXECUTABLE), "-align", input_path, "-output", output_path]

        logger.debug("Running MUSCLE: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        if result.stderr:
            logger.debug("MUSCLE stderr: %s", result.stderr)

        # Parse the alignment manually
        aligned_seqs = []
        current_seq = ""
        current_label = None

        with open(output_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_label is not None and current_seq:
                        aligned_seqs.append(current_seq)
                    current_label = line[1:]
                    current_seq = ""
                else:
                    current_seq += line

            # Add the last sequence
            if current_label is not None and current_seq:
                aligned_seqs.append(current_seq)

        if len(aligned_seqs) != len(sequences):
            raise ValueError(
                f"Expected {len(sequences)} aligned sequences, got {len(aligned_seqs)}"
            )

        logger.debug("MSA completed successfully")
        return aligned_seqs

    except subprocess.CalledProcessError as e:
        logger.error("MUSCLE failed (exit code %d): %s", e.returncode, e.stderr)
        return sequences, [{i: i for i in range(len(seq))} for seq in sequences]
    except Exception as e:
        logger.error("MSA failed: %s (falling back to unaligned sequences)", e)
        return sequences, [{i: i for i in range(len(seq))} for seq in sequences]

    finally:
        try:
            os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
        except:
            pass


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
    logger.debug("Rendering Captum 3D structures%s", f" for {go_term}" if go_term else "")
    key = ("aa", "belongs_to", "protein")

    edge_index = hetero_explanation[key]["edge_index"].detach().cpu()
    edge_scores = hetero_explanation[key]["edge_mask"].detach().cpu()

    residue_scores = _edge_scores_to_residues(
        edge_index, edge_scores
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
