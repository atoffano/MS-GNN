"""Visualization utilities for model interpretability and analysis."""

import logging
import os
import shutil
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys

import matplotlib.pyplot as plt

import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
import torch

from src.utils.constants import RANDOM_SEED
from src.utils.api import download_alphafold, download_pdb
from src.utils.helpers import timeit

try:
    import pymol2
except ImportError:
    pymol2 = None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ProteinPlotContext:
    """Context for protein visualization."""

    seed_label: str
    seed_global: int
    seed_dir: str
    neighbor_dir: str
    protein_ids: list[int]
    labels: dict[int, str]


def build_plot_context(base_path: str, dataset, batch) -> ProteinPlotContext:
    """Build plotting context from batch."""
    protein_ids = batch["protein"].n_id.detach().cpu().tolist()
    seed_global = int(protein_ids[0])
    labels = {
        idx: dataset.idx_to_protein.get(global_id, str(global_id))
        for idx, global_id in enumerate(protein_ids)
    }
    seed_label = labels[0]
    seed_dir = os.path.join(base_path, "explanations", seed_label)
    neighbor_dir = os.path.join(seed_dir, "neighbors")
    os.makedirs(seed_dir, exist_ok=True)
    os.makedirs(neighbor_dir, exist_ok=True)

    return ProteinPlotContext(
        seed_label=seed_label,
        seed_global=seed_global,
        seed_dir=seed_dir,
        neighbor_dir=neighbor_dir,
        protein_ids=protein_ids,
        labels=labels,
    )


def ensure_structure(uniprot_id: str, out_dir: str) -> str:
    """Get structure for UniProt ID, checking caches before downloading.

    Priority: output cache -> alphafold_pdb -> esmfold_pdb -> RCSB PDB -> AlphaFold download
    """
    os.makedirs(out_dir, exist_ok=True)
    pdb_path = os.path.join(out_dir, f"{uniprot_id}.pdb")
    if os.path.exists(pdb_path):
        return pdb_path

    # Check local data directories
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    for cache_name in ["alphafold_pdb", "esmfold_pdb"]:
        cache_path = os.path.join(
            current_file_dir,
            "..",
            "..",
            "data",
            "swissprot",
            "2024_01",
            cache_name,
            f"{uniprot_id}.pdb",
        )
        if os.path.exists(cache_path):
            shutil.copy2(cache_path, pdb_path)
            return pdb_path

    # # Download from remote sources
    # if download_pdb(uniprot_id, pdb_path):
    #     logger.info(f"Downloaded PDB structure for {uniprot_id} from RCSB")
    #     return pdb_path

    # if download_alphafold(uniprot_id, pdb_path):
    #     logger.info(f"Downloaded PDB structure for {uniprot_id} from AlphaFold")
    #     return pdb_path

    raise FileNotFoundError(f"No structure available for {uniprot_id}")


def adjust_colormap(cmap, luminance_factor=1.0, saturation_factor=1.0, n_colors=256):
    """
    Adjust the luminance and saturation of a colormap.

    Parameters:
    -----------
    cmap : matplotlib.colors.Colormap
        The input colormap to adjust.
    luminance_factor : float
        Factor to scale the luminance (value) of the colors.
    saturation_factor : float
        Factor to scale the saturation of the colors.
    n_colors : int
        Number of colors in the adjusted colormap.

    Returns:
    --------
    new_cmap : matplotlib.colors.LinearSegmentedColormap
        The adjusted colormap.
    """
    # Sample the colormap
    colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]

    # Convert RGBA to RGB and adjust luminance and saturation
    adjusted_colors = []
    for color in colors:
        r, g, b = color[:3]  # Ignore alpha channel
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        s = min(1.0, s * saturation_factor)  # Scale saturation
        v = min(1.0, v * luminance_factor)  # Scale luminance
        adjusted_colors.append(colorsys.hsv_to_rgb(h, s, v))

    # Create a new colormap
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        "adjusted_cmap", adjusted_colors, N=n_colors
    )
    return new_cmap


def apply_spectrum(pymol_cmd, selection, data, cmap, n_colors=256):
    """
    Generates a custom color ramp based 'jet' cmap and applies it to a PyMOL selection.
    """
    cmap = adjust_colormap(cmap, luminance_factor=1, saturation_factor=4)
    sampled_colors = [cmap(i / (n_colors - 1))[:3] for i in range(n_colors)]

    # Convert to RGB tuples and hex names
    rgb_colors = [tuple(color) for color in sampled_colors]
    names = [mcolors.to_hex(color) for color in sampled_colors]

    for name, color in zip(names, rgb_colors):
        pymol_cmd.set_color(name, list(color))
    pymol_cmd.spectrum(
        "b",
        " ".join(names),
        selection,
        minimum=float(np.min(data)),
        maximum=float(np.max(data)),
    )


def render_structure_colormap(
    pdb_path: str,
    residue_scores: Sequence[Tuple[int, float]],
    image_path: str,
    *,
    title: str | None = None,
) -> None:
    """Color a structure by residue scores via PyMOL."""

    if not residue_scores:
        logger.warning(f"No residue scores for {pdb_path}, skipping")
        return

    res_idx, scores = zip(*residue_scores)
    res_idx = np.asarray(res_idx, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32)

    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    logger.info(f"Rendering structure {pdb_path}")

    with pymol2.PyMOL() as pymol:
        cmd = pymol.cmd
        cmd.reinitialize()
        cmd.alter("prot", "b=0.0")
        apply_spectrum(cmd, "prot", scores, cmap=plt.cm.Spectral_r)
        cmd.load(pdb_path, "prot")
        cmd.select("alpha_carbon", "prot and name CA")
        ca_ids = [atom.id for atom in cmd.get_model("prot and name CA").atom]

        if len(ca_ids) != len(scores):
            logger.warning(
                f"Number of scores ({len(scores)}) does not match the number of CA atoms ({len(ca_ids)}). Only assigning up to the shorter list."
            )
        num_assignments = min(len(ca_ids), len(scores))
        for i in range(num_assignments):
            atom_id = ca_ids[i]
            score = scores[i]
            cmd.alter(f"prot and name CA and id {atom_id}", f"b={score}")
        cmd.spectrum("b", selection="prot and name CA")

        if title:
            cmd.set_title("title", state=0, text=title)
        cmd.set("spec_reflect", 0)
        cmd.set("ray_shadows", 0)
        cmd.set("ray_opaque_background", 0)
        cmd.bg_color("black")
        cmd.orient("prot")
        cmd.png(image_path, width=1600, height=1200, dpi=300, ray=1)
        cmd.save(image_path.replace(".png", "_scene.pse"))
        logger.info(f"Saved structure rendering to {image_path}")


def _save_plot(
    context: ProteinPlotContext,
    filename_suffix: str,
    local_idx: int = 0,
    go_term: Optional[str] = None,
):
    """Save current plot to appropriate directory."""
    is_seed = context.protein_ids[local_idx] == context.seed_global
    base_dir = context.seed_dir if is_seed else context.neighbor_dir
    prefix = (
        context.seed_label
        if is_seed
        else f"{context.seed_label}_{context.labels[local_idx]}"
    )

    if go_term:
        go_subdir = go_term.replace(":", "_")
        output_dir = os.path.join(base_dir, "per-term", go_subdir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = base_dir

    filename = os.path.join(output_dir, f"{prefix}_{filename_suffix}.png")
    plt.savefig(filename)
    plt.close()


def _plot_protein_network(
    context: ProteinPlotContext,
    edge_index: torch.Tensor,
    weights: torch.Tensor,
    title: str,
    colorbar_label: str,
    filename_suffix: str,
    go_term: Optional[str] = None,
    uniform_baseline: Optional[float] = None,
):
    """Create and save protein network graph."""
    G = nx.Graph()
    for local_idx, label in context.labels.items():
        G.add_node(local_idx, label=label)

    src, dst = edge_index[0], edge_index[1]
    for i in range(edge_index.size(1)):
        G.add_edge(int(src[i]), int(dst[i]), color=float(weights[i].item()))

    # Remove isolated nodes & associated labels
    G.remove_nodes_from(list(nx.isolates(G)))
    filtered_labels = {
        idx: label for idx, label in context.labels.items() if idx in set(G.nodes())
    }

    pos = nx.spring_layout(G, seed=RANDOM_SEED)
    nx.draw_networkx_nodes(G, pos, node_color="#d9d9d9")

    edges = None
    if G.number_of_edges():
        edge_colors = [data for (_, _, data) in G.edges(data="color")]
        # if uniform_baseline is not None:
        #     vals = np.array(edge_colors)
        #     edges = nx.draw_networkx_edges(
        #         G,
        #         pos,
        #         edge_color=edge_colors,
        #         edge_cmap=plt.cm.Spectral_r,
        #         edge_vmin=vals.min(),
        #         edge_vmax=vals.max(),
        #     )
        # else:
        edges = nx.draw_networkx_edges(
            G, pos, edge_color=edge_colors, edge_cmap=plt.cm.Spectral_r
        )
    nx.draw_networkx_labels(G, pos, labels=filtered_labels, font_size=9)
    plt.title(title)

    if edges is not None:
        plt.colorbar(edges, label=colorbar_label)

    plt.tight_layout()
    _save_plot(context, filename_suffix, go_term=go_term)


def _mean_attention(attention_weights: torch.Tensor) -> torch.Tensor:
    """Average attention weights over all heads."""
    return (
        attention_weights.mean(dim=-1)
        if attention_weights.dim() > 1
        else attention_weights.view(-1)
    )


def _plot_aa_to_protein_scatter(
    context: ProteinPlotContext,
    edge_index: torch.Tensor,
    edge_values: torch.Tensor,
    title_template: str,
    ylabel: str,
    filename_suffix: str,
    go_term: Optional[str] = None,
):
    """Unified scatter plot for AA→protein edges (attention or attribution)."""
    src_local, dst_local = edge_index[0], edge_index[1]

    for dst_val in torch.unique(dst_local, sorted=True).tolist():
        mask = dst_local == dst_val
        if not torch.any(mask):
            continue

        aa_indices = src_local[mask]
        values = edge_values[mask].view(-1)
        sort_idx = torch.argsort(aa_indices)
        values_sorted = values[sort_idx]

        target_idx = int(dst_val)
        target_label = context.labels[target_idx]

        plt.figure(figsize=(8, 4))
        x_positions = torch.arange(len(values_sorted), dtype=torch.float32)
        scatter = plt.scatter(
            x_positions.numpy(),
            values_sorted.numpy(),
            c=values_sorted.numpy(),
            cmap=plt.cm.Spectral_r,
        )
        plt.colorbar(scatter, label=ylabel)
        plt.xlabel("Residue")
        plt.ylabel(ylabel)
        plt.title(title_template.format(protein=target_label))
        plt.tight_layout()
        _save_plot(context, filename_suffix, target_idx, go_term)


def _plot_aa_to_protein_msa(
    context: ProteinPlotContext,
    edge_index: torch.Tensor,
    edge_values: torch.Tensor,
    aligned_seqs: list[str],
    title: str,
    ylabel: str,
    filename_suffix: str,
    go_term: Optional[str] = None,
    window_size: Optional[int] = 3,
    center_on_uniform: bool = False,
):
    """Plot AA→protein scores aligned by MSA."""
    src_local, dst_local = edge_index[0], edge_index[1]

    # Map protein index -> (aa_index -> score)
    protein_data = {}
    for dst_val in torch.unique(dst_local, sorted=True).tolist():
        mask = dst_local == dst_val
        if not torch.any(mask):
            continue
        aa_indices = src_local[mask].numpy()
        values = edge_values[mask].view(-1).numpy()
        protein_data[dst_val] = dict(zip(map(int, aa_indices), map(float, values)))

    if not protein_data or not aligned_seqs:
        return

    # Calculate AA offsets per protein
    offsets = {}
    cumulative = 0
    for i, seq in enumerate(aligned_seqs):
        offsets[i] = cumulative
        cumulative += sum(1 for c in seq if c != "-")

    # Plot
    plt.figure(figsize=(14, 6))
    cmap = plt.cm.get_cmap("tab10")

    if center_on_uniform:
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    for idx, (dst_val, aa_to_score) in enumerate(protein_data.items()):
        target_label = context.labels[int(dst_val)]
        is_seed = context.protein_ids[int(dst_val)] == context.seed_global
        aligned_seq = aligned_seqs[int(dst_val)]
        offset = offsets[int(dst_val)]

        # Calculate uniform baseline for this sequence if requested
        baseline = 0.0
        if center_on_uniform:
            seq_len = sum(1 for c in aligned_seq if c != "-")
            if seq_len > 0:
                baseline = 1.0 / seq_len

        x_pos, y_vals = [], []
        residue_idx = 0
        for aligned_pos, char in enumerate(aligned_seq):
            if char != "-":
                global_aa_idx = offset + residue_idx
                if global_aa_idx in aa_to_score:
                    x_pos.append(aligned_pos)
                    y_vals.append(aa_to_score[global_aa_idx] - baseline)
                residue_idx += 1
            else:
                x_pos.append(aligned_pos)
                y_vals.append(np.nan)

        if not x_pos:
            continue

        # Apply sliding window if requested
        if window_size and window_size > 1 and len(y_vals) >= window_size:
            kernel = np.ones(window_size) / window_size
            y_vals = np.convolve(y_vals, kernel, mode="valid")
            # Adjust x_pos to center the window
            trim_start = (window_size - 1) // 2
            trim_end = trim_start + len(y_vals)
            x_pos = x_pos[trim_start:trim_end]

        plt.plot(
            x_pos,
            y_vals,
            label=target_label,
            color=cmap(idx % 10),
            linewidth=3.0 if is_seed else 1.5,
            alpha=0.7,
            zorder=10 if is_seed else 5,
            marker="o" if is_seed else ".",
            markersize=6 if is_seed else 4,
        )

    plt.xlabel("Aligned Position")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_plot(context, filename_suffix, go_term=go_term)


def _plot_attn_seed_vs_neighbor_scatter(
    context: ProteinPlotContext,
    edge_index: torch.Tensor,
    edge_values: torch.Tensor,
    aligned_seqs: list[str],
    title: str,
    filename_suffix: str,
    go_term: Optional[str] = None,
):
    """Scatter plot of Seed Attention vs Neighbor Attention for aligned residues."""
    src_local, dst_local = edge_index[0], edge_index[1]

    # Map protein index -> (aa_index -> score)
    protein_data = {}
    for dst_val in torch.unique(dst_local, sorted=True).tolist():
        mask = dst_local == dst_val
        if not torch.any(mask):
            continue
        aa_indices = src_local[mask].numpy()
        values = edge_values[mask].view(-1).numpy()
        protein_data[dst_val] = dict(zip(map(int, aa_indices), map(float, values)))

    if not protein_data or not aligned_seqs:
        return

    # Calculate AA offsets per protein (global index of first AA in sequence)
    offsets = {}
    cumulative = 0
    for i, seq in enumerate(aligned_seqs):
        offsets[i] = cumulative
        cumulative += sum(1 for c in seq if c != "-")

    # Identify seed
    seed_local_idx = None
    for idx, global_id in enumerate(context.protein_ids):
        if global_id == context.seed_global:
            seed_local_idx = idx
            break

    if seed_local_idx is None or seed_local_idx not in protein_data:
        return

    seed_scores_map = protein_data[seed_local_idx]  # global_aa_idx -> score
    seed_offset = offsets[seed_local_idx]
    seed_seq = aligned_seqs[seed_local_idx]

    # Map MSA position to seed score
    msa_pos_to_seed_score = {}
    residue_idx = 0
    for msa_pos, char in enumerate(seed_seq):
        if char != "-":
            global_aa_idx = seed_offset + residue_idx
            if global_aa_idx in seed_scores_map:
                msa_pos_to_seed_score[msa_pos] = seed_scores_map[global_aa_idx]
            residue_idx += 1

    plt.figure(figsize=(10, 8))
    cmap = plt.cm.get_cmap("tab10")

    has_points = False

    for idx, (dst_val, neighbor_scores_map) in enumerate(protein_data.items()):
        if dst_val == seed_local_idx:
            continue  # Skip seed vs seed

        neighbor_label = context.labels[int(dst_val)]
        neighbor_seq = aligned_seqs[int(dst_val)]
        neighbor_offset = offsets[int(dst_val)]

        xs = []  # Seed attn scores
        ys = []  # Neighbor attn scores

        residue_idx = 0
        for msa_pos, char in enumerate(neighbor_seq):
            if char != "-":
                global_aa_idx = neighbor_offset + residue_idx
                if (
                    global_aa_idx in neighbor_scores_map
                    and msa_pos in msa_pos_to_seed_score
                ):
                    xs.append(msa_pos_to_seed_score[msa_pos])
                    ys.append(neighbor_scores_map[global_aa_idx])
                residue_idx += 1

        if xs:
            color = cmap(idx % 10)

            # Add linear regression line
            label_suffix = ""
            if len(xs) > 1:
                try:
                    m, b = np.polyfit(xs, ys, 1)
                    r = np.corrcoef(xs, ys)[0, 1]
                    x_range = np.array([min(xs), max(xs)])
                    y_range = m * x_range + b
                    plt.plot(
                        x_range,
                        y_range,
                        color=color,
                        linestyle="--",
                        alpha=0.5,
                        linewidth=1.5,
                    )
                    label_suffix = f" (R={r:.2f})"
                except Exception:
                    pass

            plt.scatter(
                xs,
                ys,
                label=f"{neighbor_label}{label_suffix}",
                alpha=0.6,
                s=20,
                color=color,
            )

            has_points = True

    if not has_points:
        plt.close()
        return

    plt.xlabel(f"Protein Attention ({context.seed_label})")
    plt.ylabel("Neighbor Attention")
    plt.title(title)
    plt.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, fontsize=9
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_plot(context, filename_suffix, go_term=go_term)


def _plot_msa_alignment_violin(
    context: ProteinPlotContext,
    edge_index: torch.Tensor,
    edge_values: torch.Tensor,
    aligned_seqs: list[str],
    title: str,
    filename_suffix: str,
    go_term: Optional[str] = None,
):
    """Violin plot comparing attention of residues aligned to seed vs not aligned."""
    src_local, dst_local = edge_index[0], edge_index[1]

    # Map protein index -> (aa_index -> score)
    protein_data = {}
    for dst_val in torch.unique(dst_local, sorted=True).tolist():
        mask = dst_local == dst_val
        if not torch.any(mask):
            continue
        aa_indices = src_local[mask].numpy()
        values = edge_values[mask].view(-1).numpy()
        protein_data[dst_val] = dict(zip(map(int, aa_indices), map(float, values)))

    if not protein_data or not aligned_seqs:
        return

    # Identify seed index
    seed_local_idx = None
    for idx, global_id in enumerate(context.protein_ids):
        if global_id == context.seed_global:
            seed_local_idx = idx
            break

    if seed_local_idx is None:
        return

    seed_seq = aligned_seqs[seed_local_idx]

    # Calculate AA offsets
    offsets = {}
    cumulative = 0
    for i, seq in enumerate(aligned_seqs):
        offsets[i] = cumulative
        cumulative += sum(1 for c in seq if c != "-")

    data_records = []

    for idx, (dst_val, aa_to_score) in enumerate(protein_data.items()):
        aligned_seq = aligned_seqs[int(dst_val)]
        offset = offsets[int(dst_val)]

        # Calculate uniform baseline
        seq_len = sum(1 for c in aligned_seq if c != "-")
        baseline = 1.0 / seq_len if seq_len > 0 else 0.0

        residue_idx = 0
        for msa_pos, char in enumerate(aligned_seq):
            if char != "-":
                global_aa_idx = offset + residue_idx
                if global_aa_idx in aa_to_score:
                    score = aa_to_score[global_aa_idx] - baseline

                    # Check alignment with seed
                    is_aligned = seed_seq[msa_pos] != "-"

                    condition = (
                        "Aligned to Protein in MSA" if is_aligned else "Not Aligned"
                    )
                    data_records.append(
                        {
                            "Normalized Attention": score,
                            "Condition": condition,
                        }
                    )
                residue_idx += 1

    if not data_records:
        return

    df = pd.DataFrame(data_records)

    if df.empty:
        return

    plt.figure(figsize=(6, 6))
    sns.violinplot(
        data=df,
        x="Condition",
        y="Normalized Attention",
        hue="Condition",
        inner="quartile",
        palette="muted",
        legend=False,
    )
    plt.title(title)
    plt.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    plt.tight_layout()
    _save_plot(context, filename_suffix, go_term=go_term)


def perform_msa_from_batch(batch) -> Optional[list[str]]:
    """Perform MSA on batch sequences, returning None on failure."""
    from src.utils.structure_renderer import _perform_msa

    sequences = batch["protein"].sequences
    labels = batch["protein"].protein_ids

    if len(sequences) < 2:
        logger.warning("Not enough sequences for MSA alignment")
        return None

    aligned_seqs = _perform_msa(sequences, labels)
    if len(aligned_seqs) != len(sequences):
        logger.error("MSA failed")
        return None

    return aligned_seqs


def plot_protein_explanation_msa(
    path: str,
    hetero_explanation,
    dataset,
    batch,
    title_suffix: Optional[str] = None,
    go_term: Optional[str] = None,
    aligned_seqs: Optional[list[str]] = None,
):
    """Plot amino acid to protein explanation aligned by MSA."""
    if aligned_seqs is None:
        aligned_seqs = perform_msa_from_batch(batch)
    if aligned_seqs is None:
        return

    context = build_plot_context(path, dataset, hetero_explanation.batch)
    key = ("aa", "belongs_to", "protein")

    title = f"AA-Protein Explanation (MSA-aligned): {context.seed_label}"
    if title_suffix:
        title += f" ({title_suffix})"

    _plot_aa_to_protein_msa(
        context,
        hetero_explanation[key]["edge_index"].detach().cpu(),
        hetero_explanation[key]["edge_mask"].detach().cpu(),
        aligned_seqs,
        title,
        "Edge Importance",
        "aa_explanation_msa",
        go_term,
    )


def plot_protein_attention_msa(
    path: str,
    layer_attention,
    dataset,
    batch,
    layer_idx: int,
    go_term: Optional[str] = None,
    aligned_seqs: Optional[list[str]] = None,
):
    """Plot amino acid to protein attention weights aligned by MSA."""
    if aligned_seqs is None:
        aligned_seqs = perform_msa_from_batch(batch)
    if aligned_seqs is None:
        return

    context = build_plot_context(path, dataset, batch)
    key = ("aa", "belongs_to", "protein")

    if layer_attention is None or key not in layer_attention:
        return

    edge_index, attn_weights = layer_attention[key]
    mean_attn = _mean_attention(attn_weights.detach().cpu())
    edge_index_cpu = edge_index.detach().cpu()

    _plot_aa_to_protein_msa(
        context,
        edge_index_cpu,
        mean_attn,
        aligned_seqs,
        f"AA-Protein Attention (Layer {layer_idx}, MSA-aligned): {context.seed_label}",
        "Normalized Attention Weight",
        f"aa_attention_layer{layer_idx}_msa",
        go_term,
        center_on_uniform=True,
    )

    _plot_msa_alignment_violin(
        context,
        edge_index_cpu,
        mean_attn,
        aligned_seqs,
        f"Aligned vs Unaligned Attention (Layer {layer_idx}): {context.seed_label}",
        f"aa_attention_layer{layer_idx}_msa_violin",
        go_term,
    )


def plot_attn_seed_vs_neighbor_scatter(
    path: str,
    layer_attention,
    dataset,
    batch,
    layer_idx: int,
    go_term: Optional[str] = None,
    aligned_seqs: Optional[list[str]] = None,
):
    """Plot scatter of seed vs neighbor attention for aligned residues."""
    if aligned_seqs is None:
        aligned_seqs = perform_msa_from_batch(batch)
    if aligned_seqs is None:
        return

    context = build_plot_context(path, dataset, batch)
    key = ("aa", "belongs_to", "protein")

    if layer_attention is None or key not in layer_attention:
        return

    edge_index, attn_weights = layer_attention[key]

    _plot_attn_seed_vs_neighbor_scatter(
        context,
        edge_index.detach().cpu(),
        _mean_attention(attn_weights.detach().cpu()),
        aligned_seqs,
        f"Seed vs Neighbor Normalized Attention (Layer {layer_idx}): {context.seed_label}",
        f"attn_seed_vs_neighbor_layer{layer_idx}",
        go_term,
    )


def plot_attn_stringdb_vs_aligned_scatter(
    path: str,
    layer_attention,
    dataset,
    batch,
    layer_idx: int,
    go_term: Optional[str] = None,
):
    """Plot scatter of StringDB vs Aligned attention for common neighbors."""
    context = build_plot_context(path, dataset, batch)

    # Identify keys
    stringdb_key = None
    aligned_key = None

    if layer_attention is None:
        return

    for key in layer_attention.keys():
        if not isinstance(key, tuple) or len(key) != 3:
            continue
        src, rel, dst = key
        if src == "protein" and dst == "protein":
            if "stringdb" in rel:
                stringdb_key = key
            elif "aligned" in rel:
                aligned_key = key

    if not stringdb_key or not aligned_key:
        return

    # Helper to extract weights for edges pointing to seed (index 0)
    def get_weights(key):
        edge_index, weights = layer_attention[key]
        weights = _mean_attention(weights.detach().cpu())
        edge_index = edge_index.detach().cpu()

        # Assuming seed is at index 0
        mask = edge_index[1] == 0
        neighbors = edge_index[0][mask]
        w = weights[mask]
        return {int(n): float(v) for n, v in zip(neighbors, w)}

    w_stringdb = get_weights(stringdb_key)
    w_aligned = get_weights(aligned_key)

    common = sorted(list(set(w_stringdb.keys()) & set(w_aligned.keys())))
    if len(common) < 2:
        return

    x_vals = [w_stringdb[n] for n in common]
    y_vals = [w_aligned[n] for n in common]

    # Correlation
    if np.std(x_vals) > 1e-9 and np.std(y_vals) > 1e-9:
        corr = np.corrcoef(x_vals, y_vals)[0, 1]
    else:
        corr = 0.0

    plt.figure(figsize=(6, 6))
    plt.scatter(x_vals, y_vals, alpha=0.7, c="blue", edgecolors="k")

    # Add trendline
    if len(x_vals) > 1:
        try:
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(x_vals), max(x_vals), 100)
            plt.plot(x_range, p(x_range), "r--", alpha=0.5)
        except Exception:
            pass

    plt.xlabel(f"Attention ({stringdb_key[1]})")
    plt.ylabel(f"Attention ({aligned_key[1]})")
    plt.title(f"Layer {layer_idx} Attention Correlation\n{context.seed_label}")
    plt.legend([f"Pearson R = {corr:.3f}"], loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    _save_plot(context, f"attn_stringdb_vs_aligned_layer{layer_idx}", go_term=go_term)


def plot_systemic_explanation(
    path, hetero_explanation, dataset, title_suffix=None, go_term=None
):
    """Plot protein-protein explanation graph."""
    context = build_plot_context(path, dataset, hetero_explanation.batch)

    plotted = False
    for edge_type in hetero_explanation.edge_types:
        src, rel, dst = edge_type
        if src == "protein" and dst == "protein":
            if edge_type not in hetero_explanation:
                continue

            expl_data = hetero_explanation[edge_type]
            if (
                not hasattr(expl_data, "edge_index")
                or expl_data.edge_index.numel() == 0
            ):
                continue

            edge_index = expl_data.edge_index.detach().cpu()
            edge_mask = expl_data.edge_mask.detach().cpu().view(-1)

            title = f"Protein-Protein Explanation ({rel}): {context.seed_label}"
            if title_suffix:
                title += f" ({title_suffix})"

            _plot_protein_network(
                context,
                edge_index,
                edge_mask,
                title,
                "Edge importance",
                f"system_explanation_{rel}",
                go_term,
            )
            plotted = True

    if not plotted:
        logger.info("No systemic edges found to plot.")


def plot_systemic_attention(path, layer_attention, dataset, batch, layer_idx):
    """Plot protein-protein attention graph."""
    context = build_plot_context(path, dataset, batch)

    if layer_attention is None:
        return

    plotted = False
    for edge_type, (edge_index, attn_weights) in layer_attention.items():
        if not isinstance(edge_type, tuple) or len(edge_type) != 3:
            continue

        src, rel, dst = edge_type
        if src == "protein" and dst == "protein":
            edge_index = edge_index.detach().cpu()
            if edge_index.numel() == 0:
                continue

            attn_values = _mean_attention(attn_weights).detach().cpu()

            # Filter out self-loops for non-seed proteins (seed is at index 0 when batch_size = 1)
            src_idx, dst_idx = edge_index[0], edge_index[1]
            mask = ~((src_idx == dst_idx) & (src_idx != 0))
            edge_index = edge_index[:, mask]
            attn_values = attn_values[mask]

            # Baseline: uniform attention
            uniform_baseline = 1.0 / len(edge_index[1].unique())

            if edge_index.numel() == 0:
                continue

            title = f"Protein-Protein Attention (Layer {layer_idx}, {rel}): {context.seed_label}"
            _plot_protein_network(
                context,
                edge_index,
                attn_values,
                title,
                "Normalized Attention Weight",
                f"system_attention_layer{layer_idx}_{rel}",
                uniform_baseline=uniform_baseline,
            )
            plotted = True

    if not plotted:
        logger.info(f"No systemic attention found for layer {layer_idx}.")


def plot_protein_explanation(
    path: str,
    hetero_explanation,
    dataset,
    title_suffix: Optional[str] = None,
    go_term: Optional[str] = None,
):
    """Plot amino acid to protein explanation."""
    context = build_plot_context(path, dataset, hetero_explanation.batch)
    key = ("aa", "belongs_to", "protein")

    title = f"AA-Protein Explanation: {{protein}}"
    if title_suffix:
        title += f" ({title_suffix})"

    _plot_aa_to_protein_scatter(
        context,
        hetero_explanation[key]["edge_index"].detach().cpu(),
        hetero_explanation[key]["edge_mask"].detach().cpu(),
        title,
        "Edge Importance",
        "aa_explanation",
        go_term,
    )


def plot_protein_attention(
    path, layer_attention, dataset, batch, layer_idx, go_term=None
):
    """Plot amino acid to protein attention weights."""
    context = build_plot_context(path, dataset, batch)
    key = ("aa", "belongs_to", "protein")

    if layer_attention is None or key not in layer_attention:
        return

    edge_index, attn_weights = layer_attention[key]
    _plot_aa_to_protein_scatter(
        context,
        edge_index.detach().cpu(),
        _mean_attention(attn_weights.detach().cpu()),
        f"AA-Protein Attention (Layer {layer_idx}): {{protein}}",
        "Attention weight",
        f"aa_attention_layer{layer_idx}",
        go_term,
    )


def plot_merged_systemic_attention(
    path: str,
    attentions: list,
    dataset,
    batch,
    go_term: Optional[str] = None,
):
    """Plot merged protein-protein attention graph across all layers."""
    context = build_plot_context(path, dataset, batch)

    if not attentions:
        return

    merged_weights = {}  # (src, rel, dst) -> tensor of weights
    edge_indices = {}  # (src, rel, dst) -> edge_index

    for layer_attn in attentions:
        if layer_attn is None:
            continue

        for edge_type, (edge_index, attn_weights) in layer_attn.items():
            if not isinstance(edge_type, tuple) or len(edge_type) != 3:
                continue

            src, rel, dst = edge_type
            if src == "protein" and dst == "protein":
                edge_index = edge_index.detach().cpu()
                weights = _mean_attention(attn_weights).detach().cpu()

                if edge_type not in merged_weights:
                    merged_weights[edge_type] = weights
                    edge_indices[edge_type] = edge_index
                else:
                    if edge_index.shape == edge_indices[edge_type].shape:
                        merged_weights[edge_type] += weights
    # Average weights
    for edge_type in merged_weights.keys():
        merged_weights[edge_type] /= len(attentions)

    plotted = False
    for edge_type, _ in merged_weights.items():
        src, rel, dst = edge_type
        edge_index = edge_indices[edge_type]

        # Filter out self-loops for non-seed proteins
        src_idx, dst_idx = edge_index[0], edge_index[1]
        mask = ~((src_idx == dst_idx) & (src_idx != 0))
        edge_index = edge_index[:, mask]

        uniform_baseline = (
            1.0 / len(edge_index[1].unique())
            if len(edge_index[1].unique()) > 0
            else None
        )

        if edge_index.numel() == 0:
            continue

        title = f"Protein-Protein Attention (Merged, {rel}): {context.seed_label}"
        _plot_protein_network(
            context,
            edge_index,
            merged_weights[edge_type][mask],
            title,
            "Avg Normalized Attention",
            f"system_attention_merged_{rel}",
            go_term,
            uniform_baseline=uniform_baseline,
        )
        plotted = True

    if not plotted:
        logger.info("No systemic attention found to merge.")


def plot_merged_protein_attention(
    path: str,
    attentions: list,
    dataset,
    batch,
    go_term: Optional[str] = None,
    aligned_seqs: Optional[list[str]] = None,
):
    """Plot merged amino acid to protein attention weights across all layers."""
    context = build_plot_context(path, dataset, batch)
    key = ("aa", "belongs_to", "protein")

    if not attentions:
        return

    total_weights = None
    edge_index_ref = None
    count = 0

    for layer_attn in attentions:
        if layer_attn is None or key not in layer_attn:
            continue

        edge_index, attn_weights = layer_attn[key]
        weights = _mean_attention(attn_weights.detach().cpu())
        edge_index = edge_index.detach().cpu()

        if total_weights is None:
            total_weights = weights
            edge_index_ref = edge_index
            count = 1
        else:
            if weights.shape == total_weights.shape:
                total_weights += weights
                count += 1

    if total_weights is None:
        return

    avg_weights = total_weights / count

    # Scatter plot
    _plot_aa_to_protein_scatter(
        context,
        edge_index_ref,
        avg_weights,
        f"Merged AA-Protein Attention: {{protein}}",
        "Avg Attention weight",
        "aa_attention_merged",
        go_term,
    )

    # MSA plot
    if aligned_seqs is None:
        aligned_seqs = perform_msa_from_batch(batch)
    if aligned_seqs:
        _plot_aa_to_protein_msa(
            context,
            edge_index_ref,
            avg_weights,
            aligned_seqs,
            f"Merged AA-Protein Attention (MSA-aligned): {context.seed_label}",
            "Avg Normalized Attention",
            "aa_attention_merged_msa",
            go_term,
            center_on_uniform=True,
        )


def analyze_attention_captum_correlation(
    output_dir: str,
    dataset,
    batch,
    attentions,
    hetero_explanation,
    *,
    layer_to_plot: int = 2,
) -> None:
    """Analyze correlation between attention and Captum scores."""
    context = build_plot_context(output_dir, dataset, batch)
    key = ("aa", "belongs_to", "protein")

    captum_edge_index = hetero_explanation[key]["edge_index"].detach().cpu()
    captum_scores = hetero_explanation[key]["edge_mask"].detach().cpu()

    # Analyze only seed protein edges
    seed_mask = captum_edge_index[1] == 0
    captum_edge_index = captum_edge_index[:, seed_mask]
    captum_scores = captum_scores[seed_mask]

    edge_to_captum = {
        (int(captum_edge_index[0, i]), int(captum_edge_index[1, i])): float(
            captum_scores[i].item()
        )
        for i in range(captum_edge_index.size(1))
    }

    scatter_data = None
    for layer_idx, layer_attention in enumerate(attentions, start=1):
        if layer_attention is None or key not in layer_attention:
            continue

        edge_index, attn_weights = layer_attention[key]
        edge_index = edge_index.detach().cpu()[:, seed_mask]
        attn_vals = _mean_attention(attn_weights.detach().cpu())[seed_mask]

        # Match edges
        shared_attn, shared_captum = [], []
        for i in range(edge_index.size(1)):
            edge = (int(edge_index[0, i]), int(edge_index[1, i]))
            if edge in edge_to_captum:
                shared_attn.append(float(attn_vals[i].item()))
                shared_captum.append(edge_to_captum[edge])

        if len(shared_attn) < 2:
            continue

        attn_arr = np.asarray(shared_attn, dtype=np.float32)
        captum_arr = np.asarray(shared_captum, dtype=np.float32)

        # Compute correlation
        if (
            np.std(attn_arr) < 1e-12
            or np.std(captum_arr) < 1e-12
            or np.isnan(attn_arr).any()
            or np.isnan(captum_arr).any()
        ):
            corr_val = float("nan")
        else:
            corr_val = float(np.corrcoef(attn_arr, captum_arr)[0, 1])

        logger.info(
            f"Pearson correlation (layer {layer_idx} vs Captum): {corr_val:.4f}"
        )

        if layer_idx == layer_to_plot:
            scatter_data = (attn_arr, captum_arr)

    if scatter_data is None:
        return

    # Plot scatter
    attn_arr, captum_arr = scatter_data
    plot_path = os.path.join(
        context.seed_dir,
        f"{context.seed_label}_attn_layer{layer_to_plot}_captum_scatter.png",
    )

    plt.figure(figsize=(6, 5))
    plt.scatter(attn_arr, captum_arr, alpha=0.6)
    m, b = np.polyfit(attn_arr, captum_arr, 1)
    plt.plot(attn_arr, m * attn_arr + b, color="red")
    plt.xlabel(f"Attention Layer {layer_to_plot}")
    plt.ylabel("Captum Score")
    plt.title(f"Attention vs Captum (Layer {layer_to_plot}) – {context.seed_label}")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
