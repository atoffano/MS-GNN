"""Cross-metric comparison and correlation plots."""

import logging
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import mannwhitneyu

from src.utils.visualize.context import ProteinPlotContext, build_plot_context, save_plot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# StringDB vs Aligned attention scatter
# ---------------------------------------------------------------------------


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

    if layer_attention is None:
        return

    stringdb_key = aligned_key = None
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

    def _get_seed_weights(key):
        edge_index, weights = layer_attention[key]
        weights = weights.mean(dim=-1).detach().cpu()
        edge_index = edge_index.detach().cpu()
        mask = edge_index[1] == 0
        neighbors = edge_index[0][mask]
        w = weights[mask]
        return {int(n): float(v) for n, v in zip(neighbors, w)}

    w_stringdb = _get_seed_weights(stringdb_key)
    w_aligned = _get_seed_weights(aligned_key)

    common = sorted(set(w_stringdb.keys()) & set(w_aligned.keys()))
    if len(common) < 2:
        return

    x_vals = [w_stringdb[n] for n in common]
    y_vals = [w_aligned[n] for n in common]

    if np.std(x_vals) > 1e-9 and np.std(y_vals) > 1e-9:
        corr = np.corrcoef(x_vals, y_vals)[0, 1]
    else:
        corr = 0.0

    plt.figure(figsize=(6, 6))
    plt.scatter(x_vals, y_vals, alpha=0.7, c="blue", edgecolors="k")

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

    save_plot(context, f"scatter_stringdb_vs_aligned_L{layer_idx}", go_term=go_term)


# ---------------------------------------------------------------------------
# Attention vs Captum correlation
# ---------------------------------------------------------------------------


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
        edge_index = edge_index.detach().cpu()
        attn_vals = attn_weights.mean(dim=-1).detach().cpu()

        # Compute seed mask from this layer's own edge index
        layer_seed_mask = edge_index[1] == 0
        edge_index = edge_index[:, layer_seed_mask]
        attn_vals = attn_vals[layer_seed_mask]

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

        if (
            np.std(attn_arr) < 1e-12
            or np.std(captum_arr) < 1e-12
            or np.isnan(attn_arr).any()
            or np.isnan(captum_arr).any()
        ):
            corr_val = float("nan")
        else:
            corr_val = float(np.corrcoef(attn_arr, captum_arr)[0, 1])

        if layer_idx == layer_to_plot:
            scatter_data = (attn_arr, captum_arr, corr_val)

    if scatter_data is None:
        return

    attn_arr, captum_arr, corr_val = scatter_data
    plot_path = os.path.join(
        context.seed_dir,
        f"{context.seed_label}_scatter_captum_vs_attention_L{layer_to_plot}.png",
    )

    plt.figure(figsize=(6, 5))
    plt.scatter(attn_arr, captum_arr, alpha=0.6, label=f"Pearson R = {corr_val:.3f}")
    m, b = np.polyfit(attn_arr, captum_arr, 1)
    plt.plot(attn_arr, m * attn_arr + b, color="red")
    plt.xlabel(f"Attention Layer {layer_to_plot}")
    plt.ylabel("Captum Score")
    plt.title(f"Attention vs Captum (Layer {layer_to_plot}) – {context.seed_label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


# ---------------------------------------------------------------------------
# NEW: Shared protein name attention boxplot
# ---------------------------------------------------------------------------


def plot_shared_name_attention_boxplot(
    path: str,
    layer_attention,
    dataset,
    batch,
    layer_idx: int,
    go_term: Optional[str] = None,
):
    """Boxplot comparing alignment attention to neighbors with shared vs different protein names.

    Protein names are extracted by splitting EntryIDs on '_' and taking the prefix
    (e.g. 'DIC_RAT' -> 'DIC'). Only works for datasets using EntryID format.
    Self-loops are excluded. Only alignment edges are used.

    A Mann-Whitney U test is used to compare the two distributions.
    """
    if not dataset.uses_entryid:
        logger.info("Shared-name boxplot skipped: dataset does not use EntryIDs.")
        return

    context = build_plot_context(path, dataset, batch)
    aligned_key = ("protein", "aligned_with", "protein")

    if layer_attention is None or aligned_key not in layer_attention:
        return

    edge_index, attn_weights = layer_attention[aligned_key]
    attn_weights = attn_weights.mean(dim=-1).detach().cpu()
    edge_index = edge_index.detach().cpu()

    # Edges pointing to seed (index 0), excluding self-loops
    mask = (edge_index[1] == 0) & (edge_index[0] != 0)
    src_indices = edge_index[0][mask].tolist()
    weights = attn_weights[mask].tolist()

    if not weights:
        logger.info("No alignment edges to seed; skipping shared-name boxplot.")
        return

    # Extract protein name prefix (everything before last '_')
    seed_name = context.seed_label.rsplit("_", 1)[0]

    shared, different = [], []
    for src_idx, w in zip(src_indices, weights):
        neighbor_label = context.labels.get(src_idx, "")
        if "_" not in neighbor_label:
            continue
        neighbor_name = neighbor_label.rsplit("_", 1)[0]
        if neighbor_name == seed_name:
            shared.append(w)
        else:
            different.append(w)

    if not shared and not different:
        logger.info("No categorizable neighbors; skipping shared-name boxplot.")
        return

    # Statistical test
    p_val = None
    if len(shared) >= 1 and len(different) >= 1:
        _, p_val = mannwhitneyu(shared, different, alternative="two-sided")

    # Build dataframe
    records = [{"Attention": w, "Group": "Shared name"} for w in shared]
    records += [{"Attention": w, "Group": "Different name"} for w in different]
    df = pd.DataFrame(records)

    plt.figure(figsize=(6, 5))
    sns.boxplot(data=df, x="Group", y="Attention", hue="Group", palette="Set2", legend=False)
    sns.stripplot(data=df, x="Group", y="Attention", color="black", alpha=0.4, size=4)

    title = (
        f"Alignment Attention by Protein Name (L{layer_idx})\n{context.seed_label}"
    )
    if p_val is not None:
        title += f"\nMann-Whitney U  p = {p_val:.4g}"
    plt.title(title)
    plt.ylabel("Attention weight")
    plt.tight_layout()

    save_plot(context, f"boxplot_shared_name_attention_L{layer_idx}", go_term=go_term)


# ---------------------------------------------------------------------------
# NEW: Edge attribute vs attention scatter
# ---------------------------------------------------------------------------


def plot_edge_attr_vs_attention_scatter(
    path: str,
    layer_attention,
    dataset,
    batch,
    layer_idx: int,
    go_term: Optional[str] = None,
):
    """Scatter of input edge attribute value vs attention score per protein-protein channel.

    One plot is generated for each protein-protein edge type that has both
    edge attributes and attention weights (typically aligned_with and stringdb).
    """
    if not dataset.config["model"]["edge_attrs"]:
        logger.info("Edge attr vs attention scatter skipped: edge_attrs disabled.")
        return

    context = build_plot_context(path, dataset, batch)

    if layer_attention is None:
        return

    target_keys = [
        ("protein", "aligned_with", "protein"),
        ("protein", "stringdb", "protein"),
    ]

    for key in target_keys:
        if key not in layer_attention:
            continue

        edge_index_attn, attn_weights = layer_attention[key]
        edge_index_attn = edge_index_attn.detach().cpu()
        attn_mean = attn_weights.mean(dim=-1).detach().cpu()

        # Get edge attributes from the batch
        if key not in batch.edge_index_dict:
            continue
        batch_store = batch[key]
        if not hasattr(batch_store, "edge_attr") or batch_store.edge_attr is None:
            continue

        batch_edge_index = batch_store.edge_index.detach().cpu()
        batch_edge_attr = batch_store.edge_attr.detach().cpu().view(-1)

        # Build lookup: (src, dst) -> attr value
        edge_to_attr = {}
        for i in range(batch_edge_index.size(1)):
            s, d = int(batch_edge_index[0, i]), int(batch_edge_index[1, i])
            edge_to_attr[(s, d)] = float(batch_edge_attr[i])

        # Match attention edges to batch edges
        x_vals, y_vals = [], []
        for i in range(edge_index_attn.size(1)):
            s, d = int(edge_index_attn[0, i]), int(edge_index_attn[1, i])
            if (s, d) in edge_to_attr:
                x_vals.append(edge_to_attr[(s, d)])
                y_vals.append(float(attn_mean[i]))

        if len(x_vals) < 2:
            continue

        # Correlation
        x_arr = np.asarray(x_vals)
        y_arr = np.asarray(y_vals)
        if np.std(x_arr) > 1e-9 and np.std(y_arr) > 1e-9:
            corr = np.corrcoef(x_arr, y_arr)[0, 1]
        else:
            corr = 0.0

        rel = key[1]
        plt.figure(figsize=(6, 5))
        plt.scatter(x_vals, y_vals, alpha=0.5, s=15, edgecolors="k", linewidths=0.3)

        # Trendline
        try:
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(x_vals), max(x_vals), 100)
            plt.plot(x_range, p(x_range), "r--", alpha=0.6)
        except Exception:
            pass

        plt.xlabel("Edge Attribute (normalized)")
        plt.ylabel("Attention Weight")
        plt.title(
            f"Edge Attr vs Attention ({rel}, L{layer_idx})\n"
            f"{context.seed_label}  (R = {corr:.3f})"
        )
        plt.grid(True, alpha=0.2)
        plt.tight_layout()

        save_plot(
            context,
            f"scatter_edge_attr_vs_attention_L{layer_idx}_{rel}",
            go_term=go_term,
        )
