"""Visualization utilities for model interpretability and analysis.

This package provides plotting, rendering, and structure management
for the attribution / explainability pipeline.  All public symbols are
re-exported here so that existing ``from src.utils.visualize import …``
imports continue to work unchanged.
"""

# -- Context & helpers -------------------------------------------------------
from src.utils.visualize.context import (
    ProteinPlotContext,
    build_plot_context,
    save_plot,
    ensure_structure,
)

# -- PyMOL rendering --------------------------------------------------------
from src.utils.visualize.rendering import (
    adjust_colormap,
    render_scene,
    export_layer_attention_3d,
    export_captum_3d,
    export_captum_3d_rank,
)

# -- Network-level plots ----------------------------------------------------
from src.utils.visualize.network import (
    plot_systemic_attention,
    plot_systemic_attention_rank,
    plot_systemic_explanation,
    plot_systemic_explanation_rank,
)

# -- Residue-level plots ----------------------------------------------------
from src.utils.visualize.residue import (
    build_protein_score_map,
    plot_protein_attention,
    plot_protein_attention_rank,
    plot_protein_explanation,
    plot_protein_explanation_rank,
)

# -- Comparison / correlation plots -----------------------------------------
from src.utils.visualize.comparison import (
    analyze_attention_captum_correlation,
    plot_attn_stringdb_vs_aligned_scatter,
    plot_attn_stringdb_vs_aligned_scatter_rank,
    plot_edge_attr_vs_attention_scatter,
    plot_edge_attr_vs_attention_scatter_rank,
    plot_shared_name_attention_boxplot,
    plot_shared_name_attention_boxplot_rank,
)

__all__ = [
    # context
    "ProteinPlotContext",
    "build_plot_context",
    "save_plot",
    "ensure_structure",
    # rendering
    "adjust_colormap",
    "render_scene",
    "export_layer_attention_3d",
    "export_captum_3d",
    "export_captum_3d_rank",
    # network
    "plot_systemic_attention",
    "plot_systemic_attention_rank",
    "plot_systemic_explanation",
    "plot_systemic_explanation_rank",
    # residue
    "build_protein_score_map",
    "plot_protein_attention",
    "plot_protein_attention_rank",
    "plot_protein_explanation",
    "plot_protein_explanation_rank",
    # comparison
    "analyze_attention_captum_correlation",
    "plot_attn_stringdb_vs_aligned_scatter",
    "plot_attn_stringdb_vs_aligned_scatter_rank",
    "plot_edge_attr_vs_attention_scatter",
    "plot_edge_attr_vs_attention_scatter_rank",
    "plot_shared_name_attention_boxplot",
    "plot_shared_name_attention_boxplot_rank",
]
