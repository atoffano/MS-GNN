#     # Explanation generation
#     explainer = Explainer(
#         model,  # It is assumed that model outputs a single tensor.
#         algorithm=CaptumExplainer("IntegratedGradients"),
#         explanation_type="model",
#         node_mask_type="attributes",
#         edge_mask_type="object",
#         model_config=dict(
#             mode="multiclass_classification",
#             task_level="node",
#             return_type="raw",  # Model returns probabilities.
#         ),
#     )

#     hetero_explanation = explainer(
#         batch.x_dict,
#         batch.edge_index_dict,
#         batch=batch,
#         target=None,
#         # index=None,
#         index=torch.arange(batch["protein"].batch_size),
#         # batch["protein"].go[: batch["protein"].batch_size]
#     )
#     logger.info(
#         f"Generated explanations in {hetero_explanation.available_explanations}"
#     )

# path = os.path.join(results_dir, "feature_importance.png")
# hetero_explanation.visualize_feature_importance(path, top_k=10)
# logger.info(f"Feature importance plot has been saved to '{path}'")
# wandb.log({"feature_importance_plot": wandb.Image(path)})

# # Visualize graph via NetworkX for specified edge types
# graph_path = os.path.join(results_dir, "graph_explanation")
# plots = visualize_graph_via_networkx(
#     hetero_explanation,
#     path=graph_path,  # Base path; function will append suffix for each edge type
#     cutoff_edge=0.00001,
#     edge_types_to_plot=[("aa", "aa2protein", "protein")],
# )
# # Log each saved plot to wandb
# for edge_type, fig in plots.items():
#     suffix = "_".join(edge_type)
#     plot_path = f"{graph_path}_{suffix}.png"
#     wandb.log({f"graph_explanation_plot_{suffix}": wandb.Image(plot_path)})

# # Plot AA edge histogram
# path = os.path.join(results_dir, "aa_edge_histogram.png")
# ret = plot_aa_edge_histogram(
#     hetero_explanation,
#     edge_type=("aa", "aa2protein", "protein"),
#     path=path,
# )
# wandb.log({"aa_edge_histogram_plot": wandb.Image(ret["fig"])})
