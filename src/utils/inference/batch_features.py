"""Batch feature assembly for inference.

Replicates the feature loading logic from SwissProtDataset.get_batch_features()
(dataloading.py) but adapted for inference:
- All query protein GO features are zeroed (the model predicts, not memorizes).
- Neighbor proteins get their GO annotations loaded from annotation files.
- Edge attributes use precomputed normalization from edge_normalization.json.
- No train/val/test mask logic (all masks are zero).
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch_geometric.transforms as T

from src.utils.constants import CONTACT_CUTOFF

if TYPE_CHECKING:
    from src.utils.inference.inference_dataset import InferenceDataset

logger = logging.getLogger(__name__)


def get_batch_features(
    dataset: "InferenceDataset",
    batch,
    return_sequences: bool = False,
):
    """Load individual protein features and AA data for the sampled batch.

    This mirrors SwissProtDataset.get_batch_features() exactly, but uses the
    InferenceDataset's attributes and zeroes all query protein GO features.

    Args:
        dataset: InferenceDataset instance.
        batch: A batch from NeighborLoader.
        return_sequences: If True, attach sequences to the batch.

    Returns:
        The batch with all features populated.
    """
    with torch.no_grad():
        raw_protein_ids = [
            dataset.idx_to_protein[idx.item()] for idx in batch["protein"].n_id
        ]
        sampled_protein_ids = raw_protein_ids
        if dataset.uses_entryid:
            sampled_protein_ids = [
                dataset.pid_mapping.get(pid, pid) for pid in sampled_protein_ids
            ]

        use_edge_attrs = dataset.config["model"]["edge_attrs"]
        use_contact = ["aa", "close_to", "aa"] in dataset.config["model"]["edge_types"]
        mean_pool_aa = dataset.config["model"].get("mean_pool_aa", False)
        if mean_pool_aa:
            use_contact = False

        if return_sequences:
            sampled_sequences = []
        batch_interpro_features = []
        batch_go_features = []
        batch_aa_features = []
        aa_to_protein_edges = []
        contact_edges = [] if use_contact else None
        contact_attrs = [] if use_edge_attrs else None
        protein_sizes = []

        aa_offset = 0

        for local_idx, protein_id in enumerate(sampled_protein_ids):
            protein_graph = dataset.load_protein_graph(protein_id)
            raw_pid = raw_protein_ids[local_idx]

            if protein_graph is None:
                logger.warning(
                    "Using empty features for missing protein %s", protein_id
                )
                interpro_feat = torch.zeros(
                    dataset.ipr_vocab_size, dtype=torch.float32
                )
                go_feat = torch.zeros(dataset.go_vocab_size, dtype=torch.float32)
                aa_feat = torch.zeros(200, 1280, dtype=torch.float32)
                if mean_pool_aa:
                    aa_feat = aa_feat.mean(dim=0, keepdim=True)
                if use_contact:
                    local_contact_edge_index = torch.empty((2, 0), dtype=torch.long)
                    local_contact_edge_attr = (
                        torch.empty((0,), dtype=torch.float32)
                        if use_edge_attrs
                        else None
                    )
                if return_sequences:
                    sampled_sequences.append("")
            else:
                # Load features from the .pt graph
                if return_sequences:
                    sampled_sequences.append(protein_graph["protein"].sequence)
                interpro_feat = protein_graph["protein"].interpro.squeeze(0)
                aa_feat = protein_graph["aa"].x
                if mean_pool_aa:
                    aa_feat = aa_feat.mean(dim=0, keepdim=True)

                # GO features: use external annotations if available, with robust ID mapping fallback
                go_terms = None
                if dataset.external_annotations is not None:
                    if raw_pid in dataset.external_annotations:
                        go_terms = dataset.external_annotations[raw_pid]
                    elif dataset.uses_entryid and dataset.pid_mapping and raw_pid in dataset.pid_mapping and dataset.pid_mapping[raw_pid] in dataset.external_annotations:
                        go_terms = dataset.external_annotations[dataset.pid_mapping[raw_pid]]
                    elif not dataset.uses_entryid and dataset.rev_pid_mapping and raw_pid in dataset.rev_pid_mapping and dataset.rev_pid_mapping[raw_pid] in dataset.external_annotations:
                        go_terms = dataset.external_annotations[dataset.rev_pid_mapping[raw_pid]]

                if go_terms is not None:
                    go_feat = dataset._terms_to_onehot(go_terms)
                elif protein_graph is not None and f"go_terms_{dataset.subontology}" in protein_graph["protein"]:
                    go_feat = dataset.convert_go_terms_to_onehot(
                        protein_graph["protein"][f"go_terms_{dataset.subontology}"]
                    )
                else:
                    go_feat = torch.zeros(
                        dataset.go_vocab_size, dtype=torch.float32
                    )

                if use_contact:
                    if ("aa", "close_to", "aa") in protein_graph.edge_types:
                        contact_data = protein_graph["aa", "close_to", "aa"]
                        local_contact_edge_index = contact_data.edge_index
                        if use_edge_attrs:
                            local_contact_edge_attr = contact_data.edge_attr
                            if local_contact_edge_attr is None:
                                local_contact_edge_attr = torch.empty(
                                    (0,), dtype=torch.float32
                                )
                    else:
                        local_contact_edge_index = torch.empty(
                            (2, 0), dtype=torch.long
                        )
                        local_contact_edge_attr = (
                            torch.empty((0,), dtype=torch.float32)
                            if use_edge_attrs
                            else None
                        )

            batch_interpro_features.append(interpro_feat)
            batch_go_features.append(go_feat)
            batch_aa_features.append(aa_feat)
            protein_sizes.append(aa_feat.shape[0])

            # Create AA edges
            num_aas = aa_feat.shape[0]
            aa_indices = torch.arange(aa_offset, aa_offset + num_aas)
            protein_indices = torch.full((num_aas,), local_idx, dtype=torch.long)
            aa_to_protein_edges.append(torch.stack([aa_indices, protein_indices]))
            if use_contact:
                contact_edges.append(local_contact_edge_index + aa_offset)
            aa_offset += num_aas

            if use_edge_attrs and use_contact:
                contact_attrs.append(local_contact_edge_attr)

        # ---- Assemble batch tensors ----

        batch["protein"].interpro = torch.stack(batch_interpro_features)

        # Labels: not used during inference, but set for compatibility
        batch["protein"].y = torch.stack(batch_go_features)[
            : batch["protein"].batch_size
        ].clone()

        # GO features: zero for all seed (query) proteins
        batch["protein"].go = torch.stack(batch_go_features)
        batch["protein"].go[: batch["protein"].batch_size] = 0.0

        # For inference, we don't need to mask val/test — all masks are zero.
        # But we also don't unmask anything. Neighbors keep their GO features.

        # Set protein node features based on config
        if (
            dataset.config["model"]["interpro"]
            and dataset.config["model"]["go_neighbors"]
        ):
            batch["protein"].x = torch.cat(
                [batch["protein"].interpro, batch["protein"].go], dim=1
            )
        elif (
            dataset.config["model"]["interpro"]
            and not dataset.config["model"]["go_neighbors"]
        ):
            batch["protein"].x = batch["protein"].interpro
        elif (
            not dataset.config["model"]["interpro"]
            and dataset.config["model"]["go_neighbors"]
        ):
            batch["protein"].x = batch["protein"].go
        else:
            batch["protein"].x = torch.zeros(
                batch["protein"].num_nodes,
                dataset.ipr_vocab_size + dataset.go_vocab_size,
            ).float()

        # AA nodes
        batch["aa"].x = torch.cat(batch_aa_features, dim=0).float()
        batch["aa"].num_nodes = batch["aa"].x.shape[0]

        # AA → protein edges
        batch["aa", "belongs_to", "protein"].edge_index = torch.cat(
            aa_to_protein_edges, dim=1
        )

        if use_contact:
            batch["aa", "close_to", "aa"].edge_index = (
                torch.cat(contact_edges, dim=1)
                if contact_edges
                else torch.empty((2, 0), dtype=torch.long)
            )

        # Contact edge attribute normalization (same as training):
        # edge_attr is stored as sqrt(distance). Transform: attr^2 / CONTACT_CUTOFF
        if use_edge_attrs and use_contact:
            batch["aa", "close_to", "aa"].edge_attr = (
                torch.cat(contact_attrs, dim=0)
                if contact_attrs
                else torch.empty((0,), dtype=torch.float32)
            ).unsqueeze(1) ** 2 / CONTACT_CUTOFF

        # Store metadata
        batch["protein"].protein_ids = sampled_protein_ids
        batch["protein"].protein_sizes = protein_sizes
        if return_sequences:
            batch["protein"].sequences = sampled_sequences

        # Remove NeighborLoader metadata that confuses PyG transforms when num_edges == len(num_sampled_edges)
        for store in batch.edge_stores:
            if "num_sampled_edges" in store:
                del store["num_sampled_edges"]

        # Apply graph transforms
        try:
            batch = dataset.transform(batch)
        except Exception as e:
            logger.error("Error applying graph transformations: %s", e)

        return batch


def make_inference_batch_transform(dataset: "InferenceDataset", return_sequences: bool = False):
    """Create a batch transform function for the NeighborLoader.

    Args:
        dataset: InferenceDataset instance.
        return_sequences: If True, attach sequences to batches.

    Returns:
        Callable that transforms raw batches into feature-rich batches.
    """

    def batch_transform(batch):
        batch["mode"] = "predict"
        batch = get_batch_features(dataset, batch, return_sequences=return_sequences)
        return batch

    return batch_transform
