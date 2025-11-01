"""PyTorch Geometric data loading for heterogeneous protein graphs.

This module provides the SwissProtDataset class and data loaders for training
the protein function prediction model. It handles:
- Loading preprocessed protein graphs from disk
- Managing train/validation/test splits based on GO annotations
- Creating neighborhood sampling loaders for efficient batch processing
- On-demand feature loading for proteins
- Handling multiple GO subontologies (MFO, BPO, CCO)

The dataset maintains a static protein-protein graph structure while loading
individual protein features dynamically to manage memory efficiently.
"""

import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T
import pandas as pd
import pickle
from pathlib import Path
import logging
from src.utils.helpers import (
    timeit,
    MemoryTracker,
    track_memory,
    log_worker_checkpoint,
    worker_init_fn,
)

logger = logging.getLogger(__name__)


class SwissProtDataset:
    """Dataset with memory optimized data structures for heterogeneous protein graphs."""

    def __init__(self, config):
        self.config = config
        self.graphs_dir = Path(config["data"]["protein_graphs"])

        # Load InterPro and GO vocab as before (unchanged)
        with open(f"{self.graphs_dir}/interpro_vocab.pkl", "rb") as f:
            interpro_info = pickle.load(f)
        self.ipr_vocab_size = interpro_info["vocab_size"]

        with open(f"{self.graphs_dir}/go_vocab.pkl", "rb") as f:
            go_info = pickle.load(f)
        self.go_vocab_info = go_info
        self.go_vocab_sizes = {
            onto: info["vocab_size"] for onto, info in go_info.items()
        }
        self.subontology = (
            config["data"]["subontology"][0] if config["data"]["subontology"] else "BPO"
        )
        self.go_vocab_size = self.go_vocab_sizes[self.subontology]

        # Load protein IDs as pandas Series for memory efficiency
        pid_map_df = pd.read_csv(
            "./data/swissprot/2024_01/swissprot_2024_01_annotations.tsv",
            sep="\t",
            usecols=["EntryID", "Entry Name"],
        )
        # Convert to pandas Series with categorical dtype for memory savings
        self.pid_mapping = pd.Series(
            pid_map_df["Entry Name"].values, index=pd.Categorical(pid_map_df["EntryID"])
        )
        self.rev_pid_mapping = (
            pd.Series(
                pid_map_df["EntryID"].values,
                index=pd.Categorical(pid_map_df["Entry Name"]),
            )
            .astype("category")
            .to_dict()
        )

        if config["data"]["dataset"] in ["D1"]:
            self.uses_entryid = True
        else:
            self.uses_entryid = False

        # Load the list of protein ids from disk and convert to pandas Series
        proteins_list = [
            f.stem
            for f in self.graphs_dir.glob("*.pt")
            if f.stem not in ["metadata", "interpro_vocab", "go_vocab"]
        ]
        if self.uses_entryid:
            # Map accession numbers back to entry IDs, use PID mapping Series with map
            proteins_list = self.rev_pid_mapping.values()
        self.proteins = pd.Series(proteins_list, dtype="category")

        # Protein to index mappings as pandas Series for efficient lookups
        self.protein_to_idx = pd.Series(
            data=range(len(self.proteins)), index=self.proteins
        ).astype("int32")
        self.idx_to_protein = pd.Series(
            data=self.proteins.values, index=range(len(self.proteins))
        )

        self._load_split_masks(config)

        self.data = self._create_protein_graph(config)

        logger.info(
            f"Created protein graph with {self.data['protein'].num_nodes} proteins"
        )
        logger.info(f"Train proteins: {self.train_mask.sum().item()}")
        logger.info(f"Val proteins: {self.val_mask.sum().item()}")
        logger.info(f"Test proteins: {self.test_mask.sum().item()}")

    def _load_split_masks(self, config):
        splits = {"train": set(), "val": set(), "test": set()}
        subontology = self.subontology

        if config["data"]["train_on_swissprot"]:
            exp_suffix = "exp_" if config["data"].get("exp_only", True) else ""
            train_path = f"./data/swissprot/2024_01/swissprot_2024_01_{subontology}_{exp_suffix}annotations.tsv"
        else:
            train_path = f"./data/{config['data']['dataset']}/{config['data']['dataset']}_{subontology}_train_annotations.tsv"

        if Path(train_path).exists():
            train_df = pd.read_csv(train_path, sep="\t")
            splits["train"] = set(train_df["EntryID"].tolist())
            if config["data"]["train_on_swissprot"] and self.uses_entryid:
                splits["train"] = set(
                    self.rev_pid_mapping.loc[list(splits["train"])].dropna().values
                )
            self.train_annots_df = train_df.set_index("EntryID")
            self.train_annots_df["term"] = self.train_annots_df["term"].str.split("; ")

        logger.info(f"Using {len(splits['train'])} train proteins from {train_path}")

        dataset_name = config["data"]["dataset"]
        for split_name in ["val", "test"]:
            split_path = f"./data/{dataset_name}/{dataset_name}_{subontology}_{split_name}_annotations.tsv"
            if Path(split_path).exists():
                split_df = pd.read_csv(split_path, sep="\t")
                splits[split_name] = set(split_df["EntryID"].tolist())

        swissprot_proteins = set(self.proteins.values)
        for split in splits:
            missing = list(splits[split] - swissprot_proteins)
            if missing:
                logger.warning(
                    f"{len(missing)} proteins missing in dataset for split '{split}'."
                )

        # Consider overlaps between train and val/test when full SwissProt is used
        if config["data"]["train_on_swissprot"]:
            for split_name in ["val", "test"]:
                splits["train"] = splits["train"] - splits[split_name]

        # Convert split protein IDs to indices using pandas map for fast conversion
        def to_indices(protein_ids):
            s = pd.Series(list(protein_ids), dtype="category")
            idxs = self.protein_to_idx.reindex(s).dropna().astype("int64").values
            return torch.tensor(idxs, dtype=torch.long)

        self.train_proteins = to_indices(splits["train"])
        self.val_proteins = to_indices(splits["val"])
        self.test_proteins = to_indices(splits["test"])

        num_proteins = len(self.proteins)

        # Create masks as torch bool tensors
        self.train_mask = torch.zeros(num_proteins, dtype=torch.bool)
        self.train_mask[self.train_proteins] = True
        self.train_mask.share_memory_()

        self.val_mask = torch.zeros(num_proteins, dtype=torch.bool)
        self.val_mask[self.val_proteins] = True
        self.val_mask.share_memory_()

        self.test_mask = torch.zeros(num_proteins, dtype=torch.bool)
        self.test_mask[self.test_proteins] = True
        self.test_mask.share_memory_()

        logger.info(
            f"Loaded splits - Train: {len(self.train_proteins)}, Val: {len(self.val_proteins)}, Test: {len(self.test_proteins)}"
        )

        self.pos_weights = self._compute_pos_weights().share_memory_()
        logger.info(f"Computed pos weights for {len(self.pos_weights)} GO terms")

    @timeit
    def _compute_pos_weights(self):
        """Compute positive weights for each GO term to handle class imbalance."""
        go_to_idx = self.go_vocab_info[self.subontology]["go_to_idx"]
        term_counts = torch.zeros(len(go_to_idx), dtype=torch.float32)

        all_term_indices = []
        for pid in self.train_proteins:
            if pid in self.train_annots_df.index:
                terms = self.train_annots_df.loc[pid, "term"]
                term_indices = [go_to_idx[term] for term in terms if term in go_to_idx]
                all_term_indices.extend(term_indices)
        if all_term_indices:
            term_indices_tensor = torch.tensor(all_term_indices, dtype=torch.long)
            term_counts = torch.bincount(
                term_indices_tensor, minlength=len(go_to_idx)
            ).float()

        pos_weights = len(self.train_proteins) / (term_counts + 1e-8)
        pos_weights = torch.clamp(pos_weights, min=1.0, max=100)

        return pos_weights

    # @track_memory("create_protein_graph")
    def _create_protein_graph(self, config):
        """Creates the high-level protein network."""

        @timeit
        def alignment_edge_data():
            alignment_path = (
                f"{config['run']['project_path']}{config['data']['alignment_path'][1:]}"
            )
            alignment_df = pd.read_csv(
                alignment_path,
                sep="\t",
                header=None,
                names=[
                    "protein1",
                    "protein2",
                    "identity",
                    "aln_len",
                    "mismatch",
                    "gapopen",
                    "qstart",
                    "qend",
                    "sstart",
                    "send",
                    "evalue",
                    "bitscore",
                ],
            )
            if self.uses_entryid:
                alignment_df["protein1"] = alignment_df["protein1"].map(
                    self.rev_pid_mapping
                )
                alignment_df["protein2"] = alignment_df["protein2"].map(
                    self.rev_pid_mapping
                )

            # Filter out proteins that don't have indices (not in protein_to_idx)
            initial_edges = len(alignment_df)
            alignment_df = alignment_df[
                alignment_df["protein1"].isin(self.proteins.values)
                & alignment_df["protein2"].isin(self.proteins.values)
            ]
            filtered_edges = initial_edges - len(alignment_df)
            if filtered_edges > 0:
                logger.info(
                    f"Filtered {filtered_edges} alignment edges with proteins not in dataset"
                )

            source_indices = alignment_df["protein1"].map(self.protein_to_idx).tolist()
            target_indices = alignment_df["protein2"].map(self.protein_to_idx).tolist()

            # Protein-protein edges
            edge_index = torch.tensor(
                [source_indices, target_indices], dtype=torch.long
            )
            if config["model"]["edge_attrs"]:
                features = alignment_df.drop(columns=["protein1", "protein2"])
                means = features.mean()
                stds = features.std()
                normalized_features = (features - means) / stds
                edge_attrs = torch.tensor(
                    normalized_features.values, dtype=torch.float32
                )
            else:
                edge_attrs = None

            return edge_index, edge_attrs

        @timeit
        def stringdb_edge_data():
            # Q8Z7H7 -> 220341.gene:17585230
            stringdb_mapping = (
                pd.read_csv(
                    f"./data/swissprot/2024_01/idmapping_swissprot_stringdb.tsv",
                    sep="\t",
                    usecols=["From", "To"],
                )
                .set_index("From")
                .to_dict()["To"]
            )
            rev_stringdb_mapping = {v: k for k, v in stringdb_mapping.items()}

            stringdb_path = config["data"]["stringdb_path"]
            stringdb_df = pd.read_csv(
                stringdb_path,
                sep="\t",
                header=None,
                names=[
                    "protein1",
                    "protein2",
                    "neighborhood",
                    "fusion",
                    "cooccurence",
                    "coexpression",
                    "experimental",
                    "database",
                    "textmining",
                    "combined_score",
                ],
            )
            stringdb_df["protein1"] = stringdb_df["protein1"].map(rev_stringdb_mapping)
            stringdb_df["protein2"] = stringdb_df["protein2"].map(rev_stringdb_mapping)
            stringdb_df = stringdb_df.dropna()

            if self.uses_entryid:
                stringdb_df["protein1"] = stringdb_df["protein1"].map(
                    self.rev_pid_mapping
                )
                stringdb_df["protein2"] = stringdb_df["protein2"].map(
                    self.rev_pid_mapping
                )

            # Filter out proteins that don't have indices (not in protein_to_idx)
            initial_edges = len(stringdb_df)
            stringdb_df = stringdb_df[
                stringdb_df["protein1"].isin(self.proteins.values)
                & stringdb_df["protein2"].isin(self.proteins.values)
            ]
            filtered_edges = initial_edges - len(stringdb_df)
            if filtered_edges > 0:
                logger.info(
                    f"Filtered {filtered_edges} STRING-DB edges with proteins not in dataset"
                )

            source_indices = stringdb_df["protein1"].map(self.protein_to_idx).tolist()
            target_indices = stringdb_df["protein2"].map(self.protein_to_idx).tolist()

            # Protein-protein edges
            edge_index = torch.tensor(
                [source_indices, target_indices], dtype=torch.long
            )
            if config["model"]["edge_attrs"]:
                features = stringdb_df.drop(columns=["protein1", "protein2"])
                means = features.mean()
                stds = features.std()
                normalized_features = (features - means) / stds
                edge_attrs = torch.tensor(
                    normalized_features.values, dtype=torch.float32
                )
            else:
                edge_attrs = None

            return edge_index, edge_attrs

        data = HeteroData()
        logger.info("Creating protein-protein graph edges...")
        if ["protein", "aligned_with", "protein"] in config["model"]["edge_types"]:
            alignment_edge_index, alignment_edge_attrs = alignment_edge_data()
            data["protein", "aligned_with", "protein"].edge_index = alignment_edge_index
            logger.info(f"Alignment edges: {alignment_edge_index.shape[1]}")
            if config["model"]["edge_attrs"]:
                data["protein", "aligned_with", "protein"].edge_attr = (
                    alignment_edge_attrs
                )

        if ["protein", "stringdb", "protein"] in config["model"]["edge_types"]:
            stringdb_edge_index, stringdb_edge_attrs = stringdb_edge_data()
            data["protein", "stringdb", "protein"].edge_index = stringdb_edge_index
            logger.info(f"STRINGdb edges: {stringdb_edge_index.shape[1]}")
            if config["model"]["edge_attrs"]:
                data["protein", "stringdb", "protein"].edge_attr = stringdb_edge_attrs

        # Protein nodes - aa features are added later when batching
        num_proteins = len(self.proteins)
        data["protein"].num_nodes = num_proteins

        return data

    def load_protein_graph(self, protein_id):
        """Load a single protein graph from disk."""
        try:
            return torch.load(
                f"{self.graphs_dir}/{protein_id}.pt",
                map_location="cpu",
                weights_only=False,
            )
        except FileNotFoundError:
            logger.warning(f"Graph not found for protein {protein_id}")
            return None

    def convert_go_terms_to_onehot(self, go_terms_dict):
        """Convert GO terms to one-hot encoding based on config."""
        go_to_idx = self.go_vocab_info[self.subontology]["go_to_idx"]
        onehot = torch.zeros(
            self.go_vocab_info[self.subontology]["vocab_size"], dtype=torch.float32
        )
        if self.config["data"]["exp_only"]:
            terms = go_terms_dict.get("experimental", [])
        else:
            terms = go_terms_dict.get("curated", [])

        for term in terms:
            if term in go_to_idx:
                onehot[go_to_idx[term]] = 1.0

        return onehot

    def get_batch_features(self, batch):
        """Load individual protein features and amino acid data for the sampled batch."""
        # tracker = MemoryTracker(log_prefix="[get_batch_features] ")
        # tracker.set_baseline()
        sampled_protein_ids = [
            self.idx_to_protein.loc[idx.item()] for idx in batch["protein"].n_id
        ]

        if self.uses_entryid:
            sampled_protein_ids = [
                self.pid_mapping.get(pid, pid) for pid in sampled_protein_ids
            ]

        # Preallocate tensors for better performance
        num_proteins = len(sampled_protein_ids)
        all_interpro_features = torch.zeros(
            num_proteins, self.ipr_vocab_size, dtype=torch.float32
        )
        all_go_features = torch.zeros(
            num_proteins, self.go_vocab_size, dtype=torch.float32
        )
        all_aa_features = []
        aa_to_protein_edges = []
        protein_sizes = torch.zeros(num_proteins, dtype=torch.long)
        current_aa_offset = 0

        for local_idx, protein_id in enumerate(sampled_protein_ids):
            protein_graph = self.load_protein_graph(protein_id)
            if protein_graph is None:
                logger.warning(f"Using empty features for missing protein {protein_id}")
                # interpro and go features already zeroed in preallocated tensors
                aa_feat = torch.zeros(200, 1280, dtype=torch.float32)  # Default 200 AAs
            else:
                # Load features
                all_interpro_features[local_idx] = protein_graph[
                    "protein"
                ].interpro.squeeze(0)
                aa_feat = protein_graph["aa"].x.clone()
                all_go_features[local_idx] = self.convert_go_terms_to_onehot(
                    protein_graph["protein"][f"go_terms_{self.subontology}"]
                )

            all_aa_features.append(aa_feat)
            protein_sizes[local_idx] = aa_feat.shape[0]

            # Create AA to protein edges
            num_aas = aa_feat.shape[0]
            aa_indices = torch.arange(current_aa_offset, current_aa_offset + num_aas)
            protein_indices = torch.full((num_aas,), local_idx, dtype=torch.long)
            aa_to_protein_edges.append(torch.stack([aa_indices, protein_indices]))

            current_aa_offset += num_aas

        # Update batch with function-related features
        batch["protein"].interpro = all_interpro_features
        batch["protein"].go = all_go_features
        batch["protein"].y = batch["protein"].go[: batch["protein"].batch_size].clone()
        batch["protein"].go[: batch["protein"].batch_size] = 0.0

        # Mask GO features for val/test proteins
        if batch["mode"] == "train":
            n_id = batch["protein"].n_id
            mask_val = self.val_mask[n_id]
            mask_test = self.test_mask[n_id]
            mask = mask_val | mask_test
            batch["protein"].go[mask] = 0.0

        # Set protein node features as concatenation of InterPro and GO one hots.
        batch["protein"].x = torch.cat(
            [
                all_interpro_features,
                all_go_features,
            ],
            dim=1,
        )

        # Add amino acid nodes and features
        batch["aa"].x = torch.cat(all_aa_features, dim=0).float()
        batch["aa"].num_nodes = batch["aa"].x.shape[0]

        # Add AA to protein edges
        batch["aa", "belongs_to", "protein"].edge_index = torch.cat(
            aa_to_protein_edges, dim=1
        )

        # Store metadata
        batch["protein"].protein_ids = sampled_protein_ids
        batch["protein"].protein_sizes = protein_sizes

        return batch


def make_batch_transform(dataset, mode):
    """Create a batch transformation function for the data loader.

    Args:
        dataset: SwissProtDataset instance
        mode: Dataset mode ('train', 'val', or 'test')

    Returns:
        Function that transforms batches by adding mode and loading features
    """

    def batch_transform(batch):
        """Transform batch by adding mode and loading protein features."""
        batch["mode"] = mode
        batch = dataset.get_batch_features(batch)
        return batch

    return batch_transform


def define_loaders(config, dataset):
    """Create NeighborLoader instances for train/val/test."""

    edge_types = [tuple(et) for et in config["model"]["edge_types"]]
    protein_edge_types = [
        et for et in edge_types if et[0] == "protein" and et[2] == "protein"
    ]
    num_neighbors = {et: [-1] for et in protein_edge_types}  # 1-hop sampling
    # num_neighbors[("protein", "stringdb", "protein")] = [60]  # Limit STRINGdb neighbors
    train_loader = NeighborLoader(
        dataset.data,
        num_neighbors=num_neighbors,
        batch_size=config["model"]["batch_size"],
        input_nodes=("protein", dataset.train_mask),
        transform=make_batch_transform(dataset, mode="train"),
        shuffle=True,
        num_workers=config["trainer"]["num_workers"],
        # persistent_workers=True if config["trainer"]["num_workers"] > 0 else False,
        pin_memory=True,
        drop_last=True,
        # worker_init_fn=(
        #     worker_init_fn if config["trainer"]["num_workers"] > 0 else None
        # ),
    )

    test_loader = NeighborLoader(
        dataset.data,
        num_neighbors=num_neighbors,
        batch_size=config["model"]["batch_size"],
        input_nodes=("protein", dataset.test_mask),
        transform=make_batch_transform(dataset, mode="predict"),
        shuffle=False,
        num_workers=config["trainer"]["num_workers"],
        # persistent_workers=True if config["trainer"]["num_workers"] > 0 else False,
        pin_memory=True,
        # worker_init_fn=(
        #     worker_init_fn if config["trainer"]["num_workers"] > 0 else None
        # ),
    )

    # Some datasets, like H30, do not have a validation set
    # This is (dirtily) handled by using the test set as val too.
    if config["data"]["dataset"] != "H30":
        val_loader = NeighborLoader(
            dataset.data,
            num_neighbors=num_neighbors,
            batch_size=config["model"]["batch_size"],
            input_nodes=("protein", dataset.val_mask),
            transform=make_batch_transform(dataset, mode="predict"),
            shuffle=False,
            num_workers=config["trainer"]["num_workers"],
            # persistent_workers=True if config["trainer"]["num_workers"] > 0 else False,
            # pin_memory=True,
            # worker_init_fn=(
            #     worker_init_fn if config["trainer"]["num_workers"] > 0 else None
            # ),
        )
        return train_loader, val_loader, test_loader
    else:
        return train_loader, test_loader, test_loader
