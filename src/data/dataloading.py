import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
import pandas as pd
import pickle
from pathlib import Path
import logging
from src.utils.helpers import timeit

logger = logging.getLogger(__name__)


class SwissProtDataset:
    """Dataset that maintains a static protein-protein graph and loads individual protein features on-demand."""

    def __init__(self, config, split="train"):
        self.config = config
        self.split = split
        self.graphs_dir = Path(config["data"]["protein_graphs"])

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

        # Get preprocessed protein graphs
        self.proteins = [
            f.stem
            for f in self.graphs_dir.glob("*.pt")
            if f.stem not in ["metadata", "interpro_vocab", "go_vocab"]
        ]

        # Load GO annotations to determine train/val/test splits
        self._load_split_masks(config)

        # Create the protein-protein heterograph
        self.data = self._create_protein_graph(config)

        logger.info(
            f"Created protein graph with {self.data['protein'].num_nodes} proteins"
        )
        logger.info(f"Train proteins: {self.train_mask.sum().item()}")
        logger.info(f"Val proteins: {self.val_mask.sum().item()}")
        logger.info(f"Test proteins: {self.test_mask.sum().item()}")

    def _load_split_masks(self, config):
        """Load train/val/test splits based on GO annotations."""
        splits = {"train": set(), "val": set(), "test": set()}
        subontology = self.subontology

        if config["data"].get("train_on_swissprot"):
            exp_suffix = "exp_" if config["data"].get("exp_only", True) else ""
            train_path = f"./data/swissprot/2024_01/swissprot_2024_01_{subontology}_{exp_suffix}annotations.tsv"
        else:
            # Stick to original dataset's train set
            train_path = f"./data/{config['data']['dataset']}/{config['data']['dataset']}_{subontology}_train_annotations.tsv"

        if Path(train_path).exists():
            train_df = pd.read_csv(train_path, sep="\t")
            splits["train"] = set(train_df["EntryID"].tolist())

        # Load val and test splits
        dataset_name = config["data"]["dataset"]
        for split_name in ["val", "test"]:
            split_path = f"./data/{dataset_name}/{dataset_name}_{subontology}_{split_name}_annotations.tsv"
            if Path(split_path).exists():
                split_df = pd.read_csv(split_path, sep="\t")
                splits[split_name] = set(split_df["EntryID"].tolist())

        # Filter by available proteins and create index mappings
        protein_set = set(self.proteins)
        for split in splits:
            splits[split] = splits[split].intersection(protein_set)

        self.protein_to_idx = {pid: i for i, pid in enumerate(self.proteins)}
        self.idx_to_protein = {v: k for k, v in self.protein_to_idx.items()}

        # Create split sets with indices and masks
        self.train_proteins = list(splits["train"])
        self.val_proteins = list(splits["val"])
        self.test_proteins = list(splits["test"])

        num_proteins = len(self.proteins)
        self.train_mask = torch.zeros(num_proteins, dtype=torch.bool)
        self.val_mask = torch.zeros(num_proteins, dtype=torch.bool)
        self.test_mask = torch.zeros(num_proteins, dtype=torch.bool)

        for pid in self.train_proteins:
            if pid in self.protein_to_idx:
                self.train_mask[self.protein_to_idx[pid]] = True
        for pid in self.val_proteins:
            if pid in self.protein_to_idx:
                self.val_mask[self.protein_to_idx[pid]] = True
        for pid in self.test_proteins:
            if pid in self.protein_to_idx:
                self.test_mask[self.protein_to_idx[pid]] = True

        logger.info(
            f"Loaded splits - Train: {len(self.train_proteins)}, Val: {len(self.val_proteins)}, Test: {len(self.test_proteins)}"
        )

    def _create_protein_graph(self, config):
        """Creates the high-level protein network."""
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

        # Convert to indices
        source_indices = alignment_df["protein1"].map(self.protein_to_idx).tolist()
        target_indices = alignment_df["protein2"].map(self.protein_to_idx).tolist()

        # Create protein-protein full graph
        data = HeteroData()

        # Protein nodes - features are added later when batching
        num_proteins = len(self.proteins)
        data["protein"].num_nodes = num_proteins

        # Protein-protein edges
        protein_edge_index = torch.tensor(
            [source_indices, target_indices], dtype=torch.long
        )
        if config["data"].get("edge_attrs"):
            edge_attrs = torch.tensor(
                alignment_df["bitscore"].tolist(), dtype=torch.float32
            )
            data["protein", "aligned_with", "protein"].edge_attr = edge_attrs
        data["protein", "aligned_with", "protein"].edge_index = protein_edge_index

        logger.info(f"Built protein-protein graph with {len(source_indices)} edges")
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
        # Get the appropriate GO vocabulary for this subontology
        go_vocab = self.go_vocab_info[self.subontology]
        go_to_idx = go_vocab["go_to_idx"]
        vocab_size = go_vocab["vocab_size"]

        onehot = torch.zeros(vocab_size, dtype=torch.float32)
        use_experimental_only = self.config["data"].get("exp_only", True)
        if use_experimental_only:
            terms = go_terms_dict.get("experimental", [])
        else:
            terms = go_terms_dict.get("curated", [])

        for term in terms:
            if term in go_to_idx:
                onehot[go_to_idx[term]] = 1.0

        return onehot

    @timeit
    def get_batch_features(self, batch):
        """Load individual protein features and amino acid data for the sampled batch."""
        protein_n_id = batch["protein"].n_id
        batch_size = batch["protein"].batch_size
        sampled_protein_ids = [self.idx_to_protein[idx.item()] for idx in protein_n_id]

        # Load individual protein graphs and extract features
        all_interpro_features = []
        all_go_features = []
        all_aa_features = []
        aa_to_protein_edges = []
        protein_sizes = []
        current_aa_offset = 0

        for local_idx, protein_id in enumerate(sampled_protein_ids):
            protein_graph = self.load_protein_graph(protein_id)

            if protein_graph is None:
                logger.warning(f"Using empty features for missing protein {protein_id}")
                interpro_feat = torch.zeros(self.ipr_vocab_size, dtype=torch.float32)
                go_feat = torch.zeros(self.go_vocab_size, dtype=torch.float32)
                aa_feat = torch.zeros(200, 1280, dtype=torch.float32)  # Default 200 AAs
            else:
                # Extract InterPro features
                interpro_feat = protein_graph["protein"].interpro.squeeze(0)

                # Extract and convert GO features
                go_terms_key = f"go_terms_{self.subontology}"
                if go_terms_key in protein_graph["protein"]:
                    go_terms_dict = protein_graph["protein"][go_terms_key]

                    # For training proteins, use actual annotations
                    # For val/test, use zeros to prevent leakage
                    # Note: PyG NeighborLoader guarantees that sampled seed nodes (i.e. nodes which we are making a prediction on) are first in the batch
                    if local_idx < batch_size and protein_id in self.train_proteins:
                        go_feat = self.convert_go_terms_to_onehot(go_terms_dict)
                    else:
                        go_feat = torch.zeros(self.go_vocab_size, dtype=torch.float32)

                # Extract amino acid features
                aa_feat = protein_graph["aa"].x

            all_interpro_features.append(interpro_feat)
            all_go_features.append(go_feat)
            all_aa_features.append(aa_feat)
            protein_sizes.append(aa_feat.shape[0])

            # Create AA to protein edges
            num_aas = aa_feat.shape[0]
            aa_indices = torch.arange(current_aa_offset, current_aa_offset + num_aas)
            protein_indices = torch.full((num_aas,), local_idx, dtype=torch.long)
            aa_to_protein_edges.append(torch.stack([aa_indices, protein_indices]))

            current_aa_offset += num_aas

        # Update batch with function-related features
        batch["protein"].interpro = torch.stack(all_interpro_features)
        batch["protein"].go = torch.stack(all_go_features)
        # Set protein node features as concatenation of InterPro and GO one hots.
        batch["protein"].x = torch.cat(
            [torch.stack(all_interpro_features), torch.stack(all_go_features)], dim=1
        )

        # Add amino acid nodes and features
        batch["aa"].x = torch.cat(all_aa_features, dim=0)
        batch["aa"].num_nodes = batch["aa"].x.shape[0]

        # Add AA to protein edges
        batch["aa", "belongs_to", "protein"].edge_index = torch.cat(
            aa_to_protein_edges, dim=1
        )

        # Store metadata
        batch["protein"].protein_ids = sampled_protein_ids
        batch["protein"].protein_sizes = protein_sizes

        return batch


def define_loaders(config, dataset):
    """Create NeighborLoader instances for train/val/test."""

    train_loader = NeighborLoader(
        dataset.data,
        num_neighbors={("protein", "aligned_with", "protein"): [-1]},  # 1-hop sampling
        batch_size=config["model"]["batch_size"],
        input_nodes=("protein", dataset.train_mask),
        transform=dataset.get_batch_features,
        shuffle=True,
        num_workers=config["run"]["num_workers"],
    )

    test_loader = NeighborLoader(
        dataset.data,
        num_neighbors={("protein", "aligned_with", "protein"): [-1]},
        batch_size=config["model"]["batch_size"],
        input_nodes=("protein", dataset.test_mask),
        transform=dataset.get_batch_features,
        shuffle=True,
        num_workers=config["run"]["num_workers"],
    )

    # Some datasets, like H30, do not have a validation set
    # This is (dirtily) handled by using the test set as val too.
    if not config["run"]["test_only"]:
        val_loader = NeighborLoader(
            dataset.data,
            num_neighbors={("protein", "aligned_with", "protein"): [-1]},
            batch_size=config["model"]["batch_size"],
            input_nodes=("protein", dataset.val_mask),
            transform=dataset.get_batch_features,
            shuffle=True,
            num_workers=config["run"]["num_workers"],
        )
        return train_loader, val_loader, test_loader
    else:
        return train_loader, test_loader, test_loader
