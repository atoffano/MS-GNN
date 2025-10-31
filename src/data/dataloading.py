"""PyTorch Geometric data loading for heterogeneous protein graphs."""

import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T
import pandas as pd
import pickle
from pathlib import Path
import logging

from src.utils.helpers import timeit
from src.utils.helpers import (
    timeit,
    MemoryTracker,
    track_memory,
    log_worker_checkpoint,
    worker_init_fn,
)

logger = logging.getLogger(__name__)


class SwissProtDataset:
    """Dataset that maintains a static protein-protein graph and loads individual protein features on-demand."""

    def __init__(self, config):
        """Initialize SwissProtDataset."""
        tracker = MemoryTracker(log_prefix="[SwissProtDataset.__init__] ")
        tracker.set_baseline()

        self.config = config
        if config["data"]["dataset"] in ["D1"]:
            self.uses_entryid = True
        else:
            self.uses_entryid = False

        self.graphs_dir = Path(config["data"]["protein_graphs"])

        tracker.log_memory("After basic setup")

        # Load vocabularies
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

        tracker.log_memory("After loading vocabularies")

        # Protein IDs mapping
        self.pid_mapping = (
            pd.read_csv(
                f"./data/swissprot/2024_01/swissprot_2024_01_annotations.tsv",
                sep="\t",
                usecols=["EntryID", "Entry Name"],
            )
            .set_index("EntryID")
            .to_dict()["Entry Name"]
        )
        self.rev_pid_mapping = {v: k for k, v in self.pid_mapping.items()}

        tracker.log_memory(
            f"After loading PID mappings ({len(self.pid_mapping)} entries)"
        )

        # Get preprocessed SwissProt protein graphs
        import numpy as np

        protein_files = [
            f.stem
            for f in self.graphs_dir.glob("*.pt")
            if f.stem not in ["metadata", "interpro_vocab", "go_vocab"]
        ]
        self.proteins = np.array(protein_files, dtype=object)

        tracker.log_memory(
            f"After loading protein list ({len(self.proteins)} proteins)"
        )

        if self.uses_entryid:
            self.proteins = np.array(
                [self.rev_pid_mapping.get(pid, pid) for pid in self.proteins],
                dtype=object,
            )

        # Load GO annotations to determine train/val/test splits
        self._load_split_masks(config)
        tracker.log_memory("After loading split masks")

        # Create the protein-protein heterograph
        self.data = self._create_protein_graph(config)
        tracker.log_memory("After creating protein graph")

        logger.info(
            f"Created protein graph with {self.data['protein'].num_nodes} proteins"
        )
        logger.info(f"Train proteins: {self.train_mask.sum().item()}")
        logger.info(f"Val proteins: {self.val_mask.sum().item()}")
        logger.info(f"Test proteins: {self.test_mask.sum().item()}")

    def _load_split_masks(self, config):
        """Load train/val/test splits based on GO annotations."""
        tracker = MemoryTracker(log_prefix="[_load_split_masks] ")
        tracker.set_baseline()

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
                    [self.rev_pid_mapping.get(pid, pid) for pid in splits["train"]]
                )
            self.train_annots = train_df.set_index("EntryID").to_dict(orient="index")
            for pid in self.train_annots:
                self.train_annots[pid]["term"] = self.train_annots[pid]["term"].split(
                    "; "
                )

        tracker.log_memory(
            f"After loading train annotations ({len(splits['train'])} proteins)"
        )

        logger.info(f"Using {len(splits['train'])} train proteins from {train_path}")

        # Load val and test
        dataset_name = config["data"]["dataset"]
        for split_name in ["val", "test"]:
            split_path = f"./data/{dataset_name}/{dataset_name}_{subontology}_{split_name}_annotations.tsv"
            if Path(split_path).exists():
                split_df = pd.read_csv(split_path, sep="\t")
                splits[split_name] = set(split_df["EntryID"].tolist())

        swissprot_proteins = set(self.proteins.tolist())
        for split in splits:
            missing = list(splits[split] - swissprot_proteins)
            if missing:
                logger.warning(
                    f"{len(missing)} proteins not found in available protein feature set for split '{split}'."
                )
        if missing:
            logger.warning("Missing proteins will be ignored during training and eval.")

        if config["data"]["train_on_swissprot"]:
            for val_test_split in ["val", "test"]:
                splits["train"] = splits["train"] - splits[val_test_split]

        self.protein_to_idx = {pid: i for i, pid in enumerate(self.proteins)}
        self.idx_to_protein = {v: k for k, v in self.protein_to_idx.items()}

        import numpy as np

        self.train_proteins = np.array(list(splits["train"]), dtype=object)
        self.val_proteins = np.array(list(splits["val"]), dtype=object)
        self.test_proteins = np.array(list(splits["test"]), dtype=object)

        num_proteins = len(self.proteins)
        self.train_mask = torch.zeros(num_proteins, dtype=torch.bool)
        self.val_mask = torch.zeros(num_proteins, dtype=torch.bool)
        self.test_mask = torch.zeros(num_proteins, dtype=torch.bool)

        def fill_mask(ids):
            valid = [
                self.protein_to_idx[pid] for pid in ids if pid in self.protein_to_idx
            ]
            if valid:
                return torch.tensor(valid, dtype=torch.long)
            return torch.empty(0, dtype=torch.long)

        self.train_mask[fill_mask(self.train_proteins)] = True
        self.val_mask[fill_mask(self.val_proteins)] = True
        self.test_mask[fill_mask(self.test_proteins)] = True

        logger.info(
            f"Loaded splits - Train: {len(self.train_proteins)}, Val: {len(self.val_proteins)}, Test: {len(self.test_proteins)}"
        )

        self.pos_weights = self._compute_pos_weights()
        logger.info(f"Computed pos weights for {len(self.pos_weights)} GO terms")

        tracker.log_memory("After creating masks and pos_weights")

    @timeit
    def _compute_pos_weights(self):
        """Compute positive weights for each GO term to handle class imbalance."""
        go_to_idx = self.go_vocab_info[self.subontology]["go_to_idx"]
        term_counts = torch.zeros(len(go_to_idx), dtype=torch.float32)

        all_term_indices = []
        for pid in self.train_proteins:
            if pid in self.train_annots:
                terms = self.train_annots[pid].get("term", [])
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

    @track_memory("create_protein_graph")
    def _create_protein_graph(self, config):
        """Creates the high-level protein network."""
        tracker = MemoryTracker(log_prefix="[_create_protein_graph] ")
        tracker.set_baseline()

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

            initial_edges = len(alignment_df)
            alignment_df = alignment_df[
                alignment_df["protein1"].isin(self.protein_to_idx)
                & alignment_df["protein2"].isin(self.protein_to_idx)
            ]
            filtered_edges = initial_edges - len(alignment_df)
            if filtered_edges > 0:
                logger.info(
                    f"Filtered {filtered_edges} alignment edges with proteins not in dataset"
                )

            source_indices = alignment_df["protein1"].map(self.protein_to_idx).tolist()
            target_indices = alignment_df["protein2"].map(self.protein_to_idx).tolist()

            edge_index = torch.tensor(
                [source_indices, target_indices], dtype=torch.long
            )
            edge_attrs = None

            return edge_index, edge_attrs

        @timeit
        def stringdb_edge_data():
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

            stringdb_df = stringdb_df[
                stringdb_df["protein1"].isin(stringdb_mapping)
                & stringdb_df["protein2"].isin(stringdb_mapping)
            ]

            if self.uses_entryid:
                stringdb_df["protein1"] = stringdb_df["protein1"].map(
                    self.rev_pid_mapping
                )
                stringdb_df["protein2"] = stringdb_df["protein2"].map(
                    self.rev_pid_mapping
                )

            initial_edges = len(stringdb_df)
            stringdb_df = stringdb_df[
                stringdb_df["protein1"].isin(self.protein_to_idx)
                & stringdb_df["protein2"].isin(self.protein_to_idx)
            ]
            filtered_edges = initial_edges - len(stringdb_df)
            if filtered_edges > 0:
                logger.info(
                    f"Filtered {filtered_edges} STRING-DB edges with proteins not in dataset"
                )

            source_indices = stringdb_df["protein1"].map(self.protein_to_idx).tolist()
            target_indices = stringdb_df["protein2"].map(self.protein_to_idx).tolist()

            edge_index = torch.tensor(
                [source_indices, target_indices], dtype=torch.long
            )
            edge_attrs = None

            return edge_index, edge_attrs

        data = HeteroData()
        logger.info("Creating protein-protein graph edges...")

        alignment_edge_index, alignment_edge_attrs = alignment_edge_data()
        tracker.log_memory(
            f"After loading alignment edges ({alignment_edge_index.shape[1]} edges)"
        )
        logger.info(f"Alignment edges: {alignment_edge_index.shape[1]}")

        stringdb_edge_index, stringdb_edge_attrs = stringdb_edge_data()
        tracker.log_memory(
            f"After loading STRING-DB edges ({stringdb_edge_index.shape[1]} edges)"
        )
        logger.info(f"STRINGdb edges: {stringdb_edge_index.shape[1]}")

        num_proteins = len(self.proteins)
        data["protein"].num_nodes = num_proteins

        data["protein", "aligned_with", "protein"].edge_index = alignment_edge_index
        data["protein", "stringdb", "protein"].edge_index = stringdb_edge_index

        tracker.log_memory("After creating HeteroData structure")

        return data

    def load_protein_graph(self, protein_id):
        """Load a single protein graph from disk."""
        log_worker_checkpoint(f"Before loading protein {protein_id}")
        try:
            graph = torch.load(
                f"{self.graphs_dir}/{protein_id}.pt",
                map_location="cpu",
                weights_only=False,
            )
            log_worker_checkpoint(f"After loading protein {protein_id}")
            return graph
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
        log_worker_checkpoint("Start get_batch_features")

        sampled_protein_ids = [
            self.idx_to_protein[idx.item()] for idx in batch["protein"].n_id
        ]
        if self.uses_entryid:
            sampled_protein_ids = [
                self.pid_mapping.get(pid, pid) for pid in sampled_protein_ids
            ]

        log_worker_checkpoint(
            f"After extracting {len(sampled_protein_ids)} protein IDs"
        )

        # Load individual protein graphs and extract features
        all_interpro_features = []
        all_go_features = []
        all_aa_features = []
        aa_to_protein_edges = []
        protein_sizes = []
        current_aa_offset = 0

        for local_idx, protein_id in enumerate(sampled_protein_ids):
            if local_idx % 10 == 0:
                log_worker_checkpoint(
                    f"Loading protein {local_idx}/{len(sampled_protein_ids)}"
                )

            protein_graph = self.load_protein_graph(protein_id)

            if protein_graph is None:
                logger.warning(f"Using empty features for missing protein {protein_id}")
                interpro_feat = torch.zeros(self.ipr_vocab_size, dtype=torch.float32)
                go_feat = torch.zeros(self.go_vocab_size, dtype=torch.float32)
                aa_feat = torch.zeros(200, 1280, dtype=torch.float32)
            else:
                interpro_feat = protein_graph["protein"].interpro.squeeze(0)
                aa_feat = protein_graph["aa"].x
                go_feat = self.convert_go_terms_to_onehot(
                    protein_graph["protein"][f"go_terms_{self.subontology}"]
                )

            all_interpro_features.append(interpro_feat)
            all_go_features.append(go_feat)
            all_aa_features.append(aa_feat)
            protein_sizes.append(aa_feat.shape[0])

            num_aas = aa_feat.shape[0]
            aa_indices = torch.arange(current_aa_offset, current_aa_offset + num_aas)
            protein_indices = torch.full((num_aas,), local_idx, dtype=torch.long)
            aa_to_protein_edges.append(torch.stack([aa_indices, protein_indices]))

            current_aa_offset += num_aas

        log_worker_checkpoint("After loading all protein features")

        # Update batch with function-related features
        batch["protein"].interpro = torch.stack(all_interpro_features)
        batch["protein"].go = torch.stack(all_go_features)
        batch["protein"].y = batch["protein"].go[: batch["protein"].batch_size].clone()
        batch["protein"].go[: batch["protein"].batch_size] = 0.0

        if batch["mode"] == "train":
            n_id = batch["protein"].n_id
            mask_val = self.val_mask[n_id]
            mask_test = self.test_mask[n_id]
            mask = mask_val | mask_test
            batch["protein"].go[mask] = 0.0

        batch["protein"].x = torch.cat(
            [torch.stack(all_interpro_features), torch.stack(all_go_features)], dim=1
        )

        batch["aa"].x = torch.cat(all_aa_features, dim=0).float()
        batch["aa"].num_nodes = batch["aa"].x.shape[0]

        batch["aa", "belongs_to", "protein"].edge_index = torch.cat(
            aa_to_protein_edges, dim=1
        )

        batch["protein"].protein_ids = sampled_protein_ids
        batch["protein"].protein_sizes = protein_sizes

        log_worker_checkpoint(
            f"End get_batch_features (total AA nodes: {batch['aa'].num_nodes})"
        )

        return batch


def make_batch_transform(dataset, mode):
    """Create a batch transformation function for the data loader."""

    def batch_transform(batch):
        """Transform batch by adding mode and loading protein features."""
        log_worker_checkpoint(f"Start batch_transform (mode={mode})")
        batch["mode"] = mode
        batch = dataset.get_batch_features(batch)
        log_worker_checkpoint(f"End batch_transform (mode={mode})")
        return batch

    return batch_transform


def define_loaders(config, dataset):
    """Create NeighborLoader instances for train/val/test."""

    edge_types = [tuple(et) for et in config["model"]["edge_types"]]
    protein_edge_types = [
        et for et in edge_types if et[0] == "protein" and et[2] == "protein"
    ]
    num_neighbors = {et: [-1] for et in protein_edge_types}

    train_loader = NeighborLoader(
        dataset.data,
        num_neighbors=num_neighbors,
        batch_size=config["model"]["batch_size"],
        input_nodes=("protein", dataset.train_mask),
        transform=make_batch_transform(dataset, mode="train"),
        shuffle=True,
        num_workers=config["trainer"]["num_workers"],
        persistent_workers=True if config["trainer"]["num_workers"] > 0 else False,
        drop_last=True,
        worker_init_fn=worker_init_fn if config["trainer"]["num_workers"] > 0 else None,
    )

    test_loader = NeighborLoader(
        dataset.data,
        num_neighbors=num_neighbors,
        batch_size=config["model"]["batch_size"],
        input_nodes=("protein", dataset.test_mask),
        transform=make_batch_transform(dataset, mode="predict"),
        shuffle=False,
        num_workers=config["trainer"]["num_workers"],
        persistent_workers=True if config["trainer"]["num_workers"] > 0 else False,
        worker_init_fn=worker_init_fn if config["trainer"]["num_workers"] > 0 else None,
    )

    if config["data"]["dataset"] != "H30":
        val_loader = NeighborLoader(
            dataset.data,
            num_neighbors=num_neighbors,
            batch_size=config["model"]["batch_size"],
            input_nodes=("protein", dataset.val_mask),
            transform=make_batch_transform(dataset, mode="predict"),
            shuffle=False,
            num_workers=config["trainer"]["num_workers"],
            persistent_workers=True if config["trainer"]["num_workers"] > 0 else False,
            worker_init_fn=(
                worker_init_fn if config["trainer"]["num_workers"] > 0 else None
            ),
        )
        return train_loader, val_loader, test_loader
    else:
        return train_loader, test_loader, test_loader
