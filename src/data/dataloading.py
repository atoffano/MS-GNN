import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T
import pandas as pd
import pickle
from pathlib import Path
import logging
from src.utils.constants import *
from src.utils.helpers import timeit

logger = logging.getLogger(__name__)


class SwissProtDataset:
    """Dataset that maintains a static protein-protein graph and loads individual protein features on-demand."""

    @timeit
    def __init__(self, config):
        self.config = config
        if config["data"]["dataset"] in USES_ENTRYID:
            self.uses_entryid = True
        else:
            self.uses_entryid = False

        self.graphs_dir = Path(PROTEIN_GRAPHS_DIR)

        with open(INTERPRO_VOCAB, "rb") as f:
            self.ipr_vocab_size = pickle.load(f)["vocab_size"]

        with open(GO_VOCAB, "rb") as f:
            go_info = pickle.load(f)
        self.go_vocab_info = go_info
        self.go_vocab_sizes = {
            onto: info["vocab_size"] for onto, info in go_info.items()
        }
        self.subontology = config["data"]["subontology"]
        self.go_vocab_size = self.go_vocab_sizes[self.subontology]

        # Protein IDs are in the Accession Number format (e.g. P12345) or in the EntryID format (e.g. INS_HUMAN), depending on the dataset
        # Dict like ('INS_HUMAN', 'P01308')
        self.pid_mapping = (
            pd.read_csv(
                PID_MAPPING,  # Most up to date mapping taken from 2024_01 raw annotations
                sep="\t",
                usecols=["EntryID", "Entry Name"],
            )
            .set_index("EntryID")
            .to_dict()["Entry Name"]
        )
        self.rev_pid_mapping = {v: k for k, v in self.pid_mapping.items()}

        # Get preprocessed SwissProt protein graphs
        self.proteins = [
            f.stem
            for f in self.graphs_dir.glob("*.pt")
            if f.stem not in ["metadata", "interpro_vocab", "go_vocab"]
        ]
        if self.uses_entryid:
            # Convert potential Accession Numbers to EntryIDs
            self.proteins = [
                self.rev_pid_mapping.get(pid, pid) for pid in self.proteins
            ]

        # Load GO annotations to determine train/val/test splits
        self._load_split_masks(config)

        # Load external annotations if provided
        if config["data"].get("dataset") == "swissprot":
            annot_path = self.train_annots_path
            if Path(annot_path).exists():
                logger.info(f"Loading external annotations from {annot_path}")
                annot_df = pd.read_csv(annot_path, sep="\t")
                self.external_annotations = {}
                for _, row in annot_df.iterrows():
                    pid = row["EntryID"]
                    if self.uses_entryid:
                        pid = self.rev_pid_mapping.get(pid, pid)
                    self.external_annotations[pid] = row["term"].split("; ")
                logger.info(
                    f"Loaded external annotations for {len(self.external_annotations)} proteins"
                )
            else:
                logger.warning(f"External annotations file not found at {annot_path}")

        # Create the protein-protein heterograph
        self.data = self._create_protein_graph(config)
        self.transform = T.Compose(
            [
                T.RemoveDuplicatedEdges(key="edge_attr", reduce="mean"),
                T.ToUndirected(reduce="max"),
                T.AddRemainingSelfLoops(attr="edge_attr", fill_value=1.0),
            ]
        )
        self.data = self.transform(self.data)

        logger.info(
            f"Created protein graph with {self.data['protein'].num_nodes} proteins"
        )
        logger.info(f"Train proteins: {self.train_mask.sum().item()}")
        logger.info(f"Val proteins: {self.val_mask.sum().item()}")
        logger.info(f"Test proteins: {self.test_mask.sum().item()}")

    def _load_split_masks(self, config):
        """Load train/val/test splits based on GO annotations. In case of a longitudinal setup,
        ensure no leakage from future annotations by overriding default protein annotations
        """
        splits = {"train": set(), "val": set(), "test": set()}
        subontology = self.subontology
        release = config["data"].get("swissprot_release", None)
        dataset = config["data"]["dataset"]

        if config["data"]["train_on_swissprot"] or dataset == "swissprot":
            exp_suffix = "exp_" if config["data"].get("exp_only", True) else ""
            train_path = f"./data/swissprot/{release}/swissprot_{release}_{subontology}_{exp_suffix}annotations.tsv"
            self.train_annots_path = train_path
            logger.info(f"Training on SwissProt annotations from {train_path}")
        else:
            # Stick to original dataset's train set
            train_path = (
                f"./data/{dataset}/{dataset}_{subontology}_train_annotations.tsv"
            )
            logger.info(f"Training on original dataset annotations from {train_path}")

        if Path(train_path).exists():
            train_df = pd.read_csv(train_path, sep="\t")
            splits["train"] = set(train_df["EntryID"].tolist())
            if config["data"]["train_on_swissprot"] and self.uses_entryid:
                splits["train"] = set(
                    [self.rev_pid_mapping.get(pid, pid) for pid in splits["train"]]
                )
            # Store train annotations for on-the-fly GO term loading
            # self.train_annots = train_df.set_index("EntryID").to_dict(orient="index")
            # for pid in self.train_annots:
            #     self.train_annots[pid]["term"] = self.train_annots[pid]["term"].split(
            #         "; "
            #     )
        logger.info(f"Using {len(splits['train'])} train proteins from {train_path}")

        # Load val and test
        for split_name in ["val", "test"]:
            split_path = (
                f"./data/{dataset}/{dataset}_{subontology}_{split_name}_annotations.tsv"
            )
            if release:
                test_exp_suffix = (
                    "exp_" if config["data"].get("exp_only", True) else "cur_"
                )
                split_path = f"./data/{dataset}/{release}/{dataset}_{release}_{subontology}_{split_name}_{test_exp_suffix}annotations.tsv"

            if Path(split_path).exists():
                split_df = pd.read_csv(split_path, sep="\t")
                splits[split_name] = set(split_df["EntryID"].tolist())
            logger.info(
                f"Using {len(splits[split_name])} {split_name} proteins from {split_path}"
            )

        for split in splits:
            missing = list(splits[split] - set(self.proteins))
            if missing:
                logger.warning(
                    f"{len(missing)} proteins not found in available protein feature set for split '{split}'."
                )

        # Remove proteins from val/test if training on the full SwissProt release.
        if config["data"]["train_on_swissprot"] or dataset == "swissprot":
            for split in ["val", "test"]:
                splits["train"] = splits["train"] - splits[split]

        self.protein_to_idx = {pid: i for i, pid in enumerate(self.proteins)}
        self.idx_to_protein = {v: k for k, v in self.protein_to_idx.items()}
        num_proteins = len(self.proteins)

        def get_protein_mask(split):
            mask = torch.zeros(num_proteins, dtype=torch.bool)
            protein_idx = [
                self.protein_to_idx[pid] for pid in split if pid in self.protein_to_idx
            ]
            mask[protein_idx] = True
            mask.share_memory_()
            return protein_idx, mask

        self.train_idx, self.train_mask = get_protein_mask(splits["train"])
        self.val_idx, self.val_mask = get_protein_mask(splits["val"])
        self.test_idx, self.test_mask = get_protein_mask(splits["test"])
        assert not (
            set(self.train_idx) & set(self.val_idx)
            | set(self.train_idx) & set(self.test_idx)
            | set(self.val_idx) & set(self.test_idx)
        ), "Data leakage: masks overlap!"

        logger.info(
            f"Loaded splits - Train: {len(self.train_idx)}, Val: {len(self.val_idx)}, Test: {len(self.test_idx)}"
        )

        # BCE pos weights for handling class imbalance
        # self.pos_weights = self._compute_pos_weights()
        # logger.info(f"Computed pos weights for {len(self.pos_weights)} GO terms")

    @timeit
    def _compute_pos_weights(self):
        """LEGACY | Compute positive weights for each GO term to handle class imbalance."""
        go_to_idx = self.go_vocab_info[self.subontology]["go_to_idx"]
        term_counts = torch.zeros(len(go_to_idx), dtype=torch.float32)

        all_term_indices = []
        for protein in self.train_idx:
            pid = self.idx_to_protein[protein]
            if pid in self.train_annots:
                terms = self.train_annots[pid]["term"]
                term_indices = [go_to_idx[term] for term in terms if term in go_to_idx]
                all_term_indices.extend(term_indices)
        if all_term_indices:
            term_indices_tensor = torch.tensor(all_term_indices, dtype=torch.long)
            term_counts = torch.bincount(
                term_indices_tensor, minlength=len(go_to_idx)
            ).float()
        inf_mask = term_counts == 0
        # pos_weights = len(self.train_idx) / (term_counts + 1)
        pos_weights = len(self.train_idx) / (len(term_counts) * (term_counts + 1e-8))

        # Replace pos weights equal to len(self.train_idx) with 1
        if inf_mask.any():
            pos_weights[inf_mask] = pos_weights.min().item()
            logger.info(
                f"{inf_mask.sum().item()} GO terms had zero positive samples; set pos weight to min"
            )
        logger.info(f"Pos weight sample: {pos_weights[:10]}")
        logger.info(
            f"Pos weight stats - Min: {pos_weights.min()}, Max: {pos_weights.max()}, Mean: {pos_weights.mean()}"
        )
        sorted_weights, _ = torch.sort(pos_weights)
        logger.info(
            f"Sorted pos weights sample: {sorted_weights[:10]} ... {sorted_weights[-10:]}"
        )
        return pos_weights

    def _create_protein_graph(self, config):
        """Creates the high-level protein network."""

        @timeit
        def alignment_edge_data():
            alignment_df = pd.read_csv(
                DIAMOND_ALIGNMENT,
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
                alignment_df["protein1"].isin(self.proteins)
                & alignment_df["protein2"].isin(self.proteins)
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
                # features = alignment_df.drop(columns=["protein1", "protein2"])
                features = alignment_df[["bitscore"]]
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
                    STRINGDB_SWISSPROT_MAPPING,
                    sep="\t",
                    usecols=["From", "To"],
                )
                .set_index("From")
                .to_dict()["To"]
            )
            rev_stringdb_mapping = {v: k for k, v in stringdb_mapping.items()}

            stringdb_df = pd.read_csv(
                STRINGDB_PATH,
                sep="\t",
                header=0,
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
                stringdb_df["protein1"].isin(self.proteins)
                & stringdb_df["protein2"].isin(self.proteins)
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
                features = stringdb_df[["combined_score"]]
                # features = stringdb_df.drop(columns=["protein1", "protein2"])
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

        return data.detach().clone()

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

    def _terms_to_onehot(self, terms):
        """Helper to convert a list of GO terms to one-hot encoding."""
        go_to_idx = self.go_vocab_info[self.subontology]["go_to_idx"]
        onehot = torch.zeros(self.go_vocab_size, dtype=torch.float32)
        for term in terms:
            if term in go_to_idx:
                onehot[go_to_idx[term]] = 1.0
        return onehot

    def convert_go_terms_to_onehot(self, go_terms_dict):
        """Convert GO terms to one-hot encoding based on config."""
        if self.config["data"]["exp_only"]:
            terms = go_terms_dict.get("experimental", [])
        else:
            terms = go_terms_dict.get("curated", [])
        return self._terms_to_onehot(terms)

    def get_batch_features(self, batch, return_sequences=False):
        """Load individual protein features and amino acid data for the sampled batch."""
        with torch.no_grad():
            raw_protein_ids = [
                self.idx_to_protein[idx.item()] for idx in batch["protein"].n_id
            ]
            sampled_protein_ids = raw_protein_ids
            if self.uses_entryid:
                sampled_protein_ids = [
                    self.pid_mapping.get(pid, pid) for pid in sampled_protein_ids
                ]

            use_edge_attrs = self.config["model"]["edge_attrs"]
            use_contact = ["aa", "close_to", "aa"] in self.config["model"]["edge_types"]

            if return_sequences:
                sampled_sequences = []
            batch_interpro_features = []
            batch_go_features = []
            batch_aa_features = []
            aa_to_protein_edges = []
            contact_edges = [] if use_contact else None
            contact_attrs = [] if use_edge_attrs else None
            protein_sizes = []

            aa_offset = 0  # Dynamic offset for aa nodes id (on-the-fly attribution)

            for local_idx, protein_id in enumerate(sampled_protein_ids):
                protein_graph = self.load_protein_graph(protein_id)
                raw_pid = raw_protein_ids[local_idx]

                if protein_graph is None:
                    logger.warning(
                        f"Using empty features for missing protein {protein_id}"
                    )
                    interpro_feat = torch.zeros(
                        self.ipr_vocab_size, dtype=torch.float32
                    )
                    go_feat = torch.zeros(self.go_vocab_size, dtype=torch.float32)
                    aa_feat = torch.zeros(
                        200, 1280, dtype=torch.float32
                    )  # Default 200 AAs
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
                    # Load features
                    if return_sequences:
                        sampled_sequences.append(protein_graph["protein"].sequence)
                    interpro_feat = protein_graph["protein"].interpro.squeeze(0)
                    aa_feat = protein_graph["aa"].x

                    if self.external_annotations is not None:
                        if raw_pid in self.external_annotations:
                            go_feat = self._terms_to_onehot(
                                self.external_annotations[raw_pid]
                            )
                        else:
                            go_feat = torch.zeros(
                                self.go_vocab_size, dtype=torch.float32
                            )
                    else:
                        go_feat = self.convert_go_terms_to_onehot(
                            protein_graph["protein"][f"go_terms_{self.subontology}"]
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

                # Load aa contact edge attributes if applicable
                if use_edge_attrs and use_contact:
                    contact_attrs.append(local_contact_edge_attr)

            # Update batch with function-related features & set labels
            seed_nodes = batch["protein"].n_id[: batch["protein"].batch_size]
            batch["protein"].interpro = torch.stack(batch_interpro_features)

            batch["protein"].y = torch.stack(batch_go_features)[
                : batch["protein"].batch_size
            ].clone()
            batch["protein"].go = torch.stack(batch_go_features)
            batch["protein"].go[
                : batch["protein"].batch_size
            ] = 0.0  # Mask seed protein labels
            batch["protein"].go

            if torch.isin(
                batch["protein"].n_id[batch["protein"].batch_size :], seed_nodes
            ).any():
                logger.warning("Seed nodes found in neighborhood nodes of the batch!")
                neighborhood_nodes = batch["protein"].n_id[
                    batch["protein"].batch_size :
                ]
                overlapping_nodes = torch.isin(neighborhood_nodes, seed_nodes)
                overlapping_indices = neighborhood_nodes[overlapping_nodes]
                overlapping_proteins = [
                    self.idx_to_protein[idx.item()] for idx in overlapping_indices
                ]
                logger.warning(
                    f"Overlapping seed proteins in neighborhood nodes: {overlapping_proteins}"
                )
                logger.warning(
                    f"Nid of neighborhood nodes with overlap: {overlapping_indices}"
                )
                logger.warning(f"Seed node nids: {seed_nodes}")
                logger.warning(f"Neighborhood node nids: {neighborhood_nodes}")
                logger.warning(f"Batch protein nids: {batch['protein'].n_id}")

            # Mask GO features for val/test proteins
            if batch["mode"] == "train":
                n_id = batch["protein"].n_id
                mask_val = self.val_mask[n_id]
                mask_test = self.test_mask[n_id]
                mask = mask_val | mask_test
                batch["protein"].go[mask] = 0.0

            # Deletion experiment: set .x features to normal distribution samples
            # batch["protein"].x = torch.randn(
            #     batch["protein"].num_nodes, self.ipr_vocab_size + self.go_vocab_size
            # ).float()

            # Set protein node features as concatenation of InterPro and GO one hots.
            # batch["protein"].x = torch.cat(
            #     [batch["protein"].interpro, batch["protein"].go],
            #     dim=1,
            # )

            # IPR Ablation: only GO terms as features
            batch["protein"].x = batch["protein"].go

            # Add amino acid nodes and features
            batch["aa"].x = torch.cat(batch_aa_features, dim=0).float()
            batch["aa"].num_nodes = batch["aa"].x.shape[0]

            # Add AA to protein edges
            batch["aa", "belongs_to", "protein"].edge_index = torch.cat(
                aa_to_protein_edges, dim=1
            )

            if use_contact:
                batch["aa", "close_to", "aa"].edge_index = (
                    torch.cat(contact_edges, dim=1)
                    if contact_edges
                    else torch.empty((2, 0), dtype=torch.long)
                )

            # Normalized distance between aa as edge attributes.
            # Note: edge_attr is stored as sqrt of the Angstrom distance.
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

            batch = self.transform(batch)
            return batch


def make_batch_transform(dataset, mode, return_sequences=False):
    """Populate batch with features."""

    def batch_transform(batch):
        batch["mode"] = mode
        batch = dataset.get_batch_features(batch, return_sequences=return_sequences)
        return batch

    return batch_transform


def define_loaders(config, dataset):
    """Create NeighborLoader instances for train/val/test."""

    # Which edges to sample and how many neighbors
    num_neighbors = {}
    for edge_type_str, num_samples in config["model"]["sampled_edges"].items():
        edge_type_tuple = tuple(edge_type_str.split("__"))
        num_neighbors[edge_type_tuple] = [num_samples]
    logger.info("Subgraph sampling configuration", num_neighbors)

    # num_neighbors = {("protein", "aligned_with", "protein"): [-1]}
    train_loader = NeighborLoader(
        dataset.data,
        num_neighbors=num_neighbors,
        batch_size=config["model"]["batch_size"],
        input_nodes=("protein", dataset.train_mask),
        transform=make_batch_transform(dataset, mode="train"),
        shuffle=True,
        num_workers=config["trainer"]["num_workers"],
        drop_last=True,
    )

    test_loader = NeighborLoader(
        dataset.data,
        num_neighbors=num_neighbors,
        batch_size=config["model"]["batch_size"],
        input_nodes=("protein", dataset.test_mask),
        transform=make_batch_transform(dataset, mode="predict"),
        shuffle=False,
        num_workers=config["trainer"]["num_workers"],
    )

    # Some datasets, do not have a validation set
    # This is (dirtily) handled by using the test set as val too.
    if config["data"]["dataset"] not in USES_ENTRYID:
        val_loader = NeighborLoader(
            dataset.data,
            num_neighbors=num_neighbors,
            batch_size=config["model"]["batch_size"],
            input_nodes=("protein", dataset.val_mask),
            transform=make_batch_transform(dataset, mode="predict"),
            shuffle=False,
            num_workers=config["trainer"]["num_workers"],
        )
        return train_loader, val_loader, test_loader
    else:
        return train_loader, test_loader, test_loader
