import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T
import pandas as pd
import pickle
from pathlib import Path
import logging
from src.utils.constants import (
    CONTACT_CUTOFF,
    DIAMOND_ALIGNMENT,
    GO_VOCAB,
    INTERPRO_VOCAB,
    PID_MAPPING,
    PROTEIN_GRAPHS_DIR,
    STRINGDB_PATH,
    STRINGDB_SWISSPROT_MAPPING,
    USES_ENTRYID,
)
from src.utils.helpers import timeit

logger = logging.getLogger(__name__)


class SwissProtDataset:
    """Dataset that maintains a static protein-protein graph and loads individual protein features on-demand."""

    @timeit
    def __init__(self, config):
        self.config = config
        self.external_annotations = None
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

        # Load external annotations, if provided
        if config["data"].get("dataset") == "swissprot":
            self._load_external_annotations(self.train_annots_path)
        if config["data"].get("external_annotations_path", None):
            self._load_external_annotations(config["data"]["external_annotations_path"])

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

        # Pad edge_weight for edges added by AddRemainingSelfLoops.
        # The self-loop transform only fills `edge_attr` but not `edge_weight`,
        # causing a length mismatch. Self-loop edges get weight=1.0.
        for edge_type in self.data.edge_types:
            store = self.data[edge_type]
            if hasattr(store, "edge_weight") and store.edge_weight is not None:
                num_edges = store.edge_index.size(1)
                num_weights = store.edge_weight.size(0)
                if num_weights < num_edges:
                    pad = torch.ones(num_edges - num_weights, dtype=store.edge_weight.dtype)
                    store.edge_weight = torch.cat([store.edge_weight, pad])

        logger.info(
            f"Created protein graph with {self.data['protein'].num_nodes} proteins"
        )
        logger.info(f"Train proteins: {self.train_mask.sum().item()}")
        logger.info(f"Val proteins: {self.val_mask.sum().item()}")
        logger.info(f"Test proteins: {self.test_mask.sum().item()}")

    def _load_external_annotations(self, annot_path):
        """Load external GO annotations from a TSV file."""
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
                f"Loaded {len(self.external_annotations)} external annotations from {annot_path}"
            )
        else:
            logger.warning(f"External annotations file not found at {annot_path}")

    def _load_split_masks(self, config):
        """Load train/val/test splits based on GO annotations.
        In case of a longitudinal setup, ensure no leakage from future annotations by overriding default protein annotations.
        """
        splits = {"train": set(), "val": set(), "test": set()}
        subontology = self.subontology
        release = config["data"].get("swissprot_release", None)
        dataset = config["data"]["dataset"]
        train_proteins_config = config["data"].get("train_proteins", "dataset")

        if train_proteins_config == "swissprot" or dataset == "swissprot":
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
            if train_proteins_config == "swissprot" and self.uses_entryid:
                splits["train"] = set(
                    [self.rev_pid_mapping.get(pid, pid) for pid in splits["train"]]
                )
            # Store train annotations for on-the-fly GO term loading
            # self.train_annots = train_df.set_index("EntryID").to_dict(orient="index")
            # for pid in self.train_annots:
            #     self.train_annots[pid]["term"] = self.train_annots[pid]["term"].split(
            #         "; "
            #     )
        else:
            logger.error(f"Train annotations file not found at {train_path}")
            raise FileNotFoundError(
                f"Train annotations file not found at {train_path}. Check for conflicting config settings ?"
            )

        logger.info(f"Using {len(splits['train'])} train proteins from {train_path}")

        # Load val and test
        for split_name in ["val", "test"]:
            split_path = (
                f"./data/{dataset}/{dataset}_{subontology}_{split_name}_annotations.tsv"
            )
            if release and dataset == "swissprot":
                split_path = f"./data/swissprot/2024_01/swissprot_2024_01_{subontology}_{split_name}_annotations.tsv"
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
        if train_proteins_config == "swissprot" or dataset == "swissprot":
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



    def _create_protein_graph(self, config):
        """Creates the high-level protein network."""

        @timeit
        def alignment_edge_data():
            alignment_df = pd.read_csv(
                config["data"].get("alignment_path", DIAMOND_ALIGNMENT),
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
            if config["model"][
                "edge_attrs"
            ]:  # Z-score Normalize and shift bitscore as edge attribute
                features = alignment_df[["bitscore"]]
                z_scores = ((features - features.mean()) / features.std()).values
                z_shifted = z_scores - z_scores.min()  # Shift so min value is zero
                edge_attrs = torch.tensor(
                    z_shifted,
                    dtype=torch.float32,
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
                config["data"].get("stringdb_path", STRINGDB_PATH),
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
            if config["model"][
                "edge_attrs"
            ]:  # Z-score Normalize and shift combined_score as edge attribute
                features = stringdb_df[["combined_score"]]
                z_scores = ((features - features.mean()) / features.std()).values
                z_shifted = z_scores - z_scores.min()  # Shift so min value is zero
                edge_attrs = torch.tensor(
                    z_shifted,
                    dtype=torch.float32,
                )
                # 1D weights for biased neighbor sampling (separate from 2D edge_attr used by GATv2)
                edge_weights = edge_attrs.squeeze(-1)
            else:
                edge_attrs = None
                edge_weights = None

            return edge_index, edge_attrs, edge_weights

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
            stringdb_edge_index, stringdb_edge_attrs, stringdb_edge_weights = stringdb_edge_data()
            data["protein", "stringdb", "protein"].edge_index = stringdb_edge_index
            logger.info(f"STRINGdb edges: {stringdb_edge_index.shape[1]}")
            if config["model"]["edge_attrs"]:
                data["protein", "stringdb", "protein"].edge_attr = stringdb_edge_attrs
            if stringdb_edge_weights is not None:
                data["protein", "stringdb", "protein"].edge_weight = stringdb_edge_weights

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
            mean_pool_aa = self.config["model"].get("mean_pool_aa", False)
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
                    # Load features
                    if return_sequences:
                        sampled_sequences.append(protein_graph["protein"].sequence)
                    interpro_feat = protein_graph["protein"].interpro.squeeze(0)
                    aa_feat = protein_graph["aa"].x
                    if mean_pool_aa:
                        aa_feat = aa_feat.mean(dim=0, keepdim=True)

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

            if (
                self.config["model"]["interpro"]
                and self.config["model"]["go_neighbors"]
            ):
                # Set protein node features as concatenation of InterPro and GO one hots.
                batch["protein"].x = torch.cat(
                    [batch["protein"].interpro, batch["protein"].go],
                    dim=1,
                )
            elif (
                self.config["model"]["interpro"]
                and not self.config["model"]["go_neighbors"]
            ):  # GO Ablation: only IPR features
                batch["protein"].x = batch["protein"].interpro
            elif (
                not self.config["model"]["interpro"]
                and self.config["model"]["go_neighbors"]
            ):  # IPR Ablation: only GO terms as features
                batch["protein"].x = batch["protein"].go
            else:
                # Zero input
                batch["protein"].x = torch.zeros(
                    batch["protein"].num_nodes, self.ipr_vocab_size + self.go_vocab_size
                ).float()

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

            try:
                batch = self.transform(batch)
            except Exception as e:
                logger.error(f"Error applying graph transformations: {e}")
                logger.error(f"Batch before transformation: {batch}")
                # logger.error(
                #     f"Batch edge_attrs before transformation: {batch['protein', 'stringdb', 'protein'].edge_attr}"
                # )
                # logger.error(
                #     f"Batch edge_attrs before transformation: {batch['protein', 'aligned_with', 'protein'].edge_attr}"
                # )

            return batch

def make_batch_transform(dataset, mode, return_sequences=False):
    """Populate batch with features.

    Args:
        dataset: SwissProtDataset instance.
        mode: 'train' or 'predict'.
        return_sequences: Whether to include protein sequences in the batch.
    """

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

    logger.info("Subgraph sampling configuration: %s", num_neighbors)

    # Weighted/biased neighbor sampling: edges with higher edge_weight are sampled preferentially.
    # Only edge types that have the "edge_weight" attribute are affected (currently: stringdb).
    # Edge types without it (aligned_with) keep uniform sampling.
    weight_attr = "edge_weight" if config["model"].get("weighted_sampling", False) else None
    if weight_attr:
        logger.info("Weighted neighbor sampling enabled (weight_attr='%s')", weight_attr)

    # num_neighbors = {("protein", "aligned_with", "protein"): [-1]}
    train_loader = NeighborLoader(
        dataset.data,
        num_neighbors=num_neighbors,
        weight_attr=weight_attr,
        batch_size=config["model"]["batch_size"],
        input_nodes=("protein", dataset.train_mask),
        transform=make_batch_transform(
            dataset, mode="train",
        ),
        shuffle=True,
        num_workers=config["trainer"]["num_workers"],
        drop_last=True,
    )

    test_loader = NeighborLoader(
        dataset.data,
        num_neighbors=num_neighbors,
        weight_attr=weight_attr,
        batch_size=config["model"]["batch_size"],
        input_nodes=("protein", dataset.test_mask),
        transform=make_batch_transform(
            dataset, mode="predict",
        ),
        shuffle=False,
        num_workers=config["trainer"]["num_workers"],
    )

    # Some datasets, do not have a validation set
    # This is (dirtily) handled by using the test set as val too.
    if dataset.val_mask.sum() > 0:
        val_loader = NeighborLoader(
            dataset.data,
            num_neighbors=num_neighbors,
            weight_attr=weight_attr,
            batch_size=config["model"]["batch_size"],
            input_nodes=("protein", dataset.val_mask),
            transform=make_batch_transform(
                dataset, mode="predict",
            ),
            shuffle=False,
            num_workers=config["trainer"]["num_workers"],
        )
        return train_loader, val_loader, test_loader
    else:
        logger.info("No validation set found; using test set as validation.")
        return train_loader, test_loader, test_loader
