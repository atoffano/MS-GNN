import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
import pandas as pd
import h5py
from src.utils import constants
from src.utils.helpers import timeit


class SwissProtHeteroDataset:
    def __init__(self, h5_path, tsv_paths, go_split_paths, alignment_path, split):
        self.go_train_dict = self._load_go_annotations(go_split_paths["train"])
        self.go_val_dict = self._load_go_annotations(go_split_paths["val"])
        self.go_test_dict = self._load_go_annotations(go_split_paths["test"])
        self.split = split
        self.h5_path = h5_path
        self.interpro_dict = self._load_interpro_annotations(tsv_paths["InterPro"])

        # Load alignment info for 'aligned_with' edges
        self.alignment_df = pd.read_csv(
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

        # Collect all proteins appearing in any annotation or alignment file
        go_prot_ids = (
            set(self.go_train_dict.keys())
            | set(self.go_val_dict.keys())
            | set(self.go_test_dict.keys())
        )
        ipr_prot_ids = set(self.interpro_dict.keys())
        align_prot_ids = set(self.alignment_df["protein1"]).union(
            set(self.alignment_df["protein2"])
        )
        proteins = sorted(go_prot_ids | ipr_prot_ids | align_prot_ids)
        self.proteins = proteins

        # Map protein IDs to node indices
        self.protein_id_to_idx = {pid: i for i, pid in enumerate(proteins)}
        self.protein_idx_to_id = {v: k for k, v in self.protein_id_to_idx.items()}

        self.aa_idx_to_protein_pos = {}
        with h5py.File(self.h5_path, "r") as h5file:
            total_aa = 0
            for pid in self.proteins:
                length = h5file[pid]["embeddings"].shape[0] if pid in h5file else 0
                for pos in range(length):
                    self.aa_idx_to_protein_pos[total_aa + pos] = (pid, pos)
                total_aa += length

        # Build train/val/test masks
        self.train_mask = torch.zeros(len(proteins), dtype=torch.bool)
        self.val_mask = torch.zeros(len(proteins), dtype=torch.bool)
        self.test_mask = torch.zeros(len(proteins), dtype=torch.bool)

        # Train mask
        for pid in self.go_train_dict.keys():
            if pid in self.protein_id_to_idx:
                self.train_mask[self.protein_id_to_idx[pid]] = True

        # Include train annotations when predicting on val
        val_pids = set(self.go_val_dict.keys()).union(self.go_train_dict.keys())
        for pid in val_pids:
            if pid in self.protein_id_to_idx:
                self.val_mask[self.protein_id_to_idx[pid]] = True

        # Include train+val annotations when predicting on test
        test_pids = (
            set(self.go_test_dict.keys())
            .union(self.go_train_dict.keys())
            .union(self.go_val_dict.keys())
        )
        for pid in test_pids:
            if pid in self.protein_id_to_idx:
                self.test_mask[self.protein_id_to_idx[pid]] = True

        # Build vocab sizes (for label vector sizes)
        all_go_dicts = (
            list(self.go_train_dict.values())
            + list(self.go_val_dict.values())
            + list(self.go_test_dict.values())
        )
        self.go_vocab_size = all_go_dicts[0].shape[0] if all_go_dicts else 0

        self.ipr_vocab_size = (
            next(iter(self.interpro_dict.values())).shape[0]
            if self.interpro_dict
            else 0
        )

        # Build hetero Data base graph with nodes and edges
        self.data = self._build_hetero_data_base()
        self.data.validate(raise_on_error=True)

    @timeit
    def _load_go_annotations(self, go_tsv_path):
        """
        Load GO annotations from a TSV of format:
        EntryID   GO terms separated by ;
        Return dict: protein_id --> multi-hot tensor for GO terms
        """
        df = pd.read_csv(go_tsv_path, sep="\t")
        # Parse all unique GO terms
        all_terms = set()
        for terms_str in df["term"]:
            all_terms.update([t.strip() for t in terms_str.split(";")])
        all_terms = sorted(all_terms)
        go_to_idx = {go: i for i, go in enumerate(all_terms)}

        # Build multi-hot encoding per protein
        go_dict = {}
        for _, row in df.iterrows():
            pid = row["EntryID"]
            terms = [t.strip() for t in row["term"].split(";")]
            vec = torch.zeros(len(all_terms), dtype=torch.float32)
            for t in terms:
                vec[go_to_idx[t]] = 1.0
            go_dict[pid] = vec

        self.go_vocab_size = len(all_terms)
        return go_dict

    @timeit
    def _load_interpro_annotations(self, interpro_tsv_path):
        """
        Load InterPro annotations from TSV with columns ID, IPR (IPR ID)
        Returns dict protein_id -> multi-hot vector of interpro presence
        """
        df = pd.read_csv(interpro_tsv_path, sep="\t")
        all_ipr = sorted(df["IPR"].unique())
        ipr_to_idx = {ipr: i for i, ipr in enumerate(all_ipr)}

        grouped = df.groupby("ID")["IPR"].apply(list)
        interpro_dict = {}
        for pid, ipr_list in grouped.items():
            vec = torch.zeros(len(all_ipr), dtype=torch.float32)
            for ipr in ipr_list:
                vec[ipr_to_idx[ipr]] = 1.0
            interpro_dict[pid] = vec
        self.ipr_vocab_size = len(all_ipr)
        return interpro_dict

    @timeit
    def _build_hetero_data_base(self):
        data = HeteroData()
        num_proteins = len(self.proteins)
        data["protein"].num_nodes = num_proteins

        aa_counts = []
        protein_to_aa_start_idx = {}
        total_aa = 0
        with h5py.File(self.h5_path, "r") as h5file:
            for pid in self.proteins:
                length = h5file[pid]["embeddings"].shape[0] if pid in h5file else 0
                protein_to_aa_start_idx[pid] = total_aa
                aa_counts.append(length)
                total_aa += length

        data["aa"].num_nodes = total_aa

        src_aa = []
        dst_protein = []
        # Build compact per-aa arrays while iterating proteins
        aa_to_protein = []
        for prot_idx, pid in enumerate(self.proteins):
            start_idx = protein_to_aa_start_idx[pid]
            length = aa_counts[self.protein_id_to_idx[pid]]
            src_aa.extend(range(start_idx, start_idx + length))
            dst_protein.extend([prot_idx] * length)
            # per-aa arrays: protein index only (positions are sequential)
            aa_to_protein.extend([prot_idx] * length)

        edge_index_aa2protein = torch.tensor([src_aa, dst_protein], dtype=torch.long)
        data["aa", "aa2protein", "protein"].edge_index = edge_index_aa2protein

        source = self.alignment_df["protein1"].map(self.protein_id_to_idx).to_list()
        target = self.alignment_df["protein2"].map(self.protein_id_to_idx).to_list()
        edge_index_aligned = torch.tensor([source, target], dtype=torch.long)
        data["protein", "aligned_with", "protein"].edge_index = edge_index_aligned

        # Attach masks as attributes (for convenience)
        data["protein"].train_mask = self.train_mask
        data["protein"].val_mask = self.val_mask
        data["protein"].test_mask = self.test_mask

        return data

    @timeit
    def get_batch_features(self, batch, h5_file):
        batch["aa"].x = self.get_aa_features(batch, h5_file)
        go, interpro = self.get_function_features(batch)
        batch["protein"].go = go
        batch["protein"].interpro = interpro
        batch["protein"].x = batch["protein"].interpro  # Use InterPro as features
        return batch

    def get_aa_features(self, batch, h5_file):
        aa_embeddings = torch.zeros(batch["aa"].num_nodes, 1280, dtype=torch.float32)

        # Collect unique proteins and their AA indices/positions in the batch
        protein_to_aa_data = {}
        for local_idx, aa_global_idx in enumerate(batch["aa"].n_id.tolist()):
            protein_id, aa_pos = self.aa_idx_to_protein_pos[aa_global_idx]
            if protein_id not in protein_to_aa_data:
                protein_to_aa_data[protein_id] = []
            protein_to_aa_data[protein_id].append((local_idx, aa_pos))

        # Load embeddings per protein once and assign to AAs
        for protein_id, aa_list in protein_to_aa_data.items():
            if protein_id in h5_file:
                protein_embs = torch.from_numpy(
                    h5_file[protein_id]["embeddings"][:]
                )  # Load full array once
                for local_idx, aa_pos in aa_list:
                    aa_embeddings[local_idx] = protein_embs[aa_pos]

        return aa_embeddings

    @timeit
    def get_function_features(self, batch):
        interpro = torch.zeros(
            batch["protein"].num_nodes, self.ipr_vocab_size, dtype=torch.float32
        )
        go = torch.zeros(
            batch["protein"].num_nodes, self.go_vocab_size, dtype=torch.float32
        )

        for local_idx, protein_global_idx in enumerate(batch["protein"].n_id.tolist()):
            protein_id = self.protein_idx_to_id[protein_global_idx]
            ipr_features = self.interpro_dict.get(
                protein_id, torch.zeros(self.ipr_vocab_size)
            )
            interpro[local_idx] = ipr_features

            if self.split == "train":
                go_dict = self.go_train_dict
            elif self.split == "val":
                go_dict = self.go_val_dict
            else:
                go_dict = self.go_test_dict
            go_features = go_dict.get(protein_id, torch.zeros(self.go_vocab_size))
            go[local_idx] = go_features
        return go, interpro


def define_loaders(config, dataset):
    train_loader = NeighborLoader(
        dataset.data,
        input_nodes=("protein", dataset.data["protein"].train_mask),
        num_neighbors=[-1],  # Use all neighbors
        batch_size=config["model"]["batch_size"],
        shuffle=True,
        num_workers=4,
    )

    val_loader = NeighborLoader(
        dataset.data,
        input_nodes=("protein", dataset.data["protein"].val_mask),
        num_neighbors=[-1],  # Use all neighbors
        batch_size=config["model"]["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    test_loader = NeighborLoader(
        dataset.data,
        input_nodes=("protein", dataset.data["protein"].test_mask),
        num_neighbors=[-1],  # Use all neighbors
        batch_size=config["model"]["batch_size"],
        shuffle=False,
        num_workers=4,
    )
    return train_loader, val_loader, test_loader
