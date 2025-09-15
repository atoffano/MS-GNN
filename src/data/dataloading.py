import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
import pandas as pd
import numpy as np
import pickle
from src.utils import constants
from src.utils.helpers import timeit
import logging
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

import time


class ChunkedEmbeddingLoader:
    """Efficiently loads protein embeddings from chunked format."""

    def __init__(self, chunks_dir: str, cache_size: int = 5):
        """
        Args:
            chunks_dir: Directory containing chunked embedding files
            cache_size: Number of chunks to keep in memory cache
        """
        self.chunks_dir = Path(chunks_dir)
        self.cache_size = cache_size
        self._chunk_cache = {}
        self._cache_order = []

        # Load mapping data
        with open(f"{self.chunks_dir}/protein_mapping.pkl", "rb") as f:
            mapping_data = pickle.load(f)

        self.protein_to_chunk = mapping_data["protein_to_chunk"]
        self.chunk_metadata = mapping_data["chunk_metadata"]

        logger.info(
            f"Loaded chunked embeddings: {len(self.protein_to_chunk)} proteins in {len(self.chunk_metadata)} chunks"
        )

    def _load_chunk(self, chunk_idx: int) -> dict:
        """Load a specific chunk into memory."""
        if chunk_idx in self._chunk_cache:
            # Move to end of cache order (LRU)
            self._cache_order.remove(chunk_idx)
            self._cache_order.append(chunk_idx)
            return self._chunk_cache[chunk_idx]

        # Load chunk from disk
        chunk_path = f"{self.chunks_dir}/embeddings_chunk_{chunk_idx}.pt"
        chunk_data = torch.load(chunk_path, map_location="cpu")

        # Add to cache
        self._chunk_cache[chunk_idx] = chunk_data
        self._cache_order.append(chunk_idx)

        # Evict oldest chunk if cache is full
        if len(self._chunk_cache) > self.cache_size:
            oldest_chunk = self._cache_order.pop(0)
            del self._chunk_cache[oldest_chunk]

        logger.debug(f"Loaded chunk {chunk_idx} into cache")
        return chunk_data

    def get_protein_length(self, protein_id: str) -> int:
        """Get the sequence length for a protein without loading embeddings."""
        if protein_id not in self.protein_to_chunk:
            return 0

        chunk_idx = self.protein_to_chunk[protein_id]
        chunk_info = self.chunk_metadata[str(chunk_idx)]

        try:
            protein_idx = chunk_info["proteins"].index(protein_id)
            return chunk_info["amino_acid_counts"][protein_idx]
        except (ValueError, KeyError):
            return 0

    def get_embeddings_batch(self, protein_ids: list) -> dict:
        """Get embeddings for a batch of proteins efficiently."""
        # Group proteins by chunk
        chunk_groups = defaultdict(list)
        for protein_id in protein_ids:
            if protein_id in self.protein_to_chunk:
                chunk_idx = self.protein_to_chunk[protein_id]
                chunk_groups[chunk_idx].append(protein_id)

        # Load embeddings
        embeddings = {}
        for chunk_idx, chunk_proteins in chunk_groups.items():
            chunk_data = self._load_chunk(chunk_idx)
            for protein_id in chunk_proteins:
                if protein_id in chunk_data:
                    embeddings[protein_id] = chunk_data[protein_id]

        return embeddings


class SwissProtHeteroDataset:
    def __init__(self, datapath, split):
        logger.info("Initializing SwissProtHeteroDataset...")
        logger.debug(f"Datapath: {datapath}, Split: {split}")

        # Load GO annotations
        logger.info("Loading GO annotations...")
        self.go_train_dict = self._load_go_annotations(datapath["train"])
        self.go_val_dict = self._load_go_annotations(datapath["val"])
        self.go_test_dict = self._load_go_annotations(datapath["test"])
        logger.info(
            f"Loaded GO annotations: Train={len(self.go_train_dict)}, Val={len(self.go_val_dict)}, Test={len(self.go_test_dict)}"
        )

        self.split = split

        chunks_dir = datapath["prot_emb_path"]

        logger.info(f"Loading chunked embeddings from: {chunks_dir}")
        self.embedding_loader = ChunkedEmbeddingLoader(chunks_dir, cache_size=10)

        # Load InterPro annotations
        logger.info("Loading InterPro annotations...")
        self.interpro_dict = self._load_interpro_annotations(datapath["interpro"])
        logger.info(
            f"Loaded InterPro annotations for {len(self.interpro_dict)} proteins"
        )

        # Load alignment info
        logger.info("Loading alignment data...")
        self.alignment_df = pd.read_csv(
            datapath["alignments"],
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
        logger.info(f"Loaded alignment data with {len(self.alignment_df)} entries")

        # Collect all proteins
        logger.info("Collecting all proteins from annotations and alignments...")
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
        logger.info(f"Total unique proteins: {len(proteins)}")

        # Map protein IDs to indices
        self.protein_id_to_idx = {pid: i for i, pid in enumerate(proteins)}
        self.protein_idx_to_id = {v: k for k, v in self.protein_id_to_idx.items()}
        logger.debug("Built protein ID to index mappings")

        # Build masks
        logger.info("Building train/val/test masks...")
        self.train_mask = torch.zeros(len(proteins), dtype=torch.bool)
        self.val_mask = torch.zeros(len(proteins), dtype=torch.bool)
        self.test_mask = torch.zeros(len(proteins), dtype=torch.bool)

        for pid in self.go_train_dict.keys():
            if pid in self.protein_id_to_idx:
                self.train_mask[self.protein_id_to_idx[pid]] = True
        logger.debug(f"Train mask set for {self.train_mask.sum().item()} proteins")

        val_pids = set(self.go_val_dict.keys()).union(self.go_train_dict.keys())
        for pid in val_pids:
            if pid in self.protein_id_to_idx:
                self.val_mask[self.protein_id_to_idx[pid]] = True
        logger.debug(f"Val mask set for {self.val_mask.sum().item()} proteins")

        test_pids = (
            set(self.go_test_dict.keys())
            .union(self.go_train_dict.keys())
            .union(self.go_val_dict.keys())
        )
        for pid in test_pids:
            if pid in self.protein_id_to_idx:
                self.test_mask[self.protein_id_to_idx[pid]] = True
        logger.debug(f"Test mask set for {self.test_mask.sum().item()} proteins")

        # Build vocab sizes
        logger.info("Building vocabulary sizes...")
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
        logger.info(
            f"GO vocab size: {self.go_vocab_size}, InterPro vocab size: {self.ipr_vocab_size}"
        )

        # Build hetero data
        logger.info("Building heterogeneous data graph...")
        self.data = self._build_hetero_data_base()
        logger.info("Validating heterogeneous data...")
        self.data.validate(raise_on_error=True)
        logger.info("Dataset initialization complete!")

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

        # Build AA mappings using chunked loader
        aa_counts = []
        self.protein_to_aa_start_idx = {}
        total_aa = 0

        logger.info("Building AA mappings from chunked embeddings...")
        for pid in self.proteins:
            length = self.embedding_loader.get_protein_length(pid)
            self.protein_to_aa_start_idx[pid] = total_aa
            aa_counts.append(length)
            total_aa += length

        logger.info(f"Total AA nodes: {total_aa}")
        data["aa"].num_nodes = total_aa

        # Build edges
        src_aa = []
        dst_protein = []
        for prot_idx, pid in enumerate(self.proteins):
            start_idx = self.protein_to_aa_start_idx[pid]
            length = aa_counts[prot_idx]
            src_aa.extend(range(start_idx, start_idx + length))
            dst_protein.extend([prot_idx] * length)

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
    def get_batch_features(self, batch):
        batch["aa"].x = self.get_aa_features(batch)
        go, interpro = self.get_function_features(batch)
        batch["protein"].go = go
        batch["protein"].interpro = interpro
        batch["protein"].x = batch["protein"].interpro  # Use InterPro as features
        return batch

    def get_aa_features(self, batch):
        aa_embeddings = torch.zeros(batch["aa"].num_nodes, 1280, dtype=torch.float32)

        # Get AA-to-protein edge index to map AA local indices to protein local indices
        edge_index = batch["aa", "aa2protein", "protein"].edge_index
        aa_to_protein_local = {}  # Map AA local idx to protein local idx
        for aa_local, prot_local in zip(edge_index[0], edge_index[1]):
            aa_to_protein_local[aa_local.item()] = prot_local.item()

        # Group AA local indices by protein and compute positions
        protein_to_aa_data = {}
        for aa_local_idx in range(batch["aa"].num_nodes):
            if aa_local_idx in aa_to_protein_local:
                prot_local_idx = aa_to_protein_local[aa_local_idx]
                prot_global_idx = batch["protein"].n_id[prot_local_idx].item()
                protein_id = self.protein_idx_to_id[prot_global_idx]
                aa_global_idx = batch["aa"].n_id[aa_local_idx].item()
                # Compute position on the fly: global AA idx - protein's start idx
                aa_pos = aa_global_idx - self.protein_to_aa_start_idx.get(protein_id, 0)
                protein_to_aa_data.setdefault(protein_id, []).append(
                    (aa_local_idx, aa_pos)
                )

        # Get unique protein IDs for batch loading
        protein_ids = list(protein_to_aa_data.keys())

        # Load embeddings for all proteins in this batch using chunked loader
        protein_embeddings = self.embedding_loader.get_embeddings_batch(protein_ids)

        # Assign embeddings to AA nodes
        for protein_id, aa_list in protein_to_aa_data.items():
            if protein_id in protein_embeddings:
                protein_embs = protein_embeddings[protein_id]
                for aa_local_idx, aa_pos in aa_list:
                    if aa_pos < len(protein_embs):
                        aa_embeddings[aa_local_idx] = protein_embs[aa_pos].to(
                            torch.float32
                        )

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
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = NeighborLoader(
        dataset.data,
        input_nodes=("protein", dataset.data["protein"].val_mask),
        num_neighbors=[-1],  # Use all neighbors
        batch_size=config["model"]["batch_size"],
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader = NeighborLoader(
        dataset.data,
        input_nodes=("protein", dataset.data["protein"].test_mask),
        num_neighbors=[-1],  # Use all neighbors
        batch_size=config["model"]["batch_size"],
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, val_loader, test_loader
