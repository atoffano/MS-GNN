"""Lightweight inference dataset replacing SwissProtDataset.

InferenceDataset builds a minimal heterogeneous protein-protein graph
containing only the query protein(s) and their DIAMOND/STRING-DB neighbors.
It exposes the same attributes and methods that ProteinGNN and
get_batch_features() rely on, enabling seamless model reuse.

Key differences from SwissProtDataset:
- Contains only query + neighbor proteins (not the full ~570K SwissProt).
- Builds edges from per-protein DIAMOND output + local STRING-DB file.
- Uses precomputed edge normalization constants (from edge_normalization.json).
- All split masks are zero (no training/validation/test logic).
- GO features for neighbors are loaded from SwissProt annotation files.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

from src.utils.constants import (
    CONTACT_CUTOFF,
    GO_ANNOTATION_TEMPLATE,
    GO_EXP_ANNOTATION_TEMPLATE,
    GO_VOCAB,
    INTERPRO_VOCAB,
    PID_MAPPING,
    PROJECT_ROOT,
    PROTEIN_GRAPHS_DIR,
    USES_ENTRYID,
)
from src.utils.inference.resolve_ids import ProteinInfo

logger = logging.getLogger(__name__)

# Module-level caches for fast repeated dataset instantiation
_IPR_VOCAB_CACHE: Dict[Path, int] = {}
_GO_VOCAB_CACHE: Dict[Path, dict] = {}
_PID_MAPPING_CACHE: Dict[Path, Tuple[Dict[str, str], Dict[str, str]]] = {}
_PROTEIN_GRAPH_CACHE: Dict[Tuple[Path, Optional[Path], str], Optional[HeteroData]] = {}


def get_ipr_vocab_size(ipr_vocab_path: Path = INTERPRO_VOCAB) -> int:
    """Load and cache InterPro vocabulary size."""
    p = Path(ipr_vocab_path)
    if p not in _IPR_VOCAB_CACHE:
        with open(p, "rb") as f:
            _IPR_VOCAB_CACHE[p] = pickle.load(f)["vocab_size"]
    return _IPR_VOCAB_CACHE[p]


def get_go_vocab_info(go_vocab_path: Path = GO_VOCAB) -> dict:
    """Load and cache Gene Ontology vocabulary info."""
    p = Path(go_vocab_path)
    if p not in _GO_VOCAB_CACHE:
        with open(p, "rb") as f:
            _GO_VOCAB_CACHE[p] = pickle.load(f)
    return _GO_VOCAB_CACHE[p]


def get_pid_mapping(pid_mapping_path: Path = PID_MAPPING) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load and cache SwissProt ID mappings."""
    p = Path(pid_mapping_path)
    if p not in _PID_MAPPING_CACHE:
        if p.exists():
            df = pd.read_csv(
                p, sep="\t", usecols=["EntryID", "Entry Name"]
            )
            pid_map = df.set_index("EntryID")["Entry Name"].to_dict()
            rev_map = {v: k for k, v in pid_map.items()}
            _PID_MAPPING_CACHE[p] = (pid_map, rev_map)
        else:
            _PID_MAPPING_CACHE[p] = ({}, {})
    return _PID_MAPPING_CACHE[p]


class InferenceDataset:
    """Lightweight dataset for inference with pre-trained MS-GNN models.

    Maintains API compatibility with SwissProtDataset so that ProteinGNN,
    get_batch_features(), and NeighborLoader work unchanged.

    Args:
        query_proteins: List of resolved ProteinInfo for query proteins.
        config: Model configuration dictionary (from cfg.yaml).
        edge_norm: Edge normalization constants from edge_normalization.json.
                   Format: {"aligned_with": {"mean": ..., "std": ...}, ...}
        neighbor_info: Dict mapping neighbor accession → {
            "bitscore": float | None,
            "combined_score": float | None,
            "has_pt": bool,
        }
        graphs_dir: Directory containing .pt graph files.
        target_graphs_dir: Directory containing newly built target .pt files.
        go_vocab_path: Path to go_vocab.pkl.
        ipr_vocab_path: Path to interpro_vocab.pkl.
        pid_mapping_path: Path to swissprot annotations TSV for ID mapping.
    """

    def __init__(
        self,
        query_proteins: List[ProteinInfo],
        config: dict,
        edge_norm: dict,
        neighbor_info: Optional[Dict[str, dict]] = None,
        per_query_neighbors: Optional[Dict[str, Dict[str, dict]]] = None,
        pairwise_edges: Optional[List[Tuple[str, str, str, float]]] = None,
        graphs_dir: Path = PROTEIN_GRAPHS_DIR,
        target_graphs_dir: Optional[Path] = None,
        go_vocab_path: Path = GO_VOCAB,
        ipr_vocab_path: Path = INTERPRO_VOCAB,
        pid_mapping_path: Path = PID_MAPPING,
        external_annotations: Optional[Dict[str, List[str]]] = None,
        id_mapping: Optional[Tuple[Dict[str, str], Dict[str, str]]] = None,
        mask_query_neighbors: bool = False,
    ):
        self.config = config
        self.edge_norm = edge_norm
        self.graphs_dir = Path(graphs_dir)
        self.target_graphs_dir = Path(target_graphs_dir) if target_graphs_dir else None
        self.external_annotations = external_annotations
        self.mask_query_neighbors = mask_query_neighbors

        # ---- Dataset identity ----
        dataset_name = config["data"].get("dataset", "D1")
        self.uses_entryid = dataset_name in USES_ENTRYID

        # ---- ID mappings ----
        if id_mapping is not None:
            self.pid_mapping, self.rev_pid_mapping = id_mapping
        else:
            self.pid_mapping, self.rev_pid_mapping = get_pid_mapping(pid_mapping_path)

        # ---- Vocabularies ----
        self.ipr_vocab_size = get_ipr_vocab_size(ipr_vocab_path)
        self.go_vocab_info = get_go_vocab_info(go_vocab_path)
        self.go_vocab_sizes = {
            onto: info["vocab_size"] for onto, info in self.go_vocab_info.items()
        }
        self.subontology = config["data"]["subontology"]
        self.go_vocab_size = self.go_vocab_sizes[self.subontology]

        # ---- Build protein list ----
        # Query proteins first (they will be seed nodes)
        query_accessions = {p.accession for p in query_proteins}

        # Resolve per-query neighbors mapping (isolated per query)
        if per_query_neighbors is None:
            if neighbor_info is not None:
                per_query_neighbors = {p.accession: neighbor_info for p in query_proteins}
            else:
                per_query_neighbors = {p.accession: {} for p in query_proteins}
        self.per_query_neighbors = per_query_neighbors
        self.pairwise_edges = pairwise_edges or []

        # Neighbors with available .pt files
        neighbor_accessions: Set[str] = set()
        self._neighbor_edges: Dict[str, dict] = {}  # acc → edge info
        for q_acc, q_neighbors in self.per_query_neighbors.items():
            for acc, info in q_neighbors.items():
                if info.get("has_pt", False) and acc not in query_accessions:
                    neighbor_accessions.add(acc)
                    self._neighbor_edges[acc] = info

        # Include any additional neighbor nodes referenced in pairwise induced edges
        for p1, p2, _, _ in self.pairwise_edges:
            for acc in (p1, p2):
                if acc not in query_accessions and acc not in neighbor_accessions:
                    if (self.graphs_dir / f"{acc}.pt").exists() or (
                        self.target_graphs_dir and (self.target_graphs_dir / f"{acc}.pt").exists()
                    ):
                        neighbor_accessions.add(acc)

        # All proteins in the graph
        self.proteins: List[str] = []
        for p in query_proteins:
            pid = p.accession
            if self.uses_entryid:
                pid = self.rev_pid_mapping.get(p.accession, p.accession)
            self.proteins.append(pid)

        for acc in sorted(neighbor_accessions):
            pid = acc
            if self.uses_entryid:
                pid = self.rev_pid_mapping.get(acc, acc)
            self.proteins.append(pid)

        self.protein_to_idx = {pid: i for i, pid in enumerate(self.proteins)}
        self.idx_to_protein = {v: k for k, v in self.protein_to_idx.items()}

        num_proteins = len(self.proteins)
        logger.info(
            "InferenceDataset: %d proteins (%d query, %d neighbors)",
            num_proteins,
            len(query_accessions),
            len(neighbor_accessions),
        )

        # ---- Masks (all zeros — no training) ----
        self.train_mask = torch.zeros(num_proteins, dtype=torch.bool)
        self.val_mask = torch.zeros(num_proteins, dtype=torch.bool)
        self.test_mask = torch.zeros(num_proteins, dtype=torch.bool)

        # Seed mask: only query proteins
        self._query_mask = torch.zeros(num_proteins, dtype=torch.bool)
        for p in query_proteins:
            pid = p.accession
            if self.uses_entryid:
                pid = self.rev_pid_mapping.get(p.accession, p.accession)
            if pid in self.protein_to_idx:
                self._query_mask[self.protein_to_idx[pid]] = True

        # ---- Load GO annotations for neighbors ----
        if self.external_annotations is None:
            self._load_neighbor_annotations(query_accessions)

        # ---- Build protein-protein graph ----
        self.data = self._create_protein_graph(
            query_proteins, self.per_query_neighbors, self.pairwise_edges
        )
        self.transform = T.Compose(
            [
                T.RemoveDuplicatedEdges(key="edge_attr", reduce="mean"),
                T.ToUndirected(reduce="max"),
                T.AddRemainingSelfLoops(attr="edge_attr", fill_value=1.0),
            ]
        )
        self.data = self.transform(self.data)

        logger.info(
            "InferenceDataset ready: %d protein nodes in graph.",
            self.data["protein"].num_nodes,
        )

    def _load_neighbor_annotations(self, query_accessions: Set[str]) -> None:
        """Load GO annotations for neighbor proteins from SwissProt annotation files.

        These annotations are used as neighbor protein features (go_neighbors).
        Query proteins always get zeroed GO features.
        """
        ext_annot_path = self.config["data"].get("external_annotations_path")
        annot_path = None
        if ext_annot_path:
            ext_p = Path(str(ext_annot_path).format(onto=self.subontology))
            if not ext_p.is_absolute():
                ext_p = (PROJECT_ROOT / ext_p).resolve()
            if ext_p.exists():
                annot_path = ext_p
                logger.info("Using configured external_annotations_path: %s", annot_path)
        if annot_path is None:
            exp_only = self.config["data"].get("exp_only", True)
            annot_template = GO_EXP_ANNOTATION_TEMPLATE if exp_only else GO_ANNOTATION_TEMPLATE
            annot_path = annot_template.with_name(annot_template.name.format(onto=self.subontology))

        if not annot_path.exists():
            logger.warning(
                "GO annotation file not found: %s. Neighbors will have empty GO features.",
                annot_path,
            )
            return

        logger.info("Loading neighbor GO annotations from %s", annot_path)
        annot_df = pd.read_csv(annot_path, sep="\t")
        self.external_annotations = {}
        for acc_val, term_val in zip(annot_df["EntryID"].astype(str), annot_df["term"].astype(str)):
            acc = acc_val.strip()
            if self.mask_query_neighbors and acc in query_accessions:
                continue  # Only mask query neighbors if explicitly requested
            terms = [t.strip() for t in term_val.split(";") if t.strip()]
            self.external_annotations[acc] = terms
            if self.rev_pid_mapping and acc in self.rev_pid_mapping:
                self.external_annotations[self.rev_pid_mapping[acc]] = terms
            if self.pid_mapping and acc in self.pid_mapping:
                self.external_annotations[self.pid_mapping[acc]] = terms

        logger.info(
            "Loaded GO annotations for %d neighbor proteins (dual-indexed).",
            len(self.external_annotations),
        )

    def _create_protein_graph(
        self,
        query_proteins: List[ProteinInfo],
        per_query_neighbors: Dict[str, Dict[str, dict]],
        pairwise_edges: Optional[List[Tuple[str, str, str, float]]] = None,
    ) -> HeteroData:
        """Build the protein-protein heterogeneous graph.

        Creates aligned_with and stringdb edges between proteins using
        precomputed normalization constants from edge_normalization.json.
        Each query is wired only to its own neighbors, and induced pairwise
        edges between neighbors are added.

        Args:
            query_proteins: List of query ProteinInfo.
            per_query_neighbors: Mapping from query accession → neighbor info.
            pairwise_edges: List of (acc_a, acc_b, edge_type, score) tuples.

        Returns:
            HeteroData with protein-protein edges.
        """
        data = HeteroData()
        data["protein"].num_nodes = len(self.proteins)

        config = self.config
        pairwise_edges = pairwise_edges or []

        # ---- aligned_with edges ----
        if ["protein", "aligned_with", "protein"] in config["model"]["edge_types"]:
            src_indices = []
            dst_indices = []
            bitscores = []

            # 1. Query-neighbor edges (and query self-loops if present)
            for query_p in query_proteins:
                q_pid = query_p.accession
                if self.uses_entryid:
                    q_pid = self.rev_pid_mapping.get(query_p.accession, query_p.accession)
                if q_pid not in self.protein_to_idx:
                    continue
                q_idx = self.protein_to_idx[q_pid]

                q_neighbors = per_query_neighbors.get(query_p.accession, {})
                for acc, info in q_neighbors.items():
                    if info.get("bitscore") is None:
                        continue
                    n_pid = acc
                    if self.uses_entryid:
                        n_pid = self.rev_pid_mapping.get(acc, acc)
                    if n_pid not in self.protein_to_idx:
                        continue
                    n_idx = self.protein_to_idx[n_pid]

                    src_indices.append(q_idx)
                    dst_indices.append(n_idx)
                    bitscores.append(info["bitscore"])

            # 2. Pairwise induced edges for aligned_with
            for edge in pairwise_edges:
                p1, p2, edge_type, score = edge
                if edge_type != "aligned_with":
                    continue
                pid1 = self.rev_pid_mapping.get(p1, p1) if self.uses_entryid else p1
                pid2 = self.rev_pid_mapping.get(p2, p2) if self.uses_entryid else p2
                if pid1 in self.protein_to_idx and pid2 in self.protein_to_idx:
                    src_indices.append(self.protein_to_idx[pid1])
                    dst_indices.append(self.protein_to_idx[pid2])
                    bitscores.append(score)

            if src_indices:
                edge_index = torch.tensor(
                    [src_indices, dst_indices], dtype=torch.long
                )
                data["protein", "aligned_with", "protein"].edge_index = edge_index

                if config["model"]["edge_attrs"]:
                    raw = torch.tensor(bitscores, dtype=torch.float32).unsqueeze(1)
                    norm = self.edge_norm.get("aligned_with", {})
                    mean = norm.get("mean", 0.0)
                    std = norm.get("std", 1.0)
                    normalized = (raw - mean) / std
                    data["protein", "aligned_with", "protein"].edge_attr = normalized

                logger.info("aligned_with edges: %d", len(src_indices))
            else:
                data["protein", "aligned_with", "protein"].edge_index = torch.empty(
                    (2, 0), dtype=torch.long
                )
                if config["model"]["edge_attrs"]:
                    data["protein", "aligned_with", "protein"].edge_attr = torch.empty(
                        (0, 1), dtype=torch.float32
                    )

        # ---- stringdb edges ----
        if ["protein", "stringdb", "protein"] in config["model"]["edge_types"]:
            src_indices = []
            dst_indices = []
            scores = []

            # 1. Query-neighbor edges
            for query_p in query_proteins:
                q_pid = query_p.accession
                if self.uses_entryid:
                    q_pid = self.rev_pid_mapping.get(query_p.accession, query_p.accession)
                if q_pid not in self.protein_to_idx:
                    continue
                q_idx = self.protein_to_idx[q_pid]

                q_neighbors = per_query_neighbors.get(query_p.accession, {})
                for acc, info in q_neighbors.items():
                    if info.get("combined_score") is None:
                        continue
                    n_pid = acc
                    if self.uses_entryid:
                        n_pid = self.rev_pid_mapping.get(acc, acc)
                    if n_pid not in self.protein_to_idx:
                        continue
                    n_idx = self.protein_to_idx[n_pid]

                    src_indices.append(q_idx)
                    dst_indices.append(n_idx)
                    scores.append(info["combined_score"])

            # 2. Pairwise induced edges for stringdb
            for edge in pairwise_edges:
                p1, p2, edge_type, score = edge
                if edge_type != "stringdb":
                    continue
                pid1 = self.rev_pid_mapping.get(p1, p1) if self.uses_entryid else p1
                pid2 = self.rev_pid_mapping.get(p2, p2) if self.uses_entryid else p2
                if pid1 in self.protein_to_idx and pid2 in self.protein_to_idx:
                    src_indices.append(self.protein_to_idx[pid1])
                    dst_indices.append(self.protein_to_idx[pid2])
                    scores.append(score)

            if src_indices:
                edge_index = torch.tensor(
                    [src_indices, dst_indices], dtype=torch.long
                )
                data["protein", "stringdb", "protein"].edge_index = edge_index

                if config["model"]["edge_attrs"]:
                    raw = torch.tensor(scores, dtype=torch.float32).unsqueeze(1)
                    norm = self.edge_norm.get("stringdb", {})
                    mean = norm.get("mean", 0.0)
                    std = norm.get("std", 1.0)
                    normalized = (raw - mean) / std
                    data["protein", "stringdb", "protein"].edge_attr = normalized

                logger.info("stringdb edges: %d", len(src_indices))
            else:
                data["protein", "stringdb", "protein"].edge_index = torch.empty(
                    (2, 0), dtype=torch.long
                )
                if config["model"]["edge_attrs"]:
                    data["protein", "stringdb", "protein"].edge_attr = torch.empty(
                        (0, 1), dtype=torch.float32
                    )

        return data

    def load_protein_graph(self, protein_id: str) -> Optional[HeteroData]:
        """Load a single protein graph from disk with in-memory caching.

        Checks both the standard graphs_dir and the target_graphs_dir
        (for newly preprocessed query proteins).

        Args:
            protein_id: Protein identifier (accession number).

        Returns:
            HeteroData graph or None if not found.
        """
        cache_key = (self.graphs_dir, self.target_graphs_dir, protein_id)
        if cache_key in _PROTEIN_GRAPH_CACHE:
            return _PROTEIN_GRAPH_CACHE[cache_key]

        # Try target graphs dir first (newly built query protein graphs)
        if self.target_graphs_dir is not None:
            target_path = self.target_graphs_dir / f"{protein_id}.pt"
            if target_path.exists():
                try:
                    graph = torch.load(
                        target_path, map_location="cpu", weights_only=False
                    )
                    _PROTEIN_GRAPH_CACHE[cache_key] = graph
                    return graph
                except Exception as e:
                    logger.warning("Failed to load %s: %s", target_path, e)

        # Try standard graphs dir
        std_path = self.graphs_dir / f"{protein_id}.pt"
        if std_path.exists():
            try:
                graph = torch.load(std_path, map_location="cpu", weights_only=False)
                _PROTEIN_GRAPH_CACHE[cache_key] = graph
                return graph
            except Exception as e:
                logger.warning("Failed to load %s: %s", std_path, e)

        logger.warning("Graph not found for protein %s", protein_id)
        _PROTEIN_GRAPH_CACHE[cache_key] = None
        return None

    def _terms_to_onehot(self, terms: List[str]) -> torch.Tensor:
        """Convert a list of GO terms to one-hot encoding.

        Args:
            terms: List of GO term IDs.

        Returns:
            One-hot tensor of shape (go_vocab_size,).
        """
        go_to_idx = self.go_vocab_info[self.subontology]["go_to_idx"]
        onehot = torch.zeros(self.go_vocab_size, dtype=torch.float32)
        for term in terms:
            if term in go_to_idx:
                onehot[go_to_idx[term]] = 1.0
        return onehot

    def convert_go_terms_to_onehot(self, go_terms_dict: dict) -> torch.Tensor:
        """Convert GO terms dict to one-hot encoding based on config.

        Args:
            go_terms_dict: Dict with "experimental" and "curated" keys.

        Returns:
            One-hot tensor of shape (go_vocab_size,).
        """
        ext_path = self.config["data"].get("external_annotations_path")
        if ext_path and "exp_" not in str(ext_path):
            terms = go_terms_dict.get("curated", [])
        elif self.config["data"].get("exp_only", True):
            terms = go_terms_dict.get("experimental", [])
        else:
            terms = go_terms_dict.get("curated", [])
        return self._terms_to_onehot(terms)

    @property
    def query_mask(self) -> torch.Tensor:
        """Boolean mask indicating query proteins in the graph."""
        return self._query_mask
