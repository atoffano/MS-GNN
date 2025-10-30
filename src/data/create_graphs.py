"""Heterogeneous graph construction for protein function prediction.

This module creates individual protein graphs combining multiple data sources:
- Protein sequences and metadata
- ESM embeddings for amino acid residues
- 3D structure information (AlphaFold/ESMFold)
- InterPro domain annotations
- Gene Ontology term annotations (MFO, BPO, CCO)

The resulting graphs are heterogeneous with protein and amino acid nodes,
connected through various edge types including spatial contacts and protein
membership relationships.
"""

import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import pickle
import torch
from Bio import SeqIO
from Bio.PDB import PDBParser
from torch_geometric.data import HeteroData
from tqdm.auto import tqdm

from src.utils.constants import (
    SWISSPROT_ROOT,
    PROTEIN_GRAPHS_DIR,
    SWISSPROT_FASTA,
    INTERPRO_TSV,
    GO_OBO_PATH,
    GO_ANNOTATION_TEMPLATE,
    GO_EXP_ANNOTATION_TEMPLATE,
    EMBED_H5_PATH,
    STRUCTURE_MISSING_PATH,
    CONTACT_CUTOFF,
    CONTACT_CHUNK_SIZE,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Create output directory if it doesn't exist
PROTEIN_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
STRUCTURE_MISSING_PATH.touch(exist_ok=True)


def build_and_save_protein_graph(
    protein_id: str,
    sequence: str,
    h5f: h5py.File,
    interpro_dict: Dict[str, torch.Tensor],
    go_annotations: Dict[str, Dict[str, Dict[str, List[str]]]],
    interpro_vocab_size: int,
) -> Tuple[bool, Optional[str], bool]:
    """Build and save a heterogeneous protein graph to disk.

    Args:
        protein_id: Unique protein identifier
        sequence: Amino acid sequence
        h5f: Open HDF5 file containing ESM embeddings
        interpro_dict: Dictionary mapping protein IDs to InterPro annotations
        go_annotations: Nested dict of GO term annotations by ontology
        interpro_vocab_size: Size of InterPro vocabulary

    Returns:
        Tuple of (success, error_message, embeddings_missing)
    """
    # # Check if graph already exists with valid contact edges
    # existing_path = PROTEIN_GRAPHS_DIR / f"{protein_id}.pt"
    # if existing_path.is_file():
    #     try:
    #         existing_data = torch.load(existing_path)
    #         if existing_data["aa", "close_to", "aa"].edge_index.size(1) > 0:
    #             return True, None, False
    #     except Exception as exc:
    #         logger.warning(
    #             "Failed to load existing graph for %s: %s. Rebuilding.", protein_id, exc
    #         )

    if protein_id not in h5f:
        return False, "Embeddings missing in H5", True

    try:
        embeddings_np = h5f[protein_id]["embeddings"][:]
        embeddings = torch.from_numpy(embeddings_np)

        data = build_protein_graph(
            protein_id,
            embeddings,
            sequence,
            interpro_dict,
            go_annotations,
            interpro_vocab_size,
        )
        if data is None:
            return False, "Graph construction returned None", False

        # Try to add close contact edges - returns False if protein should be skipped
        if not add_close_contact_edges(data, protein_id):
            return False, "Sequence/structure mismatch (not truncated)", False

        torch.save(data, PROTEIN_GRAPHS_DIR / f"{protein_id}.pt")
        return True, None, False
    except Exception as exc:
        return False, str(exc), False


def count_ca_atoms(pdb_path: Path) -> int:
    """Count the number of C-alpha atoms in a PDB file.

    Args:
        pdb_path: Path to PDB structure file

    Returns:
        Number of C-alpha atoms found in the structure
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))

    ca_count = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ca_count += 1

    return ca_count


def load_ca_coordinates(pdb_path: Path, sequence_length: int) -> torch.Tensor:
    """Load C-alpha atom coordinates from a PDB file.

    This function creates a coordinate tensor aligned with the protein sequence,
    Residues missing from the PDB structure are marked with NaN.

    Example:
    ---------------
    Sequence: X A P Q N X M N G L  (10 residues, indices 0-9)
    PDB file contains residues: 2, 3, 4, 5, 7, 8, 9, 10 (missing positions 1 and 6)

    Result tensor (10 x 3):
    Index:  0     1      2      3      4     5     6      7      8      9
    Coord: NaN  CA_2   CA_3   CA_4   CA_5  NaN   CA_7   CA_8   CA_9   CA_10
           ↑     ↑      ↑      ↑      ↑     ↑     ↑      ↑      ↑      ↑
          Pos1  Pos2   Pos3   Pos4   Pos5  Pos6  Pos7   Pos8   Pos9   Pos10
          (X)   (A)    (P)    (Q)    (N)   (X)   (M)    (N)    (G)    (L)

    This ensures that edges computed from coordinates correctly reference
    the amino acid nodes in the graph at their true sequence positions.

    Args:
        pdb_path: Path to PDB structure file
        sequence_length: Expected length of the full protein sequence

    Returns:
        Tensor of C-alpha coordinates with NaN for missing residues,
        shape (sequence_length, 3)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))

    coords = np.full((sequence_length, 3), np.nan, dtype=np.float32)

    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" not in residue:
                    continue

                # Extract residue number from PDB (1-based numbering) and convert to 0-based
                res_num = residue.id[1]
                idx = res_num - 1
                if 0 <= idx < sequence_length:
                    coords[idx] = residue["CA"].get_coord()

    if np.all(np.isnan(coords)):
        raise ValueError(f"No valid CA atoms found in {pdb_path}")

    return torch.from_numpy(coords)


def build_close_contact_edges(
    coords: torch.Tensor,
    cutoff: float = CONTACT_CUTOFF,
    chunk_size: int = CONTACT_CHUNK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build edges between spatially close amino acid residues.

    Process:
    --------
    1. Identify valid (non-NaN) residue positions
    2. Compute pairwise distances only among valid residues
    3. Find pairs within cutoff distance
    4. Map edge indices back to original sequence positions

    Example:
    --------
    Input coords shape: (10, 3) with NaN at positions [0, 5]
    Valid positions: [1, 2, 3, 4, 6, 7, 8, 9] (8 residues)

    Distance computation: 8x8 matrix among valid residues
    If residue at index 3 is close to residue at index 7:
      - Local indices in valid array: 2 → 5
      - Global indices in sequence: 3 → 7
      - Edge created: (3, 7) ← Uses original sequence positions!

    Args:
        coords: C-alpha coordinates tensor with NaN for missing residues,
                shape (n_residues, 3)
        cutoff: Distance threshold in Angstroms for defining contacts
        chunk_size: Chunk size for memory-efficient distance computation

    Returns:
        Tuple of (edge_index, edge_distances) where edge_index contains
        indices referring to the original sequence positions
    """
    # Identify which residues have valid coordinates (not NaN)
    # Check only the first coordinate (x) since if one is NaN, all three are
    valid_mask = ~torch.isnan(coords[:, 0])
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).view(-1)

    # Handle edge cases: no residues or only one residue
    if valid_indices.numel() <= 1:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )

    # Extract only the valid (non-NaN) coordinates for distance computation
    valid_coords = coords[valid_indices]
    n = valid_coords.size(0)

    edges: List[Tuple[int, int]] = []
    dists: List[float] = []

    # Process in chunks to manage memory for large proteins
    for start in range(0, n, chunk_size):
        # Get a chunk of source residues
        sub = valid_coords[start : start + chunk_size]

        # Compute pairwise distances: sub rows × all valid_coords columns
        dist_sq = torch.cdist(sub, valid_coords, p=2)

        # For each residue in the chunk
        for i in range(sub.size(0)):
            # Find residues within cutoff distance (excluding self with dist > 0)
            mask = (dist_sq[i] <= cutoff) & (dist_sq[i] > 0)
            cols = torch.nonzero(mask, as_tuple=False).view(-1)

            if cols.numel() == 0:
                continue

            # Map from local chunk index to local valid array index
            src_local = start + i

            # Map from local valid array index to global sequence index
            src_global = int(valid_indices[src_local])

            # Create edges using global sequence indices
            for col in cols:
                dst_global = int(valid_indices[col])
                edges.append((src_global, dst_global))
                dists.append(float(dist_sq[i, col].sqrt().item()))

    # Return empty tensors if no contacts found
    if not edges:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )

    # Convert to PyTorch Geometric format: [2, num_edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(dists, dtype=torch.float32)
    return edge_index, edge_attr


def load_interpro_annotations() -> Tuple[Dict[str, torch.Tensor], int]:
    """Load and vectorize InterPro domain annotations.

    Returns:
        Tuple of (protein_to_interpro_vector_dict, interpro_vocab_size)
    """
    df = pd.read_csv(INTERPRO_TSV, sep="\t")
    ipr_ids = sorted(df["IPR"].unique())
    ipr_to_idx = {ipr: i for i, ipr in enumerate(ipr_ids)}
    grouped = df.groupby("ID")["IPR"].apply(list)

    interpro: Dict[str, torch.Tensor] = {}
    for pid, terms in grouped.items():
        vec = torch.zeros(len(ipr_ids), dtype=torch.float32)
        for ipr in terms:
            vec[ipr_to_idx[ipr]] = 1.0
        interpro[pid] = vec

    with open(PROTEIN_GRAPHS_DIR / "interpro_vocab.pkl", "wb") as handle:
        pickle.dump({"ipr_to_idx": ipr_to_idx, "vocab_size": len(ipr_ids)}, handle)

    return interpro, len(ipr_ids)


def load_go_vocab() -> Dict[str, int]:
    """Load GO term vocabulary from OBO file.

    Returns:
        Dictionary mapping ontology names to vocabulary sizes
    """
    namespace_map = {
        "biological_process": "BPO",
        "cellular_component": "CCO",
        "molecular_function": "MFO",
    }
    terms: Dict[str, set] = {"BPO": set(), "CCO": set(), "MFO": set()}
    current_id: Optional[str] = None
    current_ns: Optional[str] = None

    with open(GO_OBO_PATH, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "[Term]":
                current_id = None
                current_ns = None
            elif line.startswith("id: GO:"):
                current_id = line.split("id: ")[1]
            elif line.startswith("namespace: "):
                current_ns = namespace_map.get(line.split("namespace: ")[1])
            elif line == "" and current_id and current_ns:
                terms[current_ns].add(current_id)

    vocab_info: Dict[str, Dict[str, object]] = {}
    for onto, onto_terms in terms.items():
        sorted_terms = sorted(onto_terms)
        go_to_idx = {go: i for i, go in enumerate(sorted_terms)}
        vocab_info[onto] = {
            "terms": sorted_terms,
            "go_to_idx": go_to_idx,
            "vocab_size": len(sorted_terms),
        }

    with open(PROTEIN_GRAPHS_DIR / "go_vocab.pkl", "wb") as handle:
        pickle.dump(vocab_info, handle)

    return {onto: info["vocab_size"] for onto, info in vocab_info.items()}


def load_go_annotations() -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """Load GO term annotations for all ontologies.

    Returns:
        Nested dictionary: protein_id -> ontology -> annotation_type -> term_list
    """
    annotations: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    for onto in ["BPO", "CCO", "MFO"]:
        for kind, template in [
            ("experimental", GO_EXP_ANNOTATION_TEMPLATE),
            ("curated", GO_ANNOTATION_TEMPLATE),
        ]:
            path = template.with_name(template.name.format(onto=onto))
            if not path.exists():
                logger.warning("GO %s annotation file missing: %s", kind, path)
                continue
            df = pd.read_csv(path, sep="\t")
            for _, row in df.iterrows():
                pid = row["EntryID"]
                terms = [
                    term.strip() for term in row["term"].split(";") if term.strip()
                ]
                annotations.setdefault(pid, {}).setdefault(onto, {})[kind] = terms
    return annotations


def build_protein_graph(
    protein_id: str,
    embeddings: torch.Tensor,
    sequence: str,
    interpro_dict: Dict[str, torch.Tensor],
    go_annotations: Dict[str, Dict[str, Dict[str, List[str]]]],
    interpro_vocab_size: int,
) -> Optional[HeteroData]:
    """Build the base heterogeneous graph structure for a protein.

    Creates a graph with:
    - 'aa' nodes: one per amino acid residue
    - 'protein' node: single node representing the whole protein
    - 'belongs_to' edges: connecting each AA to the protein node

    Args:
        protein_id: Unique protein identifier
        embeddings: ESM embeddings tensor, shape (seq_len, embed_dim)
        sequence: Amino acid sequence string
        interpro_dict: Dictionary of InterPro domain annotations
        go_annotations: Dictionary of GO term annotations
        interpro_vocab_size: Size of InterPro vocabulary

    Returns:
        HeteroData object or None if validation fails
    """
    seq_len = len(sequence)

    # Validate that embeddings match sequence length
    if embeddings.shape[0] != seq_len:
        logger.warning(
            "Embedding length mismatch for %s: %d vs %d",
            protein_id,
            embeddings.shape[0],
            seq_len,
        )
        return None

    # Initialize heterogeneous graph
    data = HeteroData()

    # Add amino acid nodes with ESM embeddings as features
    data["aa"].x = embeddings
    data["aa"].num_nodes = seq_len

    # Add single protein node
    data["protein"].num_nodes = 1

    # Add InterPro domain annotations to protein node
    data["protein"].interpro = (
        interpro_dict[protein_id].unsqueeze(0)
        if protein_id in interpro_dict
        else torch.zeros(1, interpro_vocab_size, dtype=torch.float32)
    )

    # Add GO term annotations for each ontology
    for onto in ["BPO", "CCO", "MFO"]:
        data["protein"][f"go_terms_{onto}"] = {
            "experimental": go_annotations.get(protein_id, {})
            .get(onto, {})
            .get("experimental", []),
            "curated": go_annotations.get(protein_id, {})
            .get(onto, {})
            .get("curated", []),
        }

    # Create edges connecting each amino acid to the protein node
    # Edge format: [source_nodes, target_nodes]
    aa_indices = torch.arange(seq_len)
    protein_indices = torch.zeros(seq_len, dtype=torch.long)
    data["aa", "belongs_to", "protein"].edge_index = torch.stack(
        [aa_indices, protein_indices]
    )

    # Store protein metadata
    data["protein"].protein_id = protein_id
    data["protein"].sequence = sequence
    data["protein"].sequence_length = seq_len

    return data


def add_close_contact_edges(data: HeteroData, protein_id: str) -> bool:
    """Add spatial contact edges between amino acids based on 3D structure.

    This function loads a PDB structure file (AlphaFold or ESMFold), extracts
    C-alpha coordinates, and computes spatial contacts between residues within
    the cutoff distance.

    Handles isoforms and truncated proteins:
    - Checks if sequence length matches number of CA atoms in structure
    - Accepts truncated proteins (1024 residues) even with length mismatch
    - Skips proteins with mismatched lengths that aren't truncated

    Args:
        data: HeteroData graph to add contact edges to
        protein_id: Protein identifier for locating structure file

    Returns:
        True if edges were successfully added, False if protein should be skipped
    """
    # Try to locate structure file (prefer AlphaFold over ESMFold)
    af_path = DATA_ROOT / f"alphafold_pdb/AF-{protein_id}-F1-model_v6.pdb"
    esmfold_path = DATA_ROOT / f"esmfold_pdb/ESMFold-{protein_id}.pdb"

    if not af_path.is_file():
        if not esmfold_path.is_file():
            # No structure available - write to missing list and add empty edges
            with open(STRUCTURE_MISSING_PATH, "a", encoding="utf-8") as fasta_out:
                fasta_out.write(f">{protein_id}\n{data['protein'].sequence}\n")
            logger.warning(
                "No structure found for %s; storing sequence in missing list.",
                protein_id,
            )
            data["aa", "close_to", "aa"].edge_index = torch.empty(
                (2, 0), dtype=torch.long
            )
            data["aa", "close_to", "aa"].edge_attr = torch.empty(
                (0,), dtype=torch.float32
            )
            return True
        af_path = esmfold_path

    # Check for mismatch between expected and actual CA counts
    sequence = data["protein"].sequence
    num_nodes = data["aa"].num_nodes

    expected_ca_count = sum(1 for aa in sequence if aa != "X")
    actual_ca_count = count_ca_atoms(af_path)

    if expected_ca_count != actual_ca_count:
        # Check if this is a truncated protein (1024 residues)
        if actual_ca_count == 1024:
            logger.info(
                "%s: Truncated protein detected (1024 CA atoms vs %d expected non-X residues). Proceeding with truncated structure.",
                protein_id,
                expected_ca_count,
            )
        else:
            # Length mismatch and not a truncated protein - skip this protein
            logger.warning(
                "%s: Sequence/structure mismatch (%d non-X residues vs %d CA atoms) and not truncated. Skipping graph creation.",
                protein_id,
                expected_ca_count,
                actual_ca_count,
            )
            return False

    # Load coordinates aligned to sequence (with NaN for missing residues)
    coords = load_ca_coordinates(af_path, num_nodes)

    # Count how many residues have valid coordinates
    valid_residues = (~torch.isnan(coords[:, 0])).sum().item()
    missing_residues = num_nodes - valid_residues

    # Log information about missing residues
    if missing_residues > 0:
        logger.info(
            "%s: %d/%d residues have coordinates (%d missing from structure)",
            protein_id,
            valid_residues,
            num_nodes,
            missing_residues,
        )

    # Build contact edges based on 3D distances
    edge_index, edge_attr = build_close_contact_edges(
        coords, CONTACT_CUTOFF, CONTACT_CHUNK_SIZE
    )

    data["aa", "close_to", "aa"].edge_index = edge_index
    data["aa", "close_to", "aa"].edge_attr = edge_attr

    undirected = edge_index.size(1) // 2
    total_pairs = valid_residues * (valid_residues - 1) // 2
    ratio = undirected / total_pairs if total_pairs > 0 else 0.0

    logger.info(
        "%s: %d close-contact links (%.4f density among %d valid residues)",
        protein_id,
        undirected,
        ratio,
        valid_residues,
    )

    return True


class ProteinGraphBuilder:
    """Builder class for creating protein graphs from multiple data sources."""

    def __init__(
        self,
        num_workers: Optional[int] = None,
    ):
        """Initialize the graph builder with required data sources.

        Args:
            num_workers: Number of parallel workers (currently ignored)
        """
        if not EMBED_H5_PATH.exists():
            raise FileNotFoundError(f"Embedding H5 not found: {EMBED_H5_PATH}")

        if num_workers and num_workers > 1:
            logger.warning("num_workers argument ignored; processing sequentially.")

        logger.info("Graph construction will run sequentially.")

        # Load InterPro domain annotations
        self.interpro_dict, self.interpro_vocab_size = load_interpro_annotations()
        logger.info("InterPro entries: %d", len(self.interpro_dict))

        # Load GO term annotations
        self.go_annotations = load_go_annotations()
        logger.info("GO annotations for proteins: %d", len(self.go_annotations))
        self.go_vocab_sizes = load_go_vocab()
        logger.info("GO vocab sizes: %s", self.go_vocab_sizes)

    def process_fasta(
        self,
        fasta_path: Path,
    ) -> Tuple[int, int, int]:
        """Process a FASTA file and build graphs for all proteins.

        Args:
            fasta_path: Path to input FASTA file

        Returns:
            Tuple of (success_count, failed_count, missing_embeddings_count)
        """
        fasta_path = Path(fasta_path)
        if not fasta_path.exists():
            raise FileNotFoundError(f"FASTA not found: {fasta_path}")

        records = list(SeqIO.parse(str(fasta_path), "fasta"))
        if not records:
            logger.warning("No sequences found in FASTA %s", fasta_path)
            return 0, 0, 0

        success = 0
        failed = 0
        missing_embeddings = 0

        with h5py.File(EMBED_H5_PATH, "r") as h5f, tqdm(
            total=len(records), desc="Building protein graphs", unit="protein"
        ) as progress:
            for record in records[::-1]:
                protein_id = record.id
                sequence = str(record.seq)

                ok, error, missing_flag = build_and_save_protein_graph(
                    protein_id,
                    sequence,
                    h5f,
                    self.interpro_dict,
                    self.go_annotations,
                    self.interpro_vocab_size,
                )

                if missing_flag:
                    missing_embeddings += 1
                    if error:
                        logger.warning("Skipping %s: %s", protein_id, error)
                elif ok:
                    success += 1
                else:
                    failed += 1
                    if error:
                        logger.error("Graph build failed for %s: %s", protein_id, error)
                progress.update(1)

        logger.info(
            "Graph building complete: success=%d failed=%d missing_embeddings=%d",
            success,
            failed,
            missing_embeddings,
        )
        return success, failed, missing_embeddings


def main() -> None:
    """Main entry point for graph construction."""
    parser = argparse.ArgumentParser(
        description="Create protein graphs using precomputed embeddings and structures."
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        default=FASTA_PATH,
        help="Input FASTA file (default: SwissProt 2024_01).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers for graph construction.",
    )
    args = parser.parse_args()

    builder = ProteinGraphBuilder(num_workers=args.num_workers)
    success, failed, missing = builder.process_fasta(args.fasta)
    logger.info(
        "Graph generation summary - success=%d failed=%d missing_embeddings=%d",
        success,
        failed,
        missing,
    )


if __name__ == "__main__":
    main()
