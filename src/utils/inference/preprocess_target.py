"""Target protein preprocessing for inference.

Handles the per-protein preprocessing steps needed before inference:
- ESM-1b embedding generation
- AlphaFold/PDB structure download
- Contact graph construction
- Assembly into a HeteroData .pt graph file

This module reuses existing utilities from the training pipeline
(embed_residues, create_graphs, api) to ensure consistency.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from torch_geometric.data import HeteroData

from src.data.create_graphs import (
    build_close_contact_edges,
    count_ca_atoms,
    load_ca_coordinates,
)
from src.data.embed_residues import ResidueEmbedder
from src.utils.api import download_alphafold, download_pdb
from src.utils.constants import (
    CONTACT_CHUNK_SIZE,
    CONTACT_CUTOFF,
    ESM1B_CHECKPOINT,
    ESM_DEFAULT_BATCH_SIZE,
    ESM_MAX_AA_PER_BATCH,
    INFERENCE_OUTPUT_DIR,
    PROTEIN_GRAPHS_DIR,
    TRUNCATED_PROTEIN_LENGTH,
)
from src.utils.inference.resolve_ids import ProteinInfo

logger = logging.getLogger(__name__)

# Module-level embedder (loaded lazily to avoid loading ESM at import time)
_embedder: Optional[ResidueEmbedder] = None


def get_embedder(
    checkpoint: Optional[Path] = None,
    device: str = "cpu",
) -> ResidueEmbedder:
    """Get or create a ResidueEmbedder singleton.

    Args:
        checkpoint: Path to local ESM-1b checkpoint. If None, uses default.
        device: Device for ESM inference.

    Returns:
        ResidueEmbedder instance.
    """
    global _embedder
    if _embedder is None:
        ckpt = checkpoint or (ESM1B_CHECKPOINT if ESM1B_CHECKPOINT.exists() else None)
        logger.info("Loading ESM-1b model (checkpoint=%s)...", ckpt)
        _embedder = ResidueEmbedder(
            local_checkpoint=ckpt,
            max_aa_per_batch=ESM_MAX_AA_PER_BATCH,
            esm_batch_size=ESM_DEFAULT_BATCH_SIZE,
        )
    return _embedder


def embed_protein(
    protein_info: ProteinInfo,
    checkpoint: Optional[Path] = None,
) -> torch.Tensor:
    """Generate ESM-1b per-residue embeddings for a protein.

    Args:
        protein_info: Resolved protein information.
        checkpoint: Optional path to local ESM checkpoint.

    Returns:
        Tensor of shape (seq_len, 1280) with per-residue embeddings.
    """
    embedder = get_embedder(checkpoint=checkpoint)
    logger.info(
        "Generating ESM-1b embeddings for %s (%d aa)...",
        protein_info.accession,
        len(protein_info.sequence),
    )
    embedding = embedder.embed_sequence(
        protein_info.accession, protein_info.sequence
    ).to(torch.float32)
    logger.info(
        "Embedding shape for %s: %s", protein_info.accession, tuple(embedding.shape)
    )
    return embedding


def download_structure(
    protein_info: ProteinInfo,
    output_dir: Path,
    force: bool = False,
) -> Optional[Path]:
    """Download or locate a 3D structure for the protein.

    Tries in order:
    1. AlphaFold DB download
    2. PDB (RCSB) download

    Args:
        protein_info: Resolved protein information.
        output_dir: Directory to save the PDB file.
        force: If True, re-download even if PDB already exists locally.

    Returns:
        Path to the PDB file, or None if no structure found.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = output_dir / f"{protein_info.accession}.pdb"

    if pdb_path.exists() and not force:
        logger.info("Structure already exists at %s", pdb_path)
        return pdb_path

    # Try AlphaFold
    if download_alphafold(protein_info.accession, str(pdb_path)):
        logger.info(
            "Downloaded AlphaFold structure for %s → %s",
            protein_info.accession,
            pdb_path,
        )
        return pdb_path

    # Try RCSB PDB
    if download_pdb(protein_info.accession, str(pdb_path)):
        logger.info(
            "Downloaded PDB structure for %s → %s",
            protein_info.accession,
            pdb_path,
        )
        return pdb_path

    logger.warning(
        "No structure found for %s (tried AlphaFold + PDB)", protein_info.accession
    )
    return None


def build_target_graph(
    protein_info: ProteinInfo,
    embedding: torch.Tensor,
    pdb_path: Optional[Path],
    ipr_vocab_size: int,
    contact_cutoff: float = CONTACT_CUTOFF,
    contact_chunk: int = CONTACT_CHUNK_SIZE,
) -> HeteroData:
    """Build a HeteroData graph for a single target protein.

    Creates the same graph structure as create_graphs.build_protein_graph()
    but tailored for inference (no GO annotations stored, no InterPro
    annotations since we use DEL_NOIPR models).

    Args:
        protein_info: Resolved protein information.
        embedding: ESM-1b embeddings tensor, shape (seq_len, 1280).
        pdb_path: Path to PDB structure file. If None, empty contact edges.
        ipr_vocab_size: InterPro vocabulary size (for zero-vector placeholder).
        contact_cutoff: Distance cutoff for contact edges in Angstroms.
        contact_chunk: Chunk size for contact edge computation.

    Returns:
        HeteroData object ready to be saved as .pt.
    """
    seq_len = len(protein_info.sequence)
    sequence = protein_info.sequence

    if embedding.shape[0] != seq_len:
        raise ValueError(
            f"Embedding length {embedding.shape[0]} does not match "
            f"sequence length {seq_len} for {protein_info.accession}"
        )

    data = HeteroData()

    # ----- AA nodes -----
    data["aa"].x = embedding
    data["aa"].num_nodes = seq_len

    # ----- Protein node -----
    data["protein"].num_nodes = 1
    data["protein"].protein_id = protein_info.accession
    data["protein"].sequence = sequence
    data["protein"].sequence_length = seq_len

    # InterPro: zeros (DEL_NOIPR models don't use InterPro)
    data["protein"].interpro = torch.zeros(1, ipr_vocab_size, dtype=torch.float32)

    # GO annotations: empty dicts (query protein — will be zeroed at inference)
    for onto in ["BPO", "CCO", "MFO"]:
        data["protein"][f"go_terms_{onto}"] = {"experimental": [], "curated": []}

    # ----- AA → Protein edges -----
    aa_indices = torch.arange(seq_len)
    data["aa", "belongs_to", "protein"].edge_index = torch.stack(
        [aa_indices, torch.zeros(seq_len, dtype=torch.long)]
    )

    # ----- Contact edges -----
    if pdb_path is not None and pdb_path.exists():
        try:
            actual_ca = count_ca_atoms(pdb_path)
            expected_ca = sum(1 for aa in sequence if aa != "X")

            # Handle truncated proteins
            effective_len = seq_len
            effective_embedding = embedding
            if expected_ca != actual_ca:
                if actual_ca == TRUNCATED_PROTEIN_LENGTH:
                    logger.info(
                        "%s: Truncated protein (%d CA vs %d expected). Using truncated.",
                        protein_info.accession,
                        actual_ca,
                        expected_ca,
                    )
                    effective_len = TRUNCATED_PROTEIN_LENGTH
                    effective_embedding = embedding[:effective_len]
                else:
                    logger.warning(
                        "%s: CA count mismatch (%d vs %d). Adding empty contact edges.",
                        protein_info.accession,
                        actual_ca,
                        expected_ca,
                    )
                    data["aa", "close_to", "aa"].edge_index = torch.empty(
                        (2, 0), dtype=torch.long
                    )
                    data["aa", "close_to", "aa"].edge_attr = torch.empty(
                        (0,), dtype=torch.float32
                    )
                    return data

            coords = load_ca_coordinates(pdb_path, effective_len)
            edge_index, edge_attr = build_close_contact_edges(
                coords, cutoff=contact_cutoff, chunk_size=contact_chunk
            )

            data["aa", "close_to", "aa"].edge_index = edge_index
            data["aa", "close_to", "aa"].edge_attr = edge_attr

            valid_residues = (~torch.isnan(coords[:, 0])).sum().item()
            undirected = edge_index.size(1) // 2
            logger.info(
                "%s: %d contact edges (%d valid residues)",
                protein_info.accession,
                undirected,
                valid_residues,
            )
        except Exception as exc:
            logger.warning(
                "%s: Failed to build contact edges: %s. Using empty.",
                protein_info.accession,
                exc,
            )
            data["aa", "close_to", "aa"].edge_index = torch.empty(
                (2, 0), dtype=torch.long
            )
            data["aa", "close_to", "aa"].edge_attr = torch.empty(
                (0,), dtype=torch.float32
            )
    else:
        # No structure available
        data["aa", "close_to", "aa"].edge_index = torch.empty(
            (2, 0), dtype=torch.long
        )
        data["aa", "close_to", "aa"].edge_attr = torch.empty(
            (0,), dtype=torch.float32
        )

    return data


def preprocess_target(
    protein_info: ProteinInfo,
    output_dir: Optional[Path] = None,
    ipr_vocab_size: int = 0,
    esm_checkpoint: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Run full preprocessing for a target protein.

    Steps:
    1. Check if .pt already exists (skip if not forced).
    2. Generate ESM-1b embeddings.
    3. Download AlphaFold/PDB structure.
    4. Build HeteroData graph.
    5. Save .pt file.

    Args:
        protein_info: Resolved protein information.
        output_dir: Directory for output files. Defaults to INFERENCE_OUTPUT_DIR.
        ipr_vocab_size: InterPro vocabulary size.
        esm_checkpoint: Optional local ESM-1b checkpoint path.
        force: If True, regenerate even if .pt exists.

    Returns:
        Path to the saved .pt graph file.
    """
    if output_dir is None:
        output_dir = INFERENCE_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_path = output_dir / f"{protein_info.accession}.pt"
    if pt_path.exists() and not force:
        logger.info("Graph already exists at %s, skipping preprocessing.", pt_path)
        return pt_path

    # If the protein already exists in the precomputed SwissProt graphs, reuse it directly
    std_pt_path = PROTEIN_GRAPHS_DIR / f"{protein_info.accession}.pt"
    if std_pt_path.exists() and not force:
        logger.info(
            "Graph found in SwissProt graphs repository (%s), copying to %s",
            std_pt_path,
            pt_path,
        )
        import shutil
        shutil.copy2(std_pt_path, pt_path)
        return pt_path

    # Step 1: ESM-1b embeddings
    embedding = embed_protein(protein_info, checkpoint=esm_checkpoint)

    # Step 2: Structure
    pdb_path = download_structure(protein_info, output_dir, force=force)

    # Step 3: Build graph
    graph = build_target_graph(
        protein_info=protein_info,
        embedding=embedding,
        pdb_path=pdb_path,
        ipr_vocab_size=ipr_vocab_size,
    )

    # Step 4: Save
    torch.save(graph, pt_path)
    logger.info("Saved target graph to %s", pt_path)

    return pt_path


def unload_embedder() -> None:
    """Unload the ESM-1b embedder from memory to free GPU VRAM."""
    global _embedder
    if _embedder is not None:
        del _embedder
        _embedder = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Unloaded ESM-1b embedder and released GPU memory.")
