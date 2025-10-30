"""End-to-end preprocessing pipeline for single protein inference.

This module provides a complete preprocessing workflow for preparing individual
proteins for inference with a trained model. It handles all required steps:
- ESM embedding generation
- Structure prediction via ESMFold (if needed)
- InterPro domain annotation via InterProScan API
- DIAMOND alignment to SwissProt database
- Graph construction with all node and edge features

This is particularly useful for inference on new, unannotated proteins where
the full preprocessing pipeline must be run on-the-fly.
"""

import argparse
import json
import logging
import os
import pickle
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from Bio import SeqIO
from torch_geometric.data import HeteroData

from src.data.embed_residues import ResidueEmbedder
from src.data.create_graphs import (
    build_close_contact_edges,
    load_ca_coordinates,
    count_ca_atoms,
)
from src.data.inference_pipeline.query_interproscan import submit_interproscan
from src.utils.api import download_pdb, download_alphafold
from src.utils.constants import (
    # Paths
    DIAMOND_SWISSPROT_DB,
    INTERPRO_VOCAB,
    TMP_DIR,
    # ESM parameters
    ESM_MAX_AA_PER_BATCH,
    ESM_DEFAULT_BATCH_SIZE,
    ESM1B_CHECKPOINT,
    # ESMFold parameters
    ESMFOLD_DEFAULT_CHUNK_SIZE,
    ESMFOLD_CONDA_ENV,
    ESMFOLD_MODEL_NAME,
    TRUNCATED_PROTEIN_LENGTH,
    # InterProScan parameters
    INTERPROSCAN_DEFAULT_EMAIL,
    INTERPROSCAN_DEFAULT_TIMEOUT,
    INTERPROSCAN_DEFAULT_POLL_INTERVAL,
    # DIAMOND parameters
    DIAMOND_EXECUTABLE,
    DIAMOND_DEFAULT_EVALUE,
    DIAMOND_DEFAULT_TOPK,
    # Graph construction parameters
    CONTACT_CUTOFF,
    CONTACT_CHUNK_SIZE,
)

logger = logging.getLogger("inference_preprocess")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end preprocessing for protein sequences."
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        required=True,
        help="Input FASTA file (can contain multiple sequences)",
    )

    # Embedding options
    parser.add_argument(
        "--max-aa-per-batch",
        type=int,
        default=ESM_MAX_AA_PER_BATCH,
        help=f"Maximum amino acids per batch for ESM (default: {ESM_MAX_AA_PER_BATCH})",
    )
    parser.add_argument(
        "--esm-batch-size",
        type=int,
        default=ESM_DEFAULT_BATCH_SIZE,
        help=f"Batch size for ESM embedding (default: {ESM_DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--esm-local-checkpoint",
        type=Path,
        default=ESM1B_CHECKPOINT,
        help=f"Path to local ESM checkpoint (default: {ESM1B_CHECKPOINT})",
    )

    # Fold options
    parser.add_argument(
        "--pdb",
        type=Path,
        default=None,
        help="Path to input PDB file (if structure is already available)",
    )
    parser.add_argument(
        "--fold-local-only",
        action="store_true",
        help="Use only local ESMFold model files",
    )
    parser.add_argument(
        "--fold-chunk-size",
        type=int,
        default=ESMFOLD_DEFAULT_CHUNK_SIZE,
        help=f"Chunk size for ESMFold trunk (default: {ESMFOLD_DEFAULT_CHUNK_SIZE})",
    )

    # InterProScan options
    parser.add_argument(
        "--interpro-email",
        type=str,
        default=INTERPROSCAN_DEFAULT_EMAIL,
        help=f"Email for InterProScan API (default: {INTERPROSCAN_DEFAULT_EMAIL})",
    )
    parser.add_argument(
        "--interpro-timeout",
        type=int,
        default=INTERPROSCAN_DEFAULT_TIMEOUT,
        help=f"Timeout for InterProScan queries in seconds (default: {INTERPROSCAN_DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--interpro-poll-interval",
        type=int,
        default=INTERPROSCAN_DEFAULT_POLL_INTERVAL,
        help=f"Polling interval for InterProScan in seconds (default: {INTERPROSCAN_DEFAULT_POLL_INTERVAL})",
    )
    parser.add_argument(
        "--interpro-vocab",
        type=Path,
        default=INTERPRO_VOCAB,
        help=f"Path to InterPro vocabulary pickle file (default: {INTERPRO_VOCAB})",
    )

    # DIAMOND options
    parser.add_argument(
        "--diamond-bin",
        type=str,
        default=DIAMOND_EXECUTABLE,
        help=f"Path to DIAMOND executable (default: {DIAMOND_EXECUTABLE})",
    )
    parser.add_argument(
        "--diamond-db",
        type=Path,
        default=DIAMOND_SWISSPROT_DB,
        help=f"Path to DIAMOND database (.dmnd) (default: {DIAMOND_SWISSPROT_DB})",
    )
    parser.add_argument(
        "--diamond-topk",
        type=int,
        default=DIAMOND_DEFAULT_TOPK,
        help=f"Number of top hits to retrieve (default: {DIAMOND_DEFAULT_TOPK})",
    )
    parser.add_argument(
        "--diamond-evalue",
        type=float,
        default=DIAMOND_DEFAULT_EVALUE,
        help=f"E-value threshold for DIAMOND (default: {DIAMOND_DEFAULT_EVALUE})",
    )

    # Graph options
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TMP_DIR,
        help=f"Output directory for all results (default: {TMP_DIR})",
    )
    parser.add_argument(
        "--contact-cutoff",
        type=float,
        default=CONTACT_CUTOFF,
        help=f"Distance cutoff for contact edges in Angstroms (default: {CONTACT_CUTOFF})",
    )
    parser.add_argument(
        "--contact-chunk",
        type=int,
        default=CONTACT_CHUNK_SIZE,
        help=f"Chunk size for contact edge computation (default: {CONTACT_CHUNK_SIZE})",
    )

    # Child process flags (internal use)
    parser.add_argument("--protein-id", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=TMP_DIR,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--pdb-output",
        type=Path,
        default=TMP_DIR,
        help=argparse.SUPPRESS,
    )

    return parser.parse_args()


def read_proteins_from_fasta(fasta: Path) -> List[Tuple[str, str]]:
    """Read all protein sequences from a FASTA file."""
    records = list(SeqIO.parse(str(fasta), "fasta"))
    if not records:
        raise ValueError(f"No records found in FASTA {fasta}.")

    proteins = []
    for record in records:
        protein_id = record.id
        sequence = str(record.seq).strip().upper()
        if not sequence:
            logger.warning(f"Empty sequence for {protein_id}, skipping.")
            continue
        proteins.append((protein_id, sequence))

    logger.info(f"Found {len(proteins)} protein(s) in {fasta}")
    return proteins


def ensure_output_dir(path: Path) -> Path:
    """Create output directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def embed_sequence(
    protein_id: str,
    sequence: str,
    checkpoint: Optional[Path],
    max_aa: int,
    esm_batch_size: int,
) -> torch.Tensor:
    """Generate ESM embeddings for a protein sequence."""
    logger.info("Embedding sequence %s (%d aa).", protein_id, len(sequence))
    embedder = ResidueEmbedder(
        local_checkpoint=checkpoint,
        max_aa_per_batch=max_aa,
        esm_batch_size=esm_batch_size,
    )
    embedding = embedder.embed_sequence(protein_id, sequence).to(torch.float32)
    logger.info("Embedding shape: %s.", tuple(embedding.shape))
    return embedding


def run_esmfold(
    protein_id: str,
    fasta_path: Path,
    output_path: Path,
    local_only: bool,
    chunk_size: int,
    child: bool = False,
) -> None:
    """Run ESMFold to predict protein structure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # if child:
    #     # This runs in the esmfold conda environment
    #     from transformers import AutoTokenizer, EsmForProteinFolding
    #     from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
    #     from transformers.models.esm.openfold_utils.protein import (
    #         Protein as OFProtein,
    #         to_pdb,
    #     )

    #     # Read sequence from FASTA
    #     records = list(SeqIO.parse(str(fasta_path), "fasta"))
    #     if not records:
    #         raise ValueError(f"No records in {fasta_path}")
    #     sequence = str(records[0].seq).strip().upper()

    #     logger.info(f"Loading ESMFold model (local_only={local_only})...")
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         ESMFOLD_MODEL_NAME, local_files_only=local_only
    #     )
    #     model = EsmForProteinFolding.from_pretrained(
    #         ESMFOLD_MODEL_NAME,
    #         low_cpu_mem_usage=True,
    #         local_files_only=local_only,
    #     )
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     model = model.to(device)
    #     model.eval()
    #     model.esm = model.esm.half()
    #     model.trunk.set_chunk_size(chunk_size)

    #     logger.info(f"Folding {protein_id} ({len(sequence)} aa)...")
    #     toks = tokenizer(
    #         [sequence], return_tensors="pt", add_special_tokens=False, padding=True
    #     )
    #     model_inputs = {k: v.to(device) for k, v in toks.items()}

    #     with torch.no_grad():
    #         outputs = model(**model_inputs)

    #     final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    #     outputs_np = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    #     final_atom_positions = final_atom_positions.cpu().numpy()
    #     final_atom_mask = outputs_np["atom37_atom_exists"]

    #     length = len(sequence)
    #     protein = OFProtein(
    #         aatype=outputs_np["aatype"][0][:length],
    #         atom_positions=final_atom_positions[0][:length],
    #         atom_mask=final_atom_mask[0][:length],
    #         residue_index=outputs_np["residue_index"][0][:length] + 1,
    #         b_factors=outputs_np["plddt"][0][:length],
    #         chain_index=(
    #             outputs_np["chain_index"][0][:length]
    #             if "chain_index" in outputs_np
    #             else None
    #         ),
    #     )
    #     pdb_str = to_pdb(protein)
    #     output_path.write_text(pdb_str, encoding="utf-8")
    #     logger.info("Saved PDB to %s.", output_path)
    #     return

    # Parent process: launch child in esmfold environment
    script_path = Path(__file__).resolve()
    cmd = [
        "conda",
        "run",
        "-n",
        ESMFOLD_CONDA_ENV,
        "python",
        str(script_path),
        "--fasta",
        str(fasta_path),
    ]

    env = os.environ.copy()
    cwd_entry = str(Path.cwd())
    env["PYTHONPATH"] = (
        f"{env['PYTHONPATH']}{os.pathsep}{cwd_entry}"
        if env.get("PYTHONPATH")
        else cwd_entry
    )

    logger.debug("Running command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    if not output_path.exists():
        raise RuntimeError("ESMFold did not produce a PDB file.")
    logger.info("PDB stored at %s.", output_path)


def load_interpro_vocab(path: Path) -> Tuple[Dict[str, int], int]:
    """Load InterPro vocabulary mapping."""
    if not path.exists():
        logger.warning("InterPro vocab %s not found; using empty mapping.", path)
        return {}, 0

    with path.open("rb") as handle:
        payload = pickle.load(handle)

    if "ipr_to_idx" in payload:
        return payload["ipr_to_idx"], payload["vocab_size"]
    if isinstance(payload, dict) and "terms" in payload:
        mapping = {term: idx for idx, term in enumerate(payload["terms"])}
        return mapping, len(mapping)

    raise ValueError(f"Unexpected InterPro vocab format in {path}.")


def interpro_vector(
    hits: List[str], mapping: Dict[str, int], size: int
) -> Tuple[torch.Tensor, List[str]]:
    """Convert InterPro hits to a binary vector."""
    vec = torch.zeros(size, dtype=torch.float32) if size else torch.zeros(0)
    missing: List[str] = []

    for accession in hits:
        idx = mapping.get(accession)
        if idx is None:
            missing.append(accession)
            continue
        vec[idx] = 1.0

    return vec.unsqueeze(0), missing


def run_diamond_alignment(
    protein_id: str,
    sequence: str,
    diamond_bin: str,
    database: Path,
    output_path: Path,
    topk: int,
    evalue: float,
    workspace: Path,
) -> Path:
    """Run DIAMOND alignment against SwissProt database."""
    logger.info("Running DIAMOND alignment for %s.", protein_id)

    query_fasta = workspace / f"{protein_id}_diamond_query.fasta"
    query_fasta.write_text(f">{protein_id}\n{sequence}\n", encoding="utf-8")

    cmd = [
        diamond_bin,
        "blastp",
        "--db",
        str(database),
        "--query",
        str(query_fasta),
        "--out",
        str(output_path),
        "--outfmt",
        "6",
        "--evalue",
        str(evalue),
    ]

    logger.debug("Running command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    logger.info("DIAMOND results stored at %s.", output_path)

    return query_fasta


def build_graph(
    protein_id: str,
    sequence: str,
    embedding: torch.Tensor,
    pdb_path: Path,
    interpro_vec: torch.Tensor,
    interpro_hits: List[str],
    contact_cutoff: float,
    contact_chunk: int,
) -> HeteroData:
    """Build heterogeneous graph from protein data.

    Handles isoforms and truncated proteins:
    - Checks if sequence length matches number of CA atoms in structure
    - Accepts truncated proteins (1024 residues) even with length mismatch
    - Raises error for mismatched lengths that aren't truncated
    """
    seq_len = len(sequence)

    if embedding.shape[0] != seq_len:
        raise ValueError(
            f"Embedding length {embedding.shape[0]} does not match sequence length {seq_len}."
        )

    # Check for mismatch between expected and actual CA counts
    expected_ca_count = sum(1 for aa in sequence if aa != "X")
    actual_ca_count = count_ca_atoms(pdb_path)

    if expected_ca_count != actual_ca_count:
        # Check if this is a truncated protein
        if actual_ca_count == TRUNCATED_PROTEIN_LENGTH:
            logger.info(
                "%s: Truncated protein detected (%d CA atoms vs %d expected non-X residues). Proceeding with truncated structure.",
                protein_id,
                TRUNCATED_PROTEIN_LENGTH,
                expected_ca_count,
            )
            # Adjust sequence length to match truncated structure
            seq_len = TRUNCATED_PROTEIN_LENGTH
            if embedding.shape[0] > seq_len:
                embedding = embedding[:seq_len]
                sequence = sequence[:seq_len]
        else:
            # Length mismatch and not a truncated protein - cannot process
            raise ValueError(
                f"{protein_id}: Sequence/structure mismatch ({expected_ca_count} non-X residues "
                f"vs {actual_ca_count} CA atoms) and not truncated. Cannot build graph."
            )

    # Load CA coordinates from PDB
    coords = load_ca_coordinates(pdb_path, seq_len)

    # Count valid coordinates
    valid_residues = (~torch.isnan(coords[:, 0])).sum().item()
    missing_residues = seq_len - valid_residues

    if missing_residues > 0:
        logger.info(
            "%s: %d/%d residues have coordinates (%d missing from structure)",
            protein_id,
            valid_residues,
            seq_len,
            missing_residues,
        )

    # Build contact edges
    edge_index, edge_attr = build_close_contact_edges(
        coords, cutoff=contact_cutoff, chunk_size=contact_chunk
    )

    # Log edge statistics
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

    # Create heterogeneous graph
    data = HeteroData()

    # Amino acid nodes
    data["aa"].x = embedding
    data["aa"].num_nodes = seq_len

    # Protein node
    data["protein"].num_nodes = 1
    data["protein"].protein_id = protein_id
    data["protein"].sequence = sequence[:seq_len]
    data["protein"].sequence_length = seq_len
    data["protein"].interpro = interpro_vec
    data["protein"].ipr_accessions = interpro_hits

    # Amino acid to protein edges
    aa_indices = torch.arange(seq_len)
    data["aa", "belongs_to", "protein"].edge_index = torch.stack(
        [aa_indices, torch.zeros(seq_len, dtype=torch.long)]
    )
    data["protein", "has", "aa"].edge_index = torch.stack(
        [torch.zeros(seq_len, dtype=torch.long), aa_indices]
    )

    # Contact edges between amino acids
    data["aa", "close_to", "aa"].edge_index = edge_index
    data["aa", "close_to", "aa"].edge_attr = edge_attr

    return data


def save_metadata(output_path: Path, info: Dict) -> None:
    """Save metadata to JSON file."""
    output_path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")


def process_protein(
    protein_id: str,
    sequence: str,
    args: argparse.Namespace,
    output_dir: Path,
    ipr_mapping: Dict[str, int],
    ipr_size: int,
) -> Dict:
    """Process a single protein through the full pipeline."""
    logger.info(f"Processing protein {protein_id} ({len(sequence)} aa)")

    protein_dir = output_dir / protein_id
    ensure_output_dir(protein_dir)

    # Write individual FASTA
    protein_fasta_path = protein_dir / f"{protein_id}.fasta"
    protein_fasta_path.write_text(f">{protein_id}\n{sequence}\n", encoding="utf-8")

    # Step 1: Generate ESM embeddings
    logger.info(f"[{protein_id}] Step 1/5: Generating embeddings...")
    embedding = embed_sequence(
        protein_id,
        sequence,
        args.esm_local_checkpoint,
        args.max_aa_per_batch,
        args.esm_batch_size,
    )

    # Step 2: Predict structure with ESMFold
    logger.info(f"[{protein_id}] Step 2/5: Predicting structure with ESMFold...")
    pdb_path = output_dir / f"{protein_id}.pdb"
    
    if not args.pdb:
        # Try to get from PDB
        if download_pdb(protein_id, pdb_path):
            logger.info(f"[{protein_id}] Downloaded PDB structure to {pdb_path}.")
        # Try to get from Alphafold
        elif download_alphafold(protein_id, pdb_path):
            logger.info(f"[{protein_id}] Downloaded AlphaFold structure to {pdb_path}.")
        else:
            logger.info(
                f"[{protein_id}] Failed to download AlphaFold structure, falling back to ESMFold."
            )
            run_esmfold(
                protein_id=protein_id,
                fasta_path=protein_fasta_path,
                output_path=pdb_path,
                local_only=args.fold_local_only,
                chunk_size=args.fold_chunk_size,
            )

    # Step 3: Query InterProScan
    logger.info(f"[{protein_id}] Step 3/5: Querying InterProScan...")
    ipr_hits = submit_interproscan(
        sequence,
        email=args.interpro_email,
        timeout=args.interpro_timeout,
        poll_interval=args.interpro_poll_interval,
    )
    logger.info(f"[{protein_id}] InterProScan returned {len(ipr_hits)} accessions.")

    interpro_vec, missing_ipr = interpro_vector(ipr_hits, ipr_mapping, ipr_size)
    if missing_ipr:
        logger.warning(
            f"[{protein_id}] InterPro accessions missing from vocab: {', '.join(missing_ipr[:10])}"
        )

    # Step 4: Run DIAMOND alignment
    logger.info(f"[{protein_id}] Step 4/5: Running DIAMOND alignment...")
    diamond_out = protein_dir / f"{protein_id}_diamond.tsv"
    diamond_query_path = run_diamond_alignment(
        protein_id,
        sequence,
        args.diamond_bin,
        args.diamond_db,
        diamond_out,
        args.diamond_topk,
        args.diamond_evalue,
        protein_dir,
    )

    # Step 5: Build graph
    logger.info(f"[{protein_id}] Step 5/5: Building graph...")
    graph_path = protein_dir / f"{protein_id}_graph.pt"
    graph = build_graph(
        protein_id,
        sequence,
        embedding,
        pdb_path,
        interpro_vec,
        ipr_hits,
        args.contact_cutoff,
        args.contact_chunk,
    )
    torch.save(graph, graph_path)
    logger.info(f"[{protein_id}] Graph saved to {graph_path}")

    # Save metadata
    metadata = {
        "protein_id": protein_id,
        "sequence_length": len(sequence),
        "pdb_path": str(pdb_path),
        "diamond_output": str(diamond_out),
        "diamond_query_fasta": str(diamond_query_path),
        "graph_path": str(graph_path),
        "interpro_hits": ipr_hits,
        "interpro_missing": missing_ipr,
    }
    save_metadata(protein_dir / "metadata.json", metadata)

    logger.info(f"[{protein_id}] Pipeline completed successfully!")
    return metadata


def main() -> None:
    args = parse_args()

    # Read proteins from FASTA
    proteins = read_proteins_from_fasta(args.fasta)

    # Setup output directory
    output_dir = ensure_output_dir(args.output_dir)
    logger.info(f"Using output directory {output_dir}")

    # Load InterPro vocabulary
    logger.info("Loading InterPro vocabulary...")
    ipr_mapping, ipr_size = load_interpro_vocab(args.interpro_vocab)
    logger.info(f"InterPro vocabulary size: {ipr_size}")

    # Process each protein
    all_metadata = {}
    for protein_id, sequence in proteins:
        try:
            metadata = process_protein(
                protein_id,
                sequence,
                args,
                output_dir,
                ipr_mapping,
                ipr_size,
            )
            all_metadata[protein_id] = metadata
        except Exception as e:
            logger.error(f"Failed to process {protein_id}: {e}", exc_info=True)
            all_metadata[protein_id] = {"error": str(e)}

    # Save overall summary
    summary_path = output_dir / "summary.json"
    save_metadata(
        summary_path,
        {
            "total_proteins": len(proteins),
            "successful": len([m for m in all_metadata.values() if "error" not in m]),
            "failed": len([m for m in all_metadata.values() if "error" in m]),
            "proteins": all_metadata,
        },
    )

    logger.info(f"All done! Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
