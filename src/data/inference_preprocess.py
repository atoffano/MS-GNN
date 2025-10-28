"""End-to-end preprocessing pipeline for single protein inference.

This module provides a complete preprocessing workflow for preparing individual
proteins for inference with a trained model. It handles all required steps:
- ESM embedding generation
- Structure prediction via ESMFold (if needed)
- InterPro domain annotation via InterProScan API
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
from src.data.create_graphs import build_close_contact_edges, load_ca_coordinates
from src.data.query_interproscan import submit_interproscan

logger = logging.getLogger("inference_preprocess")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end preprocessing for a single protein sequence."
    )
    parser.add_argument("--fasta", type=Path, required=True)

    # Embedding options
    parser.add_argument("--esm-local-checkpoint", type=Path, default=None)
    parser.add_argument("--max-aa-per-batch", type=int, default=4000)
    parser.add_argument("--esm-batch-size", type=int, default=8)

    # Fold options
    parser.add_argument("--pdb-output", type=Path)
    parser.add_argument("--fold-local-only", action="store_true")
    parser.add_argument("--fold-chunk-size", type=int, default=64)

    # InterProScan options
    parser.add_argument(
        "--interpro-email", type=str, default="antoine.toffano@lirmm.fr"
    )
    parser.add_argument("--interpro-timeout", type=int, default=300)
    parser.add_argument("--interpro-poll-interval", type=int, default=20)
    parser.add_argument(
        "--interpro-vocab",
        type=Path,
        default=Path("./data/swissprot/2024_01/protein_graphs/interpro_vocab.pkl"),
    )

    # DIAMOND options
    parser.add_argument("--diamond-bin", type=str, default="diamond")
    parser.add_argument("--diamond-db", type=Path)
    parser.add_argument("--diamond-out", type=Path)
    parser.add_argument("--diamond-topk", type=int, default=25)
    parser.add_argument("--diamond-evalue", type=float, default=1e-5)

    # Graph options
    parser.add_argument("--graph-path", type=Path)
    parser.add_argument("--contact-cutoff", type=float, default=10.0)
    parser.add_argument("--contact-chunk", type=int, default=512)

    parser.add_argument("--fold-child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def read_protein_from_fasta(fasta: Path) -> Tuple[str, str]:
    records = list(SeqIO.parse(str(fasta), "fasta"))
    if not records:
        raise ValueError(f"No records found in FASTA {fasta}.")
    if len(records) > 1:
        raise ValueError("FASTA must contain exactly one sequence for inference.")
    record = records[0]
    seq = str(record.seq).strip().upper()
    if not seq:
        raise ValueError("Sequence extracted from FASTA is empty.")
    return record.id, seq


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def embed_sequence(
    protein_id: str,
    sequence: str,
    checkpoint: Optional[Path],
    max_aa: int,
    esm_batch_size: int,
) -> torch.Tensor:
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
    protein_id: Optional[str],
    fasta_path: Path,
    output_path: Path,
    local_only: bool,
    chunk_size: int,
    child: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if child:
        from transformers import AutoTokenizer, EsmForProteinFolding
        from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
        from transformers.models.esm.openfold_utils.protein import (
            Protein as OFProtein,
            to_pdb,
        )

        fasta_protein_id, sequence = read_protein_from_fasta(fasta_path)
        if protein_id and protein_id != fasta_protein_id:
            logger.warning(
                "Overriding provided protein id %s with FASTA header %s.",
                protein_id,
                fasta_protein_id,
            )
        protein_id = fasta_protein_id

        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/esmfold_v1", local_files_only=local_only
        )
        model = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1",
            low_cpu_mem_usage=True,
            local_files_only=local_only,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        model.esm = model.esm.half()
        model.trunk.set_chunk_size(chunk_size)

        toks = tokenizer(
            [sequence], return_tensors="pt", add_special_tokens=False, padding=True
        )
        model_inputs = {k: v.to(device) for k, v in toks.items()}

        with torch.no_grad():
            outputs = model(**model_inputs)

        final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
        outputs_np = {k: v.to("cpu").numpy() for k, v in outputs.items()}
        final_atom_positions = final_atom_positions.cpu().numpy()
        final_atom_mask = outputs_np["atom37_atom_exists"]

        length = len(sequence)
        protein = OFProtein(
            aatype=outputs_np["aatype"][0][:length],
            atom_positions=final_atom_positions[0][:length],
            atom_mask=final_atom_mask[0][:length],
            residue_index=outputs_np["residue_index"][0][:length] + 1,
            b_factors=outputs_np["plddt"][0][:length],
            chain_index=(
                outputs_np["chain_index"][0][:length]
                if "chain_index" in outputs_np
                else None
            ),
        )
        pdb_str = to_pdb(protein)
        output_path.write_text(pdb_str, encoding="utf-8")
        logger.info("Saved PDB to %s.", output_path)
        return

    script_path = Path(__file__).resolve()
    cmd = [
        "conda",
        "run",
        "-n",
        "esmfold",
        "python",
        str(script_path),
        "--fasta",
        str(fasta_path),
        "--pdb-output",
        str(output_path),
        "--fold-chunk-size",
        str(chunk_size),
        "--fold-child",
    ]
    if protein_id:
        cmd.extend(["--protein-id", protein_id])
    if local_only:
        cmd.append("--fold-local-only")

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
        "--max-target-seqs",
        str(topk),
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
    seq_len = len(sequence)
    if embedding.shape[0] != seq_len:
        raise ValueError(
            f"Embedding length {embedding.shape[0]} does not match sequence length {seq_len}."
        )
    coords = load_ca_coordinates(pdb_path)
    if coords.size(0) < seq_len:
        logger.warning(
            "CA coordinates (%d) shorter than sequence (%d); truncating.",
            coords.size(0),
            seq_len,
        )
    coords = coords[:seq_len]

    edge_index, edge_attr = build_close_contact_edges(
        coords, cutoff=contact_cutoff, chunk_size=contact_chunk
    )

    data = HeteroData()
    data["aa"].x = embedding
    data["aa"].num_nodes = seq_len

    data["protein"].num_nodes = 1
    data["protein"].protein_id = protein_id
    data["protein"].sequence = sequence
    data["protein"].sequence_length = seq_len
    data["protein"].interpro = interpro_vec
    data["protein"].ipr_accessions = interpro_hits

    aa_indices = torch.arange(seq_len)
    data["aa", "belongs_to", "protein"].edge_index = torch.stack(
        [aa_indices, torch.zeros(seq_len, dtype=torch.long)]
    )
    data["protein", "has", "aa"].edge_index = torch.stack(
        [torch.zeros(seq_len, dtype=torch.long), aa_indices]
    )

    data["aa", "close_to", "aa"].edge_index = edge_index
    data["aa", "close_to", "aa"].edge_attr = edge_attr
    return data


def save_metadata(output_dir: Path, info: Dict[str, object]) -> None:
    (output_dir / "metadata.json").write_text(
        json.dumps(info, indent=2, default=str), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()

    protein_id, sequence = read_protein_from_fasta(args.fasta)

    if args.fold_child:
        if not args.pdb_output:
            raise ValueError("fold-child execution requires --pdb-output.")
        run_esmfold(
            protein_id=protein_id,
            fasta_path=args.fasta,
            output_path=args.pdb_output,
            local_only=args.fold_local_only,
            chunk_size=args.fold_chunk_size,
            child=True,
        )
        return

    if not args.diamond_db:
        raise ValueError("--diamond-db is required for DIAMOND alignment.")

    output_dir = ensure_output_dir(Path(f"./data/tmp_{protein_id}"))
    logger.info("Using output directory %s.", output_dir)

    protein_fasta_path = output_dir / f"{protein_id}.fasta"
    protein_fasta_path.write_text(f">{protein_id}\n{sequence}\n", encoding="utf-8")

    embedding = embed_sequence(
        protein_id,
        sequence,
        args.esm_local_checkpoint,
        args.max_aa_per_batch,
        args.esm_batch_size,
    )
    embedding_path = output_dir / f"{protein_id}_embedding.pt"
    torch.save(embedding, embedding_path)

    pdb_filename = (
        args.pdb_output.name if args.pdb_output else f"{protein_id}_esmfold.pdb"
    )
    pdb_path = output_dir / pdb_filename
    run_esmfold(
        protein_id=protein_id,
        fasta_path=protein_fasta_path,
        output_path=pdb_path,
        local_only=args.fold_local_only,
        chunk_size=args.fold_chunk_size,
    )

    logger.info("Querying InterProScan.")
    ipr_hits = submit_interproscan(
        sequence,
        email=args.interpro_email,
        timeout=args.interpro_timeout,
        poll_interval=args.interpro_poll_interval,
    )
    logger.info("InterProScan returned %d accessions.", len(ipr_hits))
    ipr_mapping, ipr_size = load_interpro_vocab(args.interpro_vocab)
    interpro_vec, missing_ipr = interpro_vector(ipr_hits, ipr_mapping, ipr_size)
    if missing_ipr:
        logger.warning(
            "InterPro accessions missing from vocab: %s", ", ".join(missing_ipr)
        )

    diamond_filename = (
        args.diamond_out.name if args.diamond_out else f"{protein_id}_diamond.tsv"
    )
    diamond_out = output_dir / diamond_filename
    diamond_query_path = run_diamond_alignment(
        protein_id,
        sequence,
        args.diamond_bin,
        args.diamond_db,
        diamond_out,
        args.diamond_topk,
        args.diamond_evalue,
        output_dir,
    )

    graph_filename = (
        args.graph_path.name if args.graph_path else f"{protein_id}_graph.pt"
    )
    graph_path = output_dir / graph_filename
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
    logger.info("Graph saved to %s.", graph_path)

    save_metadata(
        output_dir,
        {
            "protein_id": protein_id,
            "sequence_length": len(sequence),
            "embedding_path": embedding_path,
            "pdb_path": pdb_path,
            "diamond_output": diamond_out,
            "diamond_query_fasta": diamond_query_path,
            "graph_path": graph_path,
            "interpro_hits": ipr_hits,
            "interpro_missing": missing_ipr,
        },
    )
    logger.info("Pipeline completed for %s.", protein_id)


if __name__ == "__main__":
    main()

# Example usage:
# python src/data/inference_preprocess.py --fasta path/to/your_sequence.fasta --pdb-output path/to/output_structure.pdb --diamond-db path/to/diamond_db.dmnd
