import argparse
import logging
import multiprocessing as mp
import os
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

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATA_ROOT = Path("./data/swissprot/2024_01")
OUTPUT_DIR = DATA_ROOT / "protein_graphs"
FASTA_PATH = DATA_ROOT / "swissprot_2024_01.fasta"
INTERPRO_TSV = DATA_ROOT / "swissprot_interpro_106_0.tsv"
GO_OBO_PATH = Path("./data/go.obo")
GO_ANNOTATION_TEMPLATE = DATA_ROOT / "swissprot_2024_01_{onto}_annotations.tsv"
GO_EXP_ANNOTATION_TEMPLATE = DATA_ROOT / "swissprot_2024_01_{onto}_exp_annotations.tsv"
EMBED_H5_PATH = DATA_ROOT / "swissprot_esm1b_per_aa.h5"

STRUCTURE_MISSING_PATH = OUTPUT_DIR / "structure_missing_rev.fasta"
CONTACT_CUTOFF = 10.0
CONTACT_CHUNK = 512

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STRUCTURE_MISSING_PATH.touch(exist_ok=True)


def build_and_save_protein_graph(
    protein_id: str,
    sequence: str,
    h5f: h5py.File,
    interpro_dict: Dict[str, torch.Tensor],
    go_annotations: Dict[str, Dict[str, Dict[str, List[str]]]],
    interpro_vocab_size: int,
) -> Tuple[bool, Optional[str], bool]:

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

        add_close_contact_edges(data, protein_id)
        torch.save(data, OUTPUT_DIR / f"{protein_id}.pt")
        return True, None, False
    except Exception as exc:
        return False, str(exc), False


def load_ca_coordinates(pdb_path: Path) -> torch.Tensor:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    ca_coords: List[np.ndarray] = []
    for model in structure:
        for chain in model:
            residues = sorted(
                (res for res in chain if "CA" in res),
                key=lambda res: res.id[1],
            )
            ca_coords.extend(residue["CA"].get_coord() for residue in residues)
        break
    if not ca_coords:
        raise ValueError(f"No CA atoms found in {pdb_path}")
    return torch.tensor(ca_coords, dtype=torch.float32)


def build_close_contact_edges(
    coords: torch.Tensor,
    cutoff: float = CONTACT_CUTOFF,
    chunk_size: int = CONTACT_CHUNK,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = coords.size(0)
    if n <= 1:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )

    edges: List[Tuple[int, int]] = []
    dists: List[float] = []

    for start in range(0, n, chunk_size):
        sub = coords[start : start + chunk_size]
        dist_sq = torch.cdist(sub, coords, p=2)
        for i in range(sub.size(0)):
            mask = (dist_sq[i] <= cutoff) & (dist_sq[i] > 0)
            cols = torch.nonzero(mask, as_tuple=False).view(-1)
            if cols.numel() == 0:
                continue
            src = start + i
            edges.extend((src, int(col)) for col in cols)
            dists.extend(float(dist_sq[i, col].sqrt().item()) for col in cols)

    if not edges:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(dists, dtype=torch.float32)
    return edge_index, edge_attr


def load_interpro_annotations() -> Tuple[Dict[str, torch.Tensor], int]:
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

    with open(OUTPUT_DIR / "interpro_vocab.pkl", "wb") as handle:
        pickle.dump({"ipr_to_idx": ipr_to_idx, "vocab_size": len(ipr_ids)}, handle)

    return interpro, len(ipr_ids)


def load_go_vocab() -> Dict[str, int]:
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

    with open(OUTPUT_DIR / "go_vocab.pkl", "wb") as handle:
        pickle.dump(vocab_info, handle)

    return {onto: info["vocab_size"] for onto, info in vocab_info.items()}


def load_go_annotations() -> Dict[str, Dict[str, Dict[str, List[str]]]]:
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
    seq_len = len(sequence)
    if embeddings.shape[0] != seq_len:
        logger.warning(
            "Embedding length mismatch for %s: %d vs %d",
            protein_id,
            embeddings.shape[0],
            seq_len,
        )
        return None

    data = HeteroData()
    data["aa"].x = embeddings
    data["aa"].num_nodes = seq_len
    data["protein"].num_nodes = 1

    data["protein"].interpro = (
        interpro_dict[protein_id].unsqueeze(0)
        if protein_id in interpro_dict
        else torch.zeros(1, interpro_vocab_size, dtype=torch.float32)
    )

    for onto in ["BPO", "CCO", "MFO"]:
        data["protein"][f"go_terms_{onto}"] = {
            "experimental": go_annotations.get(protein_id, {})
            .get(onto, {})
            .get("experimental", []),
            "curated": go_annotations.get(protein_id, {})
            .get(onto, {})
            .get("curated", []),
        }

    aa_indices = torch.arange(seq_len)
    protein_indices = torch.zeros(seq_len, dtype=torch.long)
    data["aa", "belongs_to", "protein"].edge_index = torch.stack(
        [aa_indices, protein_indices]
    )

    data["protein"].protein_id = protein_id
    data["protein"].sequence = sequence
    data["protein"].sequence_length = seq_len

    return data


def add_close_contact_edges(data: HeteroData, protein_id: str) -> None:
    af_path = DATA_ROOT / f"alphafold_pdb/AF-{protein_id}-F1-model_v6.pdb"
    esmfold_path = DATA_ROOT / f"esmfold_pdb/ESM-{protein_id}-model_v1.pdb"

    if not af_path.is_file():
        if not esmfold_path.is_file():
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
            return
        af_path = esmfold_path

    coords = load_ca_coordinates(af_path)
    num_nodes = data["aa"].num_nodes
    if coords.size(0) < num_nodes:
        logger.warning(
            "%s: PDB residues (%d) fewer than AA nodes (%d); truncating.",
            protein_id,
            coords.size(0),
            num_nodes,
        )
    coords = coords[:num_nodes]

    edge_index, edge_attr = build_close_contact_edges(
        coords, CONTACT_CUTOFF, CONTACT_CHUNK
    )
    data["aa", "close_to", "aa"].edge_index = edge_index
    data["aa", "close_to", "aa"].edge_attr = edge_attr

    undirected = edge_index.size(1) // 2
    total_pairs = num_nodes * (num_nodes - 1) // 2
    ratio = undirected / total_pairs if total_pairs else 0.0
    logger.info(
        "%s: %d close-contact links (%.4f of possible pairs)",
        protein_id,
        undirected,
        ratio,
    )


class ProteinGraphBuilder:
    def __init__(
        self,
        num_workers: Optional[int] = None,
    ):
        if not EMBED_H5_PATH.exists():
            raise FileNotFoundError(f"Embedding H5 not found: {EMBED_H5_PATH}")

        if num_workers and num_workers > 1:
            logger.warning("num_workers argument ignored; processing sequentially.")

        logger.info("Graph construction will run sequentially.")

        self.interpro_dict, self.interpro_vocab_size = load_interpro_annotations()
        logger.info("InterPro entries: %d", len(self.interpro_dict))

        self.go_annotations = load_go_annotations()
        logger.info("GO annotations for proteins: %d", len(self.go_annotations))
        self.go_vocab_sizes = load_go_vocab()
        logger.info("GO vocab sizes: %s", self.go_vocab_sizes)

    def process_fasta(
        self,
        fasta_path: Path,
    ) -> Tuple[int, int, int]:
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
