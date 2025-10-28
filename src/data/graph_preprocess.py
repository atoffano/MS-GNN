"""Complete graph preprocessing pipeline for protein datasets.

This module orchestrates the full preprocessing pipeline for creating protein
graphs from raw data, combining multiple data sources and processing steps:
- ESM embedding generation for residues
- 3D structure loading and contact computation
- InterPro domain annotation integration
- GO term annotation processing
- Protein-protein alignment via DIAMOND

Unlike create_graphs.py which builds individual graphs, this module manages
the entire preprocessing workflow with proper ordering and dependency handling.
"""

import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pickle
import torch
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.SeqRecord import SeqRecord
from torch_geometric.data import HeteroData
from tqdm.auto import tqdm
import esm

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DATA_ROOT = Path("./data/swissprot/2024_01")
OUTPUT_DIR = DATA_ROOT / "protein_graphs"
FASTA_PATH = DATA_ROOT / "swissprot_2024_01.fasta"
INTERPRO_TSV = DATA_ROOT / "swissprot_interpro_106_0.tsv"
GO_OBO_PATH = Path("./data/go.obo")
GO_ANNOTATION_TEMPLATE = DATA_ROOT / "swissprot_2024_01_{onto}_annotations.tsv"
GO_EXP_ANNOTATION_TEMPLATE = DATA_ROOT / "swissprot_2024_01_{onto}_exp_annotations.tsv"

STRUCTURE_MISSING_PATH = OUTPUT_DIR / "structure_missing.fasta"

ESM_CHECKPOINT = Path("../torch_cache/esm1b_t33_650M_UR50S.pt")
ESM_LAYER = 33
ESM_BATCH_SIZE = 8
CONTACT_CUTOFF = 10.0
CONTACT_CHUNK = 512

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STRUCTURE_MISSING_PATH.touch(exist_ok=True)

WORKER_INTERPRO: Dict[str, torch.Tensor] = {}
WORKER_GO: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
WORKER_INTERPRO_SIZE: int = 0


def _init_worker(interpro_dict, go_annotations, interpro_vocab_size):
    global WORKER_INTERPRO, WORKER_GO, WORKER_INTERPRO_SIZE
    WORKER_INTERPRO = interpro_dict
    WORKER_GO = go_annotations
    WORKER_INTERPRO_SIZE = interpro_vocab_size


def _build_and_save_protein_graph(
    args: Tuple[str, str, torch.Tensor],
) -> Tuple[str, bool, Optional[str]]:
    protein_id, sequence, embeddings = args
    try:
        data = build_protein_graph(
            protein_id,
            embeddings,
            sequence,
            WORKER_INTERPRO,
            WORKER_GO,
            WORKER_INTERPRO_SIZE,
        )
        if data is None:
            return protein_id, False, "Graph construction returned None"
        add_close_contact_edges(data, protein_id)
        torch.save(data, OUTPUT_DIR / f"{protein_id}.pt")
        return protein_id, True, None
    except Exception as exc:
        logger.error("Worker failure for %s: %s", protein_id, exc)
        return protein_id, False, str(exc)


# ---------------------------------------------------------------------
# AlphaFold utilities
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# ESM embedding and annotation handling
# ---------------------------------------------------------------------
def load_esm_model(
    local_checkpoint: Optional[Path] = None,
) -> Tuple[torch.nn.Module, esm.Alphabet]:
    if local_checkpoint:
        checkpoint = Path(local_checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Local ESM checkpoint not found: {checkpoint}")
        logger.info("Loading ESM model from local checkpoint %s", checkpoint)
        return esm.pretrained.load_model_and_alphabet_local(checkpoint)

    model_name = "esm1b_t33_650M_UR50S"
    logger.info("Downloading ESM model %s", model_name)
    return esm.pretrained.esm1b_t33_650M_UR50S()


def load_interpro_annotations() -> Tuple[Dict[str, torch.Tensor], int]:
    df = pd.read_csv(INTERPRO_TSV, sep="\t")
    ipr_ids = sorted(df["IPR"].unique())
    ipr_to_idx = {ipr: i for i, ipr in enumerate(ipr_ids)}
    grouped = df.groupby("ID")["IPR"].apply(list)

    interpro = {}
    for pid, terms in grouped.items():
        vec = torch.zeros(len(ipr_ids), dtype=torch.float32)
        for ipr in terms:
            vec[ipr_to_idx[ipr]] = 1.0
        interpro[pid] = vec

    with open(OUTPUT_DIR / "interpro_vocab.pkl", "wb") as f:
        pickle.dump({"ipr_to_idx": ipr_to_idx, "vocab_size": len(ipr_ids)}, f)

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

    with open(GO_OBO_PATH, "r") as handle:
        for line in handle:
            line = line.strip()
            if line == "[Term]":
                current_id = None
                current_ns = None
            elif line.startswith("id: GO:"):
                current_id = line.split("id: ")[1]
            elif line.startswith("namespace: "):
                current_ns = namespace_map.get(line.split("namespace: ")[1])
            elif line == "" and current_id and current_ns:
                terms[current_ns].add(current_id)

    vocab_info = {}
    for onto, onto_terms in terms.items():
        sorted_terms = sorted(onto_terms)
        go_to_idx = {go: i for i, go in enumerate(sorted_terms)}
        vocab_info[onto] = {
            "terms": sorted_terms,
            "go_to_idx": go_to_idx,
            "vocab_size": len(sorted_terms),
        }

    with open(OUTPUT_DIR / "go_vocab.pkl", "wb") as f:
        pickle.dump(vocab_info, f)

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
                terms = [t.strip() for t in row["term"].split(";") if t.strip()]
                annotations.setdefault(pid, {}).setdefault(onto, {})[kind] = terms
    return annotations


# ---------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------
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
            with open(STRUCTURE_MISSING_PATH, "a") as fasta_out:
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
        else:
            af_path = esmfold_path  # Use ESMFold structure if AlphaFold is missing

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


# ---------------------------------------------------------------------
# Embedding executor
# ---------------------------------------------------------------------
class ProteinGraphPreprocessor:
    def __init__(
        self,
        local_checkpoint: Optional[Path] = None,
        embed_batch_size: int = DEFAULT_SEQUENCE_BATCH_SIZE,
        num_workers: Optional[int] = None,
    ):
        self.output_dir = OUTPUT_DIR
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device %s", self.device)
        self.mp_context = mp.get_context("spawn")

        self.output_dir = OUTPUT_DIR
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device %s", self.device)

        self.model, self.alphabet = load_esm_model(local_checkpoint=local_checkpoint)
        self.model = self.model.to(self.device).eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        logger.info("Loaded ESM model to %s", self.device)

        self.max_tokens = getattr(
            getattr(self.model, "args", None), "max_positions", None
        )
        self.window_len = (self.max_tokens - 2) if self.max_tokens else None
        self.window_stride = max(1, self.window_len // 2) if self.window_len else None
        logger.info(
            "ESM window length: %s",
            self.window_len if self.window_len else "unbounded",
        )

        self.interpro_dict, self.interpro_vocab_size = load_interpro_annotations()
        logger.info("InterPro entries: %d", len(self.interpro_dict))

        self.go_annotations = load_go_annotations()
        logger.info("GO annotations for proteins: %d", len(self.go_annotations))
        self.go_vocab_sizes = load_go_vocab()
        logger.info("GO vocab sizes: %s", self.go_vocab_sizes)

        self.embed_batch_size = max(1, embed_batch_size)
        available_cpus = os.cpu_count() or 1
        resolved_workers = (
            num_workers
            if num_workers and num_workers > 0
            else max(1, available_cpus - 1)
        )
        self.num_workers = resolved_workers
        self.max_inflight = max(self.num_workers, self.num_workers * 2)
        logger.info(
            "Embedding batch size: %d | CPU workers: %d | Max in-flight jobs: %d",
            self.embed_batch_size,
            self.num_workers,
            self.max_inflight,
        )

    def embed_sequence(self, seq_id: str, seq: str) -> torch.Tensor:
        L = len(seq)
        if L == 0:
            raise ValueError(f"Sequence {seq_id} is empty")

        if not self.window_len or L <= self.window_len:
            batch = [(seq_id, seq)]
            _, _, tokens = self.batch_converter(batch)
            tokens = tokens.to(self.device)
            with torch.no_grad():
                reps = self.model(
                    tokens, repr_layers=[ESM_LAYER], return_contacts=False
                )["representations"][ESM_LAYER]
            return reps[0, 1 : 1 + L].cpu().to(torch.float32).clone()

        starts = list(range(0, L, self.window_stride))
        if starts[-1] + self.window_len < L:
            starts.append(L - self.window_len)
        starts = sorted(set(starts))
        windows = [
            (
                s,
                min(s + self.window_len, L),
                seq[s : min(s + self.window_len, L)],
            )
            for s in starts
        ]

        acc = None
        counts = np.zeros((L,), dtype=np.int32)

        for i in range(0, len(windows), ESM_BATCH_SIZE):
            batch_windows = windows[i : i + ESM_BATCH_SIZE]
            batch = [
                (f"{seq_id}_w{i+j}", window_seq)
                for j, (_, _, window_seq) in enumerate(batch_windows)
            ]
            _, _, tokens = self.batch_converter(batch)
            tokens = tokens.to(self.device)
            with torch.no_grad():
                reps = (
                    self.model(tokens, repr_layers=[ESM_LAYER], return_contacts=False)[
                        "representations"
                    ][ESM_LAYER]
                    .cpu()
                    .numpy()
                )
            for j, (start, end, _) in enumerate(batch_windows):
                sub_len = end - start
                sub_emb = reps[j, 1 : 1 + sub_len]
                if acc is None:
                    acc = np.zeros((L, sub_emb.shape[-1]), dtype=np.float64)
                acc[start:end] += sub_emb
                counts[start:end] += 1

        counts[counts == 0] = 1
        emb = (acc / counts[:, None]).astype(np.float32)
        return torch.from_numpy(emb).clone()

    def embed_batch(self, records: List[SeqRecord]) -> Dict[str, torch.Tensor]:
        results: Dict[str, torch.Tensor] = {}
        short_sequences: List[Tuple[str, str]] = []

        for record in records:
            seq_id = record.id
            sequence = str(record.seq)
            if not sequence:
                raise ValueError(f"Sequence {seq_id} is empty")
            if self.window_len and len(sequence) > self.window_len:
                results[seq_id] = self.embed_sequence(seq_id, sequence)
            else:
                short_sequences.append((seq_id, sequence))

        for start in range(0, len(short_sequences), ESM_BATCH_SIZE):
            chunk = short_sequences[start : start + ESM_BATCH_SIZE]
            batch = [(seq_id, seq) for seq_id, seq in chunk]
            _, _, tokens = self.batch_converter(batch)
            tokens = tokens.to(self.device)
            with torch.no_grad():
                reps = (
                    self.model(tokens, repr_layers=[ESM_LAYER], return_contacts=False)[
                        "representations"
                    ][ESM_LAYER]
                    .detach()
                    .cpu()
                )
            for (seq_id, seq), rep in zip(chunk, reps):
                L = len(seq)
                results[seq_id] = rep[1 : 1 + L].to(torch.float32).clone()

        return results

    def process_fasta(self) -> Tuple[int, int]:
        if not FASTA_PATH.exists():
            raise FileNotFoundError(f"FASTA not found: {FASTA_PATH}")

        records = list(SeqIO.parse(str(FASTA_PATH), "fasta"))
        if not records:
            logger.warning("No sequences found in FASTA %s", FASTA_PATH)
            return 0, 0

        success = 0
        failed = 0
        pending = set()

        with ProcessPoolExecutor(
            max_workers=self.num_workers,
            mp_context=self.mp_context,
            initializer=_init_worker,
            initargs=(
                self.interpro_dict,
                self.go_annotations,
                self.interpro_vocab_size,
            ),
        ) as executor, tqdm(
            total=len(records),
            desc="Embedding + graph building",
            unit="protein",
        ) as progress:
            for batch_start in range(0, len(records), self.embed_batch_size):
                batch_records = records[
                    batch_start : batch_start + self.embed_batch_size
                ]
                try:
                    embeddings_map = self.embed_batch(batch_records)
                except Exception as exc:
                    logger.error(
                        "Failed to embed batch starting at %d: %s", batch_start, exc
                    )
                    failed += len(batch_records)
                    progress.update(len(batch_records))
                    continue

                for record in batch_records:
                    protein_id = record.id
                    sequence = str(record.seq)
                    embeddings = embeddings_map.get(protein_id)
                    if embeddings is None:
                        logger.error("Missing embeddings for %s", protein_id)
                        failed += 1
                        progress.update(1)
                        continue
                    future = executor.submit(
                        _build_and_save_protein_graph,
                        (protein_id, sequence, embeddings),
                    )
                    pending.add(future)
                    if len(pending) >= self.max_inflight:
                        done_future = next(as_completed(pending))
                        protein_id, ok, error = done_future.result()
                        if ok:
                            success += 1
                        else:
                            failed += 1
                            if error:
                                logger.error(
                                    "Graph build failed for %s: %s", protein_id, error
                                )
                        progress.update(1)
                        pending.remove(done_future)

            if pending:
                for done_future in as_completed(pending):
                    protein_id, ok, error = done_future.result()
                    if ok:
                        success += 1
                    else:
                        failed += 1
                        if error:
                            logger.error(
                                "Graph build failed for %s: %s", protein_id, error
                            )
                    progress.update(1)
                pending.clear()

        logger.info("Finished embeddings: success=%d failed=%d", success, failed)
        return success, failed


def main():
    parser = argparse.ArgumentParser(description="Protein graph preprocessing")
    parser.add_argument(
        "--local",
        type=Path,
        default=None,
        help=f"Path to a local ESM checkpoint (defaults to online download).",
    )
    args = parser.parse_args()

    logger.info("Starting protein graph preprocessing")
    preprocessor = ProteinGraphPreprocessor(local_checkpoint=args.local)
    success, failed = preprocessor.process_fasta()
    logger.info("Processing complete - Successful: %d, Failed: %d", success, failed)
    print("Embedding preprocessing complete!")


if __name__ == "__main__":
    main()
