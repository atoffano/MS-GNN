import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import torch
from Bio.PDB import PDBParser

warnings.filterwarnings("ignore")

# from src.utils.visualize import _download_alphafold

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PDB_DIR = Path(
    "/lustre/fsn1/projects/rech/dqy/uki62ne/tempdata/PFP_layer/data/swissprot/2024_01/alphafold_pdb"
)
GRAPH_DIR = Path(
    "/lustre/fsn1/projects/rech/dqy/uki62ne/tempdata/PFP_layer/data/swissprot/2024_01/protein_graphs"
)
STRUCTURE_MISSING_PATH = GRAPH_DIR / "structure_missing.fasta"
PDB_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# PDB_DIR = Path("./data/swissprot/2024_01/alphafold_pdb")
# PDB_DIR.mkdir(parents=True, exist_ok=True)
# GRAPH_DIR = Path("./data/swissprot/2024_01/protein_graphs")

# PDB_DIR = Path("./data/toy/alphafold_pdb")
# PDB_DIR.mkdir(parents=True, exist_ok=True)
# GRAPH_DIR = Path("./data/toy/protein_graphs")


def load_ca_coordinates(pdb_path: Path) -> torch.Tensor:
    """Return CA coordinates (N x 3) as torch tensor."""
    parser = PDBParser(QUIET=True)
    structure_id = pdb_path.stem
    structure = parser.get_structure(structure_id, str(pdb_path))
    ca_coords = []
    for model in structure:
        for chain in model:
            residues = sorted(
                (res for res in chain if "CA" in res),
                key=lambda res: res.id[1],
            )
            for residue in residues:
                ca_coords.append(residue["CA"].get_coord())
        break  # single model sufficient
    if not ca_coords:
        raise ValueError(f"No CA atoms found in {pdb_path}")
    return torch.tensor(ca_coords, dtype=torch.float32)


def build_close_contact_edges(
    coords: torch.Tensor,
    cutoff: float = 10.0,
    chunk_size: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return directed edge_index (2 x E) and edge_attr (E,) for residue pairs within cutoff Å."""
    n = coords.size(0)
    if n <= 1:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )

    edges: List[Tuple[int, int]] = []
    distances: List[float] = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sub = coords[start:end]
        dists_sq = torch.cdist(sub, coords, p=2.0)

        for i in range(end - start):
            mask = (dists_sq[i] <= cutoff) & (dists_sq[i] > 0)
            if not mask.any():
                continue
            cols = torch.nonzero(mask, as_tuple=False).view(-1)
            dists = dists_sq[i, cols].sqrt()
            src_idx = start + i
            edges.extend((src_idx, int(c.item())) for c in cols)
            distances.extend(float(d.item()) for d in dists)

    if not edges:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(distances, dtype=torch.float32)
    return edge_index, edge_attr


def report_close_contact_stats(
    protein_id: str,
    edge_index: torch.Tensor,
    num_nodes: int,
    cutoff: float = 10.0,
) -> None:
    total_pairs = num_nodes * (num_nodes - 1) // 2
    undirected_links = edge_index.size(1) // 2
    ratio = undirected_links / total_pairs if total_pairs else 0.0
    logger.info(
        "%s: created %d close-contact links, ratio %.4f (cutoff %.1fÅ)",
        protein_id,
        undirected_links,
        ratio,
        cutoff,
    )


def process_graph(graph_path: Path) -> None:
    data = torch.load(graph_path, map_location="cpu")
    protein_id = graph_path.stem
    pdb_path = PDB_DIR / f"AF-{protein_id}-F1-model_v6.pdb"
    if not pdb_path.is_file():
        try:
            sequence = data["sequence"]
        except (KeyError, TypeError):
            sequence = getattr(data, "sequence", None)
        if sequence:
            with open(STRUCTURE_MISSING_PATH, "a") as fasta_out:
                fasta_out.write(f">{protein_id}\n{sequence}\n")
        logger.warning("No structure found for %s; skipping.", protein_id)
        return

    ca_coords = load_ca_coordinates(pdb_path)

    num_nodes = data["aa"].num_nodes
    if ca_coords.size(0) < num_nodes:
        logger.warning(
            "%s: PDB residues (%d) fewer than AA nodes (%d); truncating.",
            protein_id,
            ca_coords.size(0),
            num_nodes,
        )
    coords = ca_coords[:num_nodes]

    edge_index, edge_attr = build_close_contact_edges(coords)
    report_close_contact_stats(protein_id, edge_index, num_nodes)

    data["aa", "close_to", "aa"].edge_index = edge_index
    data["aa", "close_to", "aa"].edge_attr = edge_attr
    torch.save(data, graph_path)


def main() -> None:
    graph_files = sorted(
        f
        for f in GRAPH_DIR.glob("*.pt")
        if f.stem not in {"metadata", "interpro_vocab", "go_vocab"}
    )
    if not graph_files:
        logger.info("No protein graphs found.")
        return

    logger.info("Found %d protein graphs to process.", len(graph_files))
    STRUCTURE_MISSING_PATH.touch(exist_ok=True)

    with ProcessPoolExecutor() as executor:
        future_map = {
            executor.submit(process_graph, graph_path): graph_path
            for graph_path in graph_files
        }
        with tqdm(total=len(future_map), desc="Processing graphs") as pbar:
            for future in as_completed(future_map):
                graph_path = future_map[future]
                try:
                    future.result()
                except Exception as exc:
                    logger.error(
                        "Failed to process %s: %s", graph_path.stem, exc, exc_info=True
                    )
                pbar.update(1)


if __name__ == "__main__":
    main()
