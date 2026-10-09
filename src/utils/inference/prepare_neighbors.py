"""Neighbor identification and .pt graph preparation for inference.

Discovers the DIAMOND homology + STRING-DB neighborhood of query proteins
and ensures each neighbor has a pre-built .pt graph file.  Neighbors that
already have a .pt in the standard protein_graphs directory are reused;
missing ones can optionally be built on-the-fly (ESM + AlphaFold + graph).

This module can be used as:
- An importable function: ``identify_neighbors()``
- A standalone CLI: ``python -m src.utils.inference.prepare_neighbors``
"""

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import requests
import torch

_STRING_MAPPING_CACHE: Dict[Path, Tuple[Dict[str, str], Dict[str, str]]] = {}
_HTTP_SESSION = requests.Session()

from src.utils.constants import (
    DIAMOND_ALIGNMENT,
    DIAMOND_DEFAULT_EVALUE,
    DIAMOND_EXECUTABLE,
    DIAMOND_SWISSPROT_DB,
    INFERENCE_OUTPUT_DIR,
    PID_MAPPING,
    PROTEIN_GRAPHS_DIR,
    SWISSPROT_FASTA,
    STRINGDB_PATH,
    STRINGDB_SWISSPROT_MAPPING,
)
from src.utils.inference.resolve_ids import ProteinInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DIAMOND alignment
# ---------------------------------------------------------------------------

def run_diamond(
    protein_info: ProteinInfo,
    workspace: Path,
    diamond_bin: str = DIAMOND_EXECUTABLE,
    database: Path = DIAMOND_SWISSPROT_DB,
    evalue: float = DIAMOND_DEFAULT_EVALUE,
    force: bool = False,
) -> Path:
    """Run DIAMOND alignment for a single protein against SwissProt DB.

    Args:
        protein_info: Resolved protein information.
        workspace: Directory for temporary files and output.
        diamond_bin: Path to DIAMOND executable.
        database: Path to DIAMOND database (.dmnd).
        evalue: E-value threshold.
        force: If True, re-run alignment even if output exists.

    Returns:
        Path to the DIAMOND output TSV (blast6 format).
    """
    workspace.mkdir(parents=True, exist_ok=True)
    query_fasta = workspace / f"{protein_info.accession}_query.fasta"
    output_path = workspace / f"{protein_info.accession}_diamond.tsv"

    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        logger.info("DIAMOND output already exists at %s, reusing.", output_path)
        return output_path

    query_fasta.write_text(
        f">{protein_info.accession}\n{protein_info.sequence}\n", encoding="utf-8"
    )

    cmd = [
        diamond_bin,
        "blastp",
        "--db", str(database),
        "--query", str(query_fasta),
        "--out", str(output_path),
        "--outfmt", "6",
        "--evalue", str(evalue),
    ]

    logger.info("Running DIAMOND: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    logger.info("DIAMOND results → %s", output_path)
    return output_path


def run_diamond_batch(
    protein_infos: List[ProteinInfo],
    workspace: Path,
    diamond_bin: str = DIAMOND_EXECUTABLE,
    database: Path = DIAMOND_SWISSPROT_DB,
    evalue: float = DIAMOND_DEFAULT_EVALUE,
    threads: int = 8,
    force: bool = False,
) -> None:
    """Run DIAMOND alignment in a single multi-threaded batched invocation for multiple proteins."""
    workspace.mkdir(parents=True, exist_ok=True)
    pending = []
    for p in protein_infos:
        tsv = workspace / f"{p.accession}_diamond.tsv"
        if force or not tsv.exists() or tsv.stat().st_size == 0:
            pending.append(p)

    if not pending:
        logger.info("All %d proteins already have DIAMOND outputs, skipping batched run.", len(protein_infos))
        return

    batch_fasta = workspace / "batch_queries.fasta"
    with open(batch_fasta, "w", encoding="utf-8") as f:
        for p in pending:
            f.write(f">{p.accession}\n{p.sequence}\n")

    batch_tsv = workspace / "batch_diamond.tsv"
    cmd = [
        diamond_bin,
        "blastp",
        "--db", str(database),
        "--query", str(batch_fasta),
        "--out", str(batch_tsv),
        "--outfmt", "6",
        "--evalue", str(evalue),
        "--threads", str(threads),
    ]
    logger.info("Running batched DIAMOND for %d proteins (threads=%d)...", len(pending), threads)
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    per_query_lines: Dict[str, List[str]] = {p.accession: [] for p in pending}
    if batch_tsv.exists():
        with open(batch_tsv, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split("\t", 1)
                qseqid = parts[0]
                if qseqid in per_query_lines:
                    per_query_lines[qseqid].append(line)

    for acc, lines in per_query_lines.items():
        out_tsv = workspace / f"{acc}_diamond.tsv"
        with open(out_tsv, "w", encoding="utf-8") as f:
            f.writelines(lines)
    logger.info("Saved individual DIAMOND outputs for %d proteins.", len(pending))


def parse_diamond_output(
    diamond_tsv: Path,
    max_hits: int = 500,
) -> Tuple[Optional[float], List[Tuple[str, float]]]:
    """Parse DIAMOND blast6 output and return (self_bitscore, hit_list).

    Args:
        diamond_tsv: Path to DIAMOND output file.
        max_hits: Maximum number of non-self hits to return.

    Returns:
        Tuple of (self_bitscore or None, list of (accession, bitscore) tuples).
    """
    if not diamond_tsv.exists() or diamond_tsv.stat().st_size == 0:
        logger.warning("DIAMOND output %s is empty or missing.", diamond_tsv)
        return None, []

    cols = [
        "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
        "qstart", "qend", "sstart", "send", "evalue", "bitscore",
    ]
    df = pd.read_csv(diamond_tsv, sep="\t", header=None, names=cols)

    # Capture true self-alignment hit
    self_df = df[df["qseqid"] == df["sseqid"]]
    self_bitscore = float(self_df.iloc[0]["bitscore"]) if not self_df.empty else None

    # Filter out self-hits for neighbor list
    df_nonself = df[df["qseqid"] != df["sseqid"]]
    df_nonself = df_nonself.sort_values("bitscore", ascending=False).head(max_hits)

    hits = [(row["sseqid"], float(row["bitscore"])) for _, row in df_nonself.iterrows()]
    logger.info(
        "Parsed %d DIAMOND hits (self_bitscore=%s, max %d).",
        len(hits),
        f"{self_bitscore:.1f}" if self_bitscore else "None",
        max_hits,
    )
    return self_bitscore, hits


# ---------------------------------------------------------------------------
# STRING-DB lookup (API with local fallback)
# ---------------------------------------------------------------------------

_STRING_DB_API_URL = "https://string-db.org/api/tsv/network"


def lookup_stringdb_neighbors(
    protein_info: ProteinInfo,
    stringdb_path: Path = STRINGDB_PATH,
    mapping_path: Path = STRINGDB_SWISSPROT_MAPPING,
    max_neighbors: int = 500,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, str, float]]]:
    """Look up STRING-DB interaction partners using the REST API with fallback.

    Args:
        protein_info: Resolved protein information.
        stringdb_path: Path to swissprot_stringdb.tsv (fallback).
        mapping_path: Path to idmapping_swissprot_stringdb.tsv.
        max_neighbors: Maximum number of STRING-DB neighbors.

    Returns:
        Tuple of:
            - direct_partners: list of (uniprot_accession, combined_score in [0, 1000])
            - pairwise_edges: list of (acc_a, acc_b, combined_score in [0, 1000])
    """
    global _STRING_MAPPING_CACHE
    if mapping_path not in _STRING_MAPPING_CACHE:
        mapping_df = pd.read_csv(mapping_path, sep="\t", usecols=["From", "To"])
        acc_to_string = mapping_df.set_index("From")["To"].to_dict()
        string_to_acc = {v: k for k, v in acc_to_string.items()}
        _STRING_MAPPING_CACHE[mapping_path] = (acc_to_string, string_to_acc)
    else:
        acc_to_string, string_to_acc = _STRING_MAPPING_CACHE[mapping_path]

    query_string_id = acc_to_string.get(protein_info.accession)
    if query_string_id is None:
        logger.info(
            "Protein %s not found in STRING-DB mapping.", protein_info.accession
        )
        return [], []

    partners: List[Tuple[str, float]] = []
    pairwise_edges: List[Tuple[str, str, float]] = []

    # 1. Try fast STRING-DB REST API query
    try:
        resp = _HTTP_SESSION.get(
            _STRING_DB_API_URL,
            params={"identifiers": protein_info.accession},
            timeout=8,
        )
        if resp.status_code == 200 and resp.text.strip():
            lines = resp.text.strip().splitlines()
            partner_map: Dict[str, float] = {}
            for line in lines[1:]:  # skip header
                cols = line.split("\t")
                if len(cols) < 6:
                    continue
                s_a, s_b = cols[0], cols[1]
                # Scale from [0.0, 1.0] to [0, 1000] integer scale of swissprot_stringdb.tsv
                raw_score = float(cols[5]) * 1000.0

                acc_a = string_to_acc.get(s_a)
                acc_b = string_to_acc.get(s_b)

                # Direct edge to query
                if s_a == query_string_id and acc_b and acc_b != protein_info.accession:
                    if acc_b not in partner_map or raw_score > partner_map[acc_b]:
                        partner_map[acc_b] = raw_score
                elif s_b == query_string_id and acc_a and acc_a != protein_info.accession:
                    if acc_a not in partner_map or raw_score > partner_map[acc_a]:
                        partner_map[acc_a] = raw_score

                # Pairwise edge between interactors (induced subgraph)
                if acc_a and acc_b and acc_a != acc_b:
                    pairwise_edges.append((acc_a, acc_b, raw_score))

            partners = sorted(partner_map.items(), key=lambda x: x[1], reverse=True)[:max_neighbors]
            logger.info(
                "Found %d STRING-DB neighbors for %s (API query, %d pairwise network edges).",
                len(partners),
                protein_info.accession,
                len(pairwise_edges),
            )
            return partners, pairwise_edges
    except Exception as exc:
        logger.warning("STRING-DB REST API query failed: %s. Using local fallback.", exc)

    # 2. Local fallback using ripgrep
    if stringdb_path.exists():
        try:
            cmd = ["rg", "-N", f"^{query_string_id}\\t", str(stringdb_path)]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if res.returncode == 0 and res.stdout:
                partner_map = {}
                for line in res.stdout.splitlines():
                    cols = line.split("\t")
                    if len(cols) >= 10:
                        p2 = cols[1]
                        score = float(cols[9])
                        acc2 = string_to_acc.get(p2)
                        if acc2 and acc2 != protein_info.accession:
                            if acc2 not in partner_map or score > partner_map[acc2]:
                                partner_map[acc2] = score
                partners = sorted(partner_map.items(), key=lambda x: x[1], reverse=True)[:max_neighbors]
        except Exception as exc:
            logger.warning("Local ripgrep STRING-DB search failed: %s", exc)

    return partners, pairwise_edges


_ALN_INDEX_CACHE: Optional[Dict[str, Dict[str, float]]] = None


def get_alignment_index(alignment_tsv: Path = DIAMOND_ALIGNMENT) -> Dict[str, Dict[str, float]]:
    """Load alignment TSV into an in-memory index for microsecond induced edge queries."""
    global _ALN_INDEX_CACHE
    if _ALN_INDEX_CACHE is not None:
        return _ALN_INDEX_CACHE

    alignment_tsv = Path(alignment_tsv)
    if not alignment_tsv.exists():
        logger.warning("Alignment TSV not found at %s", alignment_tsv)
        return {}

    import pickle
    import time
    from collections import defaultdict

    # Fast load from pickle cache if present
    pkl_cache = alignment_tsv.with_name(f"{alignment_tsv.stem}_idx.pkl")
    if pkl_cache.exists():
        try:
            t0 = time.time()
            logger.info("Loading cached DIAMOND alignment index from %s...", pkl_cache)
            with open(pkl_cache, "rb") as f:
                _ALN_INDEX_CACHE = pickle.load(f)
            logger.info(
                "Loaded cached alignment index in %.2fs for %d proteins.",
                time.time() - t0,
                len(_ALN_INDEX_CACHE),
            )
            return _ALN_INDEX_CACHE
        except Exception as exc:
            logger.warning("Failed to load cached alignment index (%s), re-parsing TSV.", exc)

    logger.info("Building in-memory DIAMOND alignment index from %s...", alignment_tsv)
    t0 = time.time()
    df = pd.read_csv(
        alignment_tsv,
        sep="\t",
        header=None,
        usecols=[0, 1, 11],
        names=["q", "s", "bit"],
    )
    q_arr = df["q"].values
    s_arr = df["s"].values
    bit_arr = df["bit"].values

    idx = defaultdict(dict)
    for i in range(len(df)):
        idx[q_arr[i]][s_arr[i]] = float(bit_arr[i])
    _ALN_INDEX_CACHE = dict(idx)
    logger.info(
        "Built in-memory alignment index in %.2fs for %d proteins (%d total edges).",
        time.time() - t0,
        len(_ALN_INDEX_CACHE),
        len(df),
    )

    # Save to pickle cache for next runs
    try:
        with open(pkl_cache, "wb") as f:
            pickle.dump(_ALN_INDEX_CACHE, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved alignment index cache to %s for fast future loading.", pkl_cache)
    except Exception as exc:
        logger.warning("Could not write alignment index cache to %s: %s", pkl_cache, exc)

    return _ALN_INDEX_CACHE


def find_induced_alignment_edges(
    accessions: Set[str],
    alignment_tsv: Path = DIAMOND_ALIGNMENT,
    timeout: int = 60,
) -> List[Tuple[str, str, float]]:
    """Extract mutual alignment edges among a set of proteins from SwissProt alignment table.

    Uses in-memory alignment index for instant, exact O(|accessions|^2) extraction,
    with fallback to ripgrep if index cannot be loaded.

    Args:
        accessions: Set of protein accessions in the subgraph.
        alignment_tsv: Path to diamond_swissprot_2024_01_alignment.tsv.
        timeout: Subprocess timeout in seconds.

    Returns:
        List of (src_accession, dst_accession, bitscore) tuples.
    """
    if not alignment_tsv.exists() or len(accessions) <= 1:
        return []

    # Fast in-memory lookup
    aln_idx = get_alignment_index(alignment_tsv)
    if aln_idx:
        edges = []
        node_list = list(accessions)
        for p1 in node_list:
            if p1 in aln_idx:
                p1_hits = aln_idx[p1]
                for p2 in node_list:
                    if p2 in p1_hits:
                        edges.append((p1, p2, p1_hits[p2]))
        logger.info(
            "Extracted %d induced alignment edges among %d proteins via in-memory index.",
            len(edges),
            len(accessions),
        )
        return edges

    acc_list = sorted(accessions)
    edges = []

    # Process in chunks of 1500 to stay well within ripgrep regex length limits
    chunk_size = 1500
    for i in range(0, len(acc_list), chunk_size):
        chunk = acc_list[i : i + chunk_size]
        pattern = "|".join(chunk)
        # Match where both query and target belong to the accession set
        cmd = ["rg", "-N", f"^({pattern})\\t({pattern})\\t", str(alignment_tsv)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            for line in res.stdout.splitlines():
                parts = line.split("\t")
                if len(parts) >= 12:
                    p1, p2, bitscore = parts[0], parts[1], float(parts[11])
                    edges.append((p1, p2, bitscore))
        except Exception as exc:
            logger.warning("Failed to extract induced alignment edges for chunk: %s", exc)

    logger.info(
        "Extracted %d induced alignment edges among %d proteins.",
        len(edges),
        len(accessions),
    )
    return edges


# ---------------------------------------------------------------------------
# Neighbor discovery
# ---------------------------------------------------------------------------

def identify_neighbors(
    protein_info: ProteinInfo,
    workspace: Path,
    graphs_dir: Path = PROTEIN_GRAPHS_DIR,
    max_neighbors: int = 500,
    diamond_bin: str = DIAMOND_EXECUTABLE,
    diamond_db: Path = DIAMOND_SWISSPROT_DB,
    alignment_tsv: Optional[Path] = DIAMOND_ALIGNMENT,
    return_pairwise: bool = False,
) -> Union[Dict[str, dict], Tuple[Dict[str, dict], List[Tuple[str, str, float]]]]:
    """Identify the full neighborhood of a protein (DIAMOND + STRING-DB).

    Args:
        protein_info: Resolved protein information.
        workspace: Directory for temporary files.
        graphs_dir: Directory containing prebuilt .pt graph files.
        max_neighbors: Maximum total neighbors to include.
        diamond_bin: Path to DIAMOND executable.
        diamond_db: Path to DIAMOND database.
        alignment_tsv: Optional path to global alignment table for induced edges.
        return_pairwise: If True, also return list of (acc_a, acc_b, score) STRING-DB edges.

    Returns:
        Dictionary mapping accession → neighbor metadata, or (neighbors, pairwise_string_edges) if return_pairwise is True.
    """
    neighbors: Dict[str, dict] = {}

    # 1. DIAMOND alignment
    diamond_tsv = run_diamond(
        protein_info, workspace,
        diamond_bin=diamond_bin, database=diamond_db,
    )
    self_bitscore, diamond_hits = parse_diamond_output(diamond_tsv, max_hits=max_neighbors)

    # Store self-hit for query protein
    if self_bitscore is not None:
        neighbors[protein_info.accession] = {
            "source": "diamond",
            "bitscore": self_bitscore,
            "combined_score": None,
            "has_pt": True,
            "is_self": True,
        }

    for acc, bitscore in diamond_hits:
        neighbors[acc] = {
            "source": "diamond",
            "bitscore": bitscore,
            "combined_score": None,
            "has_pt": False,
            "is_self": False,
        }

    # 2. STRING-DB lookup (with workspace caching)
    string_cache_file = workspace / f"{protein_info.accession}_stringdb.json"
    if string_cache_file.exists() and string_cache_file.stat().st_size > 0:
        try:
            with open(string_cache_file, "r") as f:
                cdata = json.load(f)
            stringdb_hits = cdata["partners"]
            pairwise_string_edges = cdata["pairwise_edges"]
        except Exception:
            stringdb_hits, pairwise_string_edges = lookup_stringdb_neighbors(
                protein_info, max_neighbors=max_neighbors
            )
    else:
        stringdb_hits, pairwise_string_edges = lookup_stringdb_neighbors(
            protein_info, max_neighbors=max_neighbors
        )
        try:
            with open(string_cache_file, "w") as f:
                json.dump({"partners": stringdb_hits, "pairwise_edges": pairwise_string_edges}, f)
        except Exception:
            pass

    for acc, score in stringdb_hits:
        if acc in neighbors:
            neighbors[acc]["source"] = "both"
            neighbors[acc]["combined_score"] = score
        else:
            neighbors[acc] = {
                "source": "stringdb",
                "bitscore": None,
                "combined_score": score,
                "has_pt": False,
                "is_self": False,
            }

    # Check which neighbors have existing .pt files
    available = 0
    for acc in list(neighbors.keys()):
        pt_path = graphs_dir / f"{acc}.pt"
        if pt_path.exists():
            neighbors[acc]["has_pt"] = True
            available += 1

    logger.info(
        "Neighborhood for %s: %d total neighbors (%d DIAMOND, %d STRING-DB), "
        "%d with existing .pt files.",
        protein_info.accession,
        len(neighbors),
        len(diamond_hits),
        len(stringdb_hits),
        available,
    )

    if return_pairwise:
        return neighbors, pairwise_string_edges
    return neighbors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for neighbor preparation."""
    parser = argparse.ArgumentParser(
        description="Identify and optionally prepare neighbor .pt graphs for inference."
    )
    parser.add_argument(
        "--proteins",
        nargs="+",
        required=True,
        help="UniProt accession(s) to discover neighbors for.",
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=PROTEIN_GRAPHS_DIR,
        help=f"Directory with prebuilt .pt graphs (default: {PROTEIN_GRAPHS_DIR}).",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=INFERENCE_OUTPUT_DIR / "neighbor_workspace",
        help="Workspace directory for temporary files.",
    )
    parser.add_argument(
        "--max-neighbors",
        type=int,
        default=500,
        help="Maximum neighbors per protein (default: 500).",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only report neighborhood statistics; do not build missing graphs.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    from src.utils.inference.resolve_ids import resolve_proteins

    protein_infos = resolve_proteins(args.proteins)

    for pinfo in protein_infos:
        neighbors = identify_neighbors(
            pinfo,
            workspace=args.workspace,
            graphs_dir=args.graphs_dir,
            max_neighbors=args.max_neighbors,
        )

        total = len(neighbors)
        available = sum(1 for n in neighbors.values() if n["has_pt"])
        missing = total - available
        diamond_only = sum(1 for n in neighbors.values() if n["source"] == "diamond")
        stringdb_only = sum(1 for n in neighbors.values() if n["source"] == "stringdb")
        both = sum(1 for n in neighbors.values() if n["source"] == "both")

        print(f"\n{'='*60}")
        print(f"  Neighborhood Report: {pinfo.accession} ({pinfo.entry_name})")
        print(f"{'='*60}")
        print(f"  Total neighbors:       {total}")
        print(f"  DIAMOND only:          {diamond_only}")
        print(f"  STRING-DB only:        {stringdb_only}")
        print(f"  Both:                  {both}")
        print(f"  With existing .pt:     {available}")
        print(f"  Missing .pt:           {missing}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
