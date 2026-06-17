"""Shared plotting context and output helpers."""

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

from src.utils.api import download_alphafold, download_alphafold_cif

logger = logging.getLogger(__name__)


@dataclass
class ProteinPlotContext:
    """Context for protein visualization."""

    seed_label: str
    seed_global: int
    seed_dir: str
    neighbor_dir: str
    protein_ids: List[int]
    labels: Dict[int, str]


def build_plot_context(base_path: str, dataset, batch) -> ProteinPlotContext:
    """Build plotting context from batch."""
    protein_ids = batch["protein"].n_id.detach().cpu().tolist()
    seed_global = int(protein_ids[0])
    labels = {
        idx: dataset.idx_to_protein.get(global_id, str(global_id))
        for idx, global_id in enumerate(protein_ids)
    }
    seed_label = labels[0]
    seed_dir = os.path.join(base_path, "explanations", seed_label)
    neighbor_dir = os.path.join(seed_dir, "neighbors")
    os.makedirs(seed_dir, exist_ok=True)
    os.makedirs(neighbor_dir, exist_ok=True)

    return ProteinPlotContext(
        seed_label=seed_label,
        seed_global=seed_global,
        seed_dir=seed_dir,
        neighbor_dir=neighbor_dir,
        protein_ids=protein_ids,
        labels=labels,
    )


def save_plot(
    context: ProteinPlotContext,
    filename_suffix: str,
    local_idx: int = 0,
    go_term: Optional[str] = None,
):
    """Save current plot to appropriate directory."""
    is_seed = context.protein_ids[local_idx] == context.seed_global
    base_dir = context.seed_dir if is_seed else context.neighbor_dir
    prefix = (
        context.seed_label
        if is_seed
        else f"{context.seed_label}_{context.labels[local_idx]}"
    )

    if go_term:
        go_subdir = go_term.replace(":", "_")
        output_dir = os.path.join(base_dir, "per-term", go_subdir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = base_dir

    filename = os.path.join(output_dir, f"{prefix}_{filename_suffix}.png")
    plt.savefig(filename)
    plt.close()


def ensure_structure(
    uniprot_id: str, out_dir: str, global_dir: Optional[str] = None
) -> str:
    """Get structure for UniProt ID, checking caches before downloading.

    Priority: global_dir .cif -> global_dir .pdb -> out_dir .cif -> out_dir .pdb
              -> AlphaFold CIF download -> AlphaFold PDB download
    Returns the path to the structure file (.cif preferred over .pdb).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Check local directories first
    search_dirs = [d for d in [global_dir, out_dir] if d]
    for search_dir in search_dirs:
        for ext in (".cif", ".pdb"):
            path = os.path.join(search_dir, f"{uniprot_id}{ext}")
            if os.path.exists(path):
                logger.info(f"Found {ext.upper()} structure for {uniprot_id} in {search_dir}")
                return path

    # Download to global_dir if provided, otherwise out_dir
    download_dir = global_dir if global_dir else out_dir
    os.makedirs(download_dir, exist_ok=True)
    cif_path = os.path.join(download_dir, f"{uniprot_id}.cif")
    pdb_path = os.path.join(download_dir, f"{uniprot_id}.pdb")

    if download_alphafold_cif(uniprot_id, cif_path):
        logger.info(f"Downloaded CIF structure for {uniprot_id} from AlphaFold")
        return cif_path

    if download_alphafold(uniprot_id, pdb_path):
        logger.info(f"Downloaded PDB structure for {uniprot_id} from AlphaFold")
        return pdb_path

    raise FileNotFoundError(f"No structure available for {uniprot_id}")
