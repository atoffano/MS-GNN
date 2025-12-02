"""Download missing PDB structures for proteins before running attribution on compute nodes."""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataloading import SwissProtDataset
from src.utils.api import download_alphafold, download_pdb
from torch_geometric.loader import NeighborLoader
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_dataset_from_config(model_path: str):
    """Load dataset from model config."""
    config_path = os.path.join(model_path, "cfg.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Loading dataset...")
    dataset = SwissProtDataset(config)
    return dataset, config


def resolve_protein_names(dataset, protein_names: list[str]) -> list[str]:
    """Resolve protein names to UniProt IDs."""
    proteins = []
    for protein in protein_names:
        if "_" in protein:
            # Entry ID format
            if dataset.uses_entryid:
                proteins.append(protein)
            else:
                # Convert to UniProt ID
                if protein in dataset.pid_mapping:
                    proteins.append(dataset.pid_mapping[protein])
                else:
                    logger.warning(f"Entry ID {protein} not found in reverse mapping")
        else:
            # UniProt ID format
            if not dataset.uses_entryid:
                proteins.append(protein)
            else:
                # Convert to Entry ID
                if protein in dataset.rev_pid_mapping:
                    proteins.append(dataset.rev_pid_mapping[protein])
                else:
                    logger.warning(f"UniProt ID {protein} not found in mapping")

    return proteins


def get_neighborhood_proteins(config, dataset, protein_names: list[str]) -> set[str]:
    """Get all proteins in the neighborhood of specified proteins."""
    proteins = resolve_protein_names(dataset, protein_names)

    # Get protein indices
    protein_ids = []
    for protein in proteins:
        if protein in dataset.protein_to_idx:
            protein_ids.append(dataset.protein_to_idx[protein])
        else:
            logger.warning(f"Protein {protein} not found in dataset")

    if not protein_ids:
        return set()

    # Create mask for target proteins
    mask = torch.zeros(len(dataset.proteins), dtype=torch.bool)
    mask[protein_ids] = True

    # Create neighbor loader to get all proteins in subgraph
    num_neighbors = {}
    for edge_type_str, num_samples in config["model"]["sampled_edges"].items():
        edge_type_tuple = tuple(edge_type_str.split("__"))
        num_neighbors[edge_type_tuple] = [num_samples]

    loader = NeighborLoader(
        dataset.data,
        num_neighbors=num_neighbors,
        batch_size=len(protein_ids),
        input_nodes=("protein", mask),
        shuffle=False,
        num_workers=0,
    )

    # Get batch to extract all proteins
    batch = next(iter(loader))
    protein_global_ids = batch["protein"].n_id.detach().cpu().tolist()

    # Convert to UniProt IDs
    uniprot_ids = set()
    for global_id in protein_global_ids:
        protein_id = dataset.idx_to_protein[global_id]
        # Convert to UniProt ID if using entry IDs
        if dataset.uses_entryid:
            uniprot_id = dataset.rev_pid_mapping.get(protein_id, protein_id)
        else:
            uniprot_id = protein_id
        uniprot_ids.add(uniprot_id)

    return uniprot_ids


def check_local_caches(uniprot_id: str, data_base_dir: str) -> str | None:
    """Check if structure exists in local cache directories."""
    cache_dirs = ["alphafold_pdb", "esmfold_pdb", "tmp_pdb"]

    for cache_name in cache_dirs:
        cache_path = os.path.join(data_base_dir, cache_name, f"{uniprot_id}.pdb")
        if os.path.exists(cache_path):
            return cache_path

    return None


def download_structure(uniprot_id: str, output_path: str) -> bool:
    """Download structure from RCSB PDB or AlphaFold."""
    logger.info(f"Downloading structure for {uniprot_id}...")

    # Try AlphaFold
    if download_alphafold(uniprot_id, output_path):
        logger.info(f"✓ Downloaded from AlphaFold: {uniprot_id}")
        return True

    # Try RCSB PDB
    if download_pdb(uniprot_id, output_path):
        logger.info(f"✓ Downloaded from RCSB PDB: {uniprot_id}")
        return True

    logger.error(f"✗ Failed to download structure for {uniprot_id}")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Download missing PDB structures for attribution analysis"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model directory containing cfg.yaml",
    )
    parser.add_argument(
        "--proteins",
        nargs="+",
        required=True,
        help="Protein names (UniProt IDs or Entry IDs)",
    )

    args = parser.parse_args()

    # Load dataset
    dataset, config = load_dataset_from_config(args.model_path)

    # Determine base data directory
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_dir = os.path.join(
        current_file_dir, "..", "..", "data", "swissprot", "2024_01"
    )
    base_data_dir = os.path.normpath(base_data_dir)

    output_dir = args.model_path

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Get all proteins in neighborhood
    logger.info(f"Analyzing neighborhood for {len(args.proteins)} protein(s)...")
    all_proteins = get_neighborhood_proteins(config, dataset, args.proteins)
    logger.info(f"Found {len(all_proteins)} total proteins in neighborhood")

    # Check which structures are missing
    missing_proteins = []
    cached_proteins = []

    for uniprot_id in all_proteins:
        cache_path = check_local_caches(uniprot_id, base_data_dir)
        if cache_path:
            cached_proteins.append(uniprot_id)
            logger.debug(f"Found cached: {uniprot_id} at {cache_path}")
        else:
            missing_proteins.append(uniprot_id)

    logger.info(f"Found {len(cached_proteins)} structures in local cache")
    logger.info(f"Missing {len(missing_proteins)} structures")

    if not missing_proteins:
        logger.info("All required structures are already cached!")
        return

    # Download missing structures
    logger.info(f"\nDownloading {len(missing_proteins)} missing structures...")
    success_count = 0
    failed = []

    if dataset.uses_entryid:
        missing_proteins = [
            dataset.pid_mapping.get(pid, pid) for pid in missing_proteins
        ]

    for i, uniprot_id in enumerate(missing_proteins, 1):
        output_path = os.path.join(output_dir, f"{uniprot_id}.pdb")
        logger.info(f"[{i}/{len(missing_proteins)}] {uniprot_id}")

        if download_structure(uniprot_id, output_path):
            success_count += 1
        else:
            failed.append(uniprot_id)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total proteins in neighborhood: {len(all_proteins)}")
    logger.info(f"Already cached: {len(cached_proteins)}")
    logger.info(f"Downloaded: {success_count}/{len(missing_proteins)}")

    if failed:
        logger.warning(f"\nFailed to download {len(failed)} structures:")
        for uniprot_id in failed:
            logger.warning(f"  - {uniprot_id}")
    else:
        logger.info(f"Structures saved in: {os.path.abspath(output_dir)}")
        logger.info("\n✓ All required structures are now available!")


if __name__ == "__main__":
    main()
