"""SwissProt data download and preprocessing utilities.

This module provides functions to download and preprocess data required for
protein function prediction, including:
- AlphaFold protein structures from the SwissProt database
- InterPro domain annotations
- STRING database protein interactions
- Filtering data to match the SwissProt protein set

The preprocessing pipeline ensures all required data sources are downloaded,
extracted, and filtered to contain only proteins in the target dataset.
"""

import gzip
import logging
import shutil
import tarfile
import os
from pathlib import Path
from typing import Set
import pandas as pd

import requests
from Bio import SeqIO
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_file(url: str, destination: Path, chunk_size: int = 8192) -> None:
    """Download a file with progress bar."""
    logger.info(f"Downloading {url} to {destination}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    destination.parent.mkdir(parents=True, exist_ok=True)

    with open(destination, "wb") as f, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=destination.name,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    logger.info(f"Download complete: {destination}")


def get_swissprot_protein_ids(fasta_path: Path) -> Set[str]:
    """Extract all protein IDs from SwissProt FASTA file."""
    logger.info(f"Reading protein IDs from {fasta_path}")
    protein_ids = set()

    with open(fasta_path, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            protein_ids.add(record.id)

    logger.info(f"Found {len(protein_ids)} protein IDs in FASTA file")
    return protein_ids


def extract_alphafold_structures(
    tar_path: Path, output_dir: Path, protein_ids: Set[str]
) -> None:
    """Extract and decompress PDB files from AlphaFold tar archive."""
    logger.info(f"Extracting AlphaFold structures from {tar_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    skipped_count = 0

    with tarfile.open(tar_path, "r") as tar:
        members = tar.getmembers()

        for member in tqdm(members, desc="Extracting PDB files"):
            if not member.name.endswith(".pdb.gz"):
                continue

            # Extract protein ID from filename (e.g., AF-P12345-F1-model_v4.pdb.gz)
            filename = Path(member.name).name
            parts = filename.split("-")
            if len(parts) >= 2:
                protein_id = parts[1]  # Get the UniProt ID
            else:
                skipped_count += 1
                continue

            # Only process proteins in our SwissProt dataset
            if protein_id not in protein_ids:
                skipped_count += 1
                continue

            # Extract the .pdb.gz file
            gz_path = output_dir / filename
            tar.extract(member, path=output_dir)

            # Move file if extracted to subdirectory
            extracted_path = output_dir / member.name
            if extracted_path != gz_path:
                extracted_path.rename(gz_path)
                # Clean up any empty directories
                try:
                    extracted_path.parent.rmdir()
                except OSError:
                    pass

            # Decompress .pdb.gz to .pdb
            pdb_path = gz_path.with_suffix("")  # Remove .gz extension

            with gzip.open(gz_path, "rb") as f_in:
                with open(pdb_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Remove the .gz file
            gz_path.unlink()

            extracted_count += 1

    logger.info(f"Extracted {extracted_count} PDB files, skipped {skipped_count}")


def process_interpro_annotations(
    interpro_gz_path: Path,
    protein_ids: Set[str],
    output_path: Path,
) -> None:
    """Process InterPro annotations, keeping only SwissProt proteins."""
    logger.info(f"Processing InterPro annotations from {interpro_gz_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    matched_lines = 0
    total_lines = 0

    with gzip.open(interpro_gz_path, "rt") as f_in, open(output_path, "w") as f_out:
        # Write header
        f_out.write("ID\tIPR\tdesc\tdb\tstart\tend\n")

        for line in tqdm(f_in, desc="Processing InterPro annotations"):
            total_lines += 1

            # Skip empty lines
            line = line.strip()
            if not line:
                continue

            # Split by tab
            parts = line.split("\t")
            if len(parts) < 6:
                continue

            protein_id = parts[0]

            # Only keep lines for proteins in our SwissProt dataset
            if protein_id in protein_ids:
                f_out.write(line + "\n")
                matched_lines += 1

    logger.info(
        f"Processed {total_lines} lines, kept {matched_lines} matching SwissProt proteins"
    )


def get_stringdb(
    base_dir,
):
    """Download and filter STRING database protein interactions.

    Args:
        base_dir: Base directory path for data storage
    """
    stringdb_url = (
        "https://stringdb-downloads.org/download/stream/protein.links.detailed.v12.0.onlyAB.tsv.gz",
    )
    mapping_path = (base_dir / "idmapping_swissprot_stringdb.tsv",)
    output_path = (base_dir / "swissprot_stringdb.tsv",)

    # Download STRINGdb file
    gz_path = "protein.links.detailed.v12.0.onlyAB.tsv.gz"
    if not os.path.exists(gz_path):
        print("Downloading STRINGdb data...")
        with requests.get(stringdb_url, stream=True) as r:
            r.raise_for_status()
            with open(gz_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    else:
        print("STRINGdb data already downloaded.")

    # Load SwissProt-STRINGdb mapping
    print("Loading SwissProt-STRINGdb mapping...")
    mapping_df = pd.read_csv(mapping_path, sep="\t")
    stringdb_ids = set(mapping_df["To"].astype(str))

    # Prepare output
    print("Filtering STRINGdb data...")
    with gzip.open(gz_path, "rt") as fin, open(output_path, "w") as fout:
        header = fin.readline()
        fout.write(header)
        for line in fin:
            cols = line.rstrip("\n").split("\t")
            if cols[0] in stringdb_ids or cols[1] in stringdb_ids:
                fout.write(line)
    print(f"Filtered data saved to {output_path}")


def main():
    """Main function to download and process AlphaFold and InterPro data."""

    # Define paths
    base_dir = Path("./data/swissprot/2024_01")
    fasta_path = base_dir / "swissprot_2024_01.fasta"

    # AlphaFold paths
    alphafold_url = (
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_pdb_v6.tar"
    )
    alphafold_tar = base_dir / "swissprot_pdb_v6.tar"
    alphafold_output_dir = base_dir / "alphafold_pdb"

    # InterPro paths
    interpro_url = (
        "https://ftp.ebi.ac.uk/pub/databases/interpro/releases/106.0/protein2ipr.dat.gz"
    )
    interpro_gz = base_dir / "protein2ipr.dat.gz"
    interpro_output = base_dir / "swissprot_interpro_106_0.tsv"

    # Check if FASTA file exists
    if not fasta_path.exists():
        # Parse
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    # Get protein IDs from FASTA
    protein_ids = get_swissprot_protein_ids(fasta_path)

    # Download and process AlphaFold structures
    if not alphafold_tar.exists():
        download_file(alphafold_url, alphafold_tar)
    else:
        logger.info(f"AlphaFold tar file already exists: {alphafold_tar}")

    if not alphafold_output_dir.exists() or not any(alphafold_output_dir.glob("*.pdb")):
        extract_alphafold_structures(alphafold_tar, alphafold_output_dir, protein_ids)
    else:
        logger.info(f"AlphaFold PDB files already extracted in {alphafold_output_dir}")

    # Download and process InterPro annotations
    if not interpro_gz.exists():
        download_file(interpro_url, interpro_gz)
    else:
        logger.info(f"InterPro file already exists: {interpro_gz}")

    if not interpro_output.exists():
        process_interpro_annotations(interpro_gz, protein_ids, interpro_output)
    else:
        logger.info(f"Filtered InterPro annotations already exist: {interpro_output}")

    logger.info("All processing complete!")
    logger.info(f"AlphaFold structures: {alphafold_output_dir}")
    logger.info(f"InterPro annotations: {interpro_output}")


if __name__ == "__main__":
    main()
