"""Protein alignment using DIAMOND for sequence similarity search.

This script provides functionality to run DIAMOND (a fast sequence aligner) for
protein-protein alignment. It's used to find sequence similarities between proteins
which are then used as edges in the protein-protein graph.
"""

import subprocess

# Paths
base_dir = "/home/atoffano/PFP_layer/data/"
diamond_db = f"{base_dir}/swissprot_2024_01_proteins_set.dmnd"
input_protein = f"{base_dir}/tmp/input_protein.fasta"  # Your input protein FASTA file
output_file = f"{base_dir}/tmp/diamond_alignment_output.tsv"

# DIAMOND command setup
diamond_command = [
    "diamond",
    "blastp",
    "-q",
    input_protein,
    "-d",
    diamond_db,
    "-o",
    output_file,
    "-e",
    "0.001",
]

# Run DIAMOND alignment
try:
    subprocess.run(diamond_command, check=True)
    print(f"Diamond alignment completed successfully. Output saved to {output_file}")
except subprocess.CalledProcessError as e:
    print(f"Error during diamond alignment: {e}")
except FileNotFoundError:
    print(
        "DIAMOND executable not found. Please ensure DIAMOND is installed and in your PATH."
    )
