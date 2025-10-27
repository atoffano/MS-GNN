"""Missing PDB structure identification and handling.

This script identifies proteins from the SwissProt dataset that lack both
AlphaFold and ESMFold structure predictions. It generates a FASTA file
containing sequences that need structure prediction, which can then be
processed by the folding pipeline.
"""

import sys
from pathlib import Path

FASTA_PATH = Path(
    "/lustre/fsn1/projects/rech/dqy/uki62ne/tempdata/PFP_layer/data/swissprot/2024_01/swissprot_2024_01.fasta"
)
ALPHAFOLD_DIR = Path(
    "/lustre/fsn1/projects/rech/dqy/uki62ne/tempdata/PFP_layer/data/swissprot/2024_01/alphafold_pdb"
)
ESMFOLD_DIR = Path(
    "/lustre/fsn1/projects/rech/dqy/uki62ne/tempdata/PFP_layer/data/swissprot/2024_01/esmfold_pdb"
)
MISSING_FASTA = Path(
    "/lustre/fsn1/projects/rech/dqy/uki62ne/tempdata/PFP_layer/data/swissprot/2024_01/structure_missing.fasta"
)


def read_fasta(path: Path):
    records = []
    header = None
    seq_lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_lines)))
                header = line[1:].split()[0]
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            records.append((header, "".join(seq_lines)))
    return records


def write_fasta(records, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for header, seq in records:
            f.write(f">{header}\n{seq}\n")


def main():
    records = read_fasta(FASTA_PATH)
    missing_records = []
    missing_names = []
    for header, seq in records:
        af_path = ALPHAFOLD_DIR / f"AF-{header}-F1-model_v6.pdb"
        esm_path = ESMFOLD_DIR / f"ESMFold-{header}.pdb"
        if not (af_path.exists() or esm_path.exists()):
            missing_records.append((header, seq))
            missing_names.append(header)
            print(f"Missing structure for: {header}")
    print(f"Total missing structures: {len(missing_records)}")
    if missing_records:
        write_fasta(missing_records, MISSING_FASTA)
        print(f"Wrote missing sequences to {MISSING_FASTA}")


if __name__ == "__main__":
    main()