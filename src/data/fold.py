import argparse
import math
from pathlib import Path
from typing import Iterable, List, Tuple
import time
import tqdm
import torch
import random
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import Protein as OFProtein, to_pdb
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

DATA_DIR = Path("./data/swissprot/2024_01")
ALPHAFOLD_DIR = DATA_DIR / "alphafold_pdb"
ESMFOLD_DIR = DATA_DIR / "esmfold_pdb"
MISSING_FASTA = DATA_DIR / "structure_missing.fasta"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ESMFold structures for sequences lacking Alphafold PDBs."
    )
    parser.add_argument("--fasta", required=True, type=Path, help="Input FASTA file.")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local checkpoint (local_files_only=True). Defaults to remote weights.",
    )
    return parser.parse_args()


def read_fasta(path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header = None
    pieces: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(pieces).upper()))
                header = line[1:].split()[0]
                pieces = []
            else:
                pieces.append(line)
        if header is not None:
            records.append((header, "".join(pieces).upper()))
    return records


def write_fasta(records: Iterable[Tuple[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for rid, seq in records:
            handle.write(f">{rid}\n{seq}\n")


def load_model(
    local: bool,
) -> Tuple[AutoTokenizer, EsmForProteinFolding, torch.device]:
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/esmfold_v1", local_files_only=local
    )
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1",
        low_cpu_mem_usage=True,
        local_files_only=local,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.esm = model.esm.half()
    torch.backends.cuda.matmul.allow_tf32 = True
    model.trunk.set_chunk_size(64)
    model.eval()
    print(f"Using device: {device}")
    return tokenizer, model, device


def convert_outputs_to_pdb(raw_outputs, seq_lengths: List[int]) -> List[str]:
    final_atom_positions = atom14_to_atom37(raw_outputs["positions"][-1], raw_outputs)
    outputs_np = {k: v.to("cpu").numpy() for k, v in raw_outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs_np["atom37_atom_exists"]
    pdbs: List[str] = []
    for i, length in enumerate(seq_lengths):
        protein = OFProtein(
            aatype=outputs_np["aatype"][i][:length],
            atom_positions=final_atom_positions[i][:length],
            atom_mask=final_atom_mask[i][:length],
            residue_index=outputs_np["residue_index"][i][:length] + 1,
            b_factors=outputs_np["plddt"][i][:length],
            chain_index=(
                outputs_np["chain_index"][i][:length]
                if "chain_index" in outputs_np
                else None
            ),
        )
        pdbs.append(to_pdb(protein))
    return pdbs


def batched_by_length(records: List[Tuple[str, str]], max_aa: int = 1000):
    """Yield batches where total amino acids < max_aa, or single large proteins."""
    current_batch = []
    current_length = 0

    for seq_id, seq in records:
        seq_len = len(seq)

        # If protein is larger than threshold, yield it alone
        if seq_len >= max_aa:
            # First yield any accumulated batch
            if current_batch:
                yield current_batch
                current_batch = []
                current_length = 0
            # Then yield the large protein alone
            yield [(seq_id, seq)]
            continue

        # If adding this protein exceeds threshold, yield current batch first
        if current_length + seq_len >= max_aa and current_batch:
            yield current_batch
            current_batch = []
            current_length = 0

        # Add protein to current batch
        current_batch.append((seq_id, seq))
        current_length += seq_len

    # Yield remaining batch
    if current_batch:
        yield current_batch


def generate_pdb_batch(
    batch: List[Tuple[str, str]], tokenizer, model, device, truncation=False
) -> Tuple[List[str], dict, List[int]]:
    """Generate model outputs for a batch. Returns (seq_ids, raw_outputs, seq_lengths)"""
    sequences = [seq for _, seq in batch]
    seq_ids = [seq_id for seq_id, _ in batch]
    seq_lengths = [len(seq) for seq in sequences]
    tokenized = tokenizer(
        sequences,
        return_tensors="pt",
        add_special_tokens=False,
        padding="max_length",
        max_length=1024,
        truncation=truncation,
    )
    model_inputs = {k: v.to(device) for k, v in tokenized.items()}
    with torch.no_grad():
        outputs = model(**model_inputs)
    return seq_ids, outputs, seq_lengths


def fold(batch, tokenizer, model, device, truncation):
    seq_ids, outputs, seq_lengths = generate_pdb_batch(
        batch, tokenizer, model, device, truncation=truncation
    )
    pdb_outputs = convert_outputs_to_pdb(outputs, seq_lengths)
    for seq_id, pdb_str in zip(seq_ids, pdb_outputs):
        out_file = ESMFOLD_DIR / f"ESMFold-{seq_id}.pdb"
        out_file.write_text(pdb_str, encoding="utf-8")
        print(f"Wrote ESMFold .pdb output to {out_file}")
    del outputs


def main() -> None:
    args = parse_args()

    sequences = read_fasta(args.fasta)
    if not sequences:
        print("No sequences found in FASTA.")
        return

    missing_records: List[Tuple[str, str]] = []
    for seq_id, seq in tqdm.tqdm(sequences, desc="Checking existing structures"):
        alphafold_path = ALPHAFOLD_DIR / f"AF-{seq_id}-F1-model_v6.pdb"
        if not alphafold_path.exists():
            esmfold_dir = ESMFOLD_DIR / f"ESMFold-{seq_id}.pdb"
            if not esmfold_dir.exists():
                missing_records.append((seq_id, seq))

    if not missing_records:
        if MISSING_FASTA.exists():
            MISSING_FASTA.unlink()
        print("All sequences already have structures.")
        return

    # # Sort batches by descending length.
    # missing_records.sort(key=lambda x: len(x[1]))
    select_records = random.sample(missing_records, min(len(missing_records), 400))
    # Remove selected records from missing_records to avoid duplication
    for record in select_records:
        missing_records.remove(record)

    MISSING_FASTA.parent.mkdir(parents=True, exist_ok=True)
    write_fasta(missing_records, MISSING_FASTA)
    print(f"Wrote {len(missing_records)} missing sequences to {MISSING_FASTA}")
    missing_records = select_records

    tokenizer, model, device = load_model(args.local)
    print(f"Loaded ESMFold model on {device}.")
    ESMFOLD_DIR.mkdir(parents=True, exist_ok=True)

    max_aa_per_batch = 100

    # Count total batches for progress bar
    total_batches = 0
    for _ in batched_by_length(missing_records, max_aa_per_batch):
        total_batches += 1

    for batch in tqdm.tqdm(
        batched_by_length(missing_records, max_aa_per_batch),
        total=total_batches,
        desc="Generating ESMFold structures",
    ):
        batch_aa = sum(len(seq) for _, seq in batch)
        print(f"Processing batch of {len(batch)} proteins, {batch_aa} amino acids.")

        try:
            fold(batch, tokenizer, model, device, truncation=False)
        except Exception as err:
            seq_ids_str = ", ".join(seq_id for seq_id, _ in batch)
            print(f"Failed batch [{seq_ids_str}, {batch_aa} aa]: {err}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                print("Retrying with truncation=True...")
                fold(batch, tokenizer, model, device, truncation=True)
            except Exception as err2:
                print(f"Failed again with truncation: {err2}. Skipping batch.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            continue
    print(f"ESMFold structures stored in {ESMFOLD_DIR}")


if __name__ == "__main__":
    main()
