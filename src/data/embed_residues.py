"""ESM (Evolutionary Scale Modeling) embedding generation for protein residues.

This module generates per-residue embeddings using the ESM-1b language model.
The embeddings capture evolutionary and structural information at the amino acid
level and are used as node features in the protein graphs.

Features:
- Batch processing for efficient computation
- HDF5 storage for large-scale embedding datasets
- Support for local ESM model checkpoints
- Memory-efficient processing with configurable batch sizes
"""

import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
import torch
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from tqdm.auto import tqdm
import esm

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATA_ROOT = Path("./data/swissprot/2024_01")
FASTA_PATH = DATA_ROOT / "swissprot_2024_01.fasta"
EMBED_H5_PATH = DATA_ROOT / "swissprot_esm1b_per_aa_test.h5"

ESM_LAYER = 33
ESM_BATCH_SIZE = 8
DEFAULT_MAX_AA_PER_BATCH = 4000


def load_esm_model(
    local_checkpoint: Optional[Path] = None,
) -> Tuple[torch.nn.Module, esm.Alphabet]:
    """Load ESM-1b protein language model.
    
    Args:
        local_checkpoint: Optional path to local model checkpoint
        
    Returns:
        Tuple of (model, alphabet)
    """
    if local_checkpoint:
        checkpoint = Path(local_checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Local ESM checkpoint not found: {checkpoint}")
        logger.info("Loading ESM model from local checkpoint %s", checkpoint)
        return esm.pretrained.load_model_and_alphabet_local(checkpoint)

    model_name = "esm1b_t33_650M_UR50S"
    logger.info("Downloading ESM model %s", model_name)
    return esm.pretrained.esm1b_t33_650M_UR50S()


def batched_by_length(
    records: Iterable[SeqRecord], max_aa: int
) -> Iterable[List[SeqRecord]]:
    """Batch sequences by total amino acid count.
    
    Args:
        records: Iterable of sequence records
        max_aa: Maximum total amino acids per batch
        
    Yields:
        Batches of sequence records
    """
    batch: List[SeqRecord] = []
    total_len = 0

    for record in records:
        length = len(record.seq)
        if length >= max_aa:
            if batch:
                yield batch
                batch = []
                total_len = 0
            yield [record]
            continue

        if batch and total_len + length >= max_aa:
            yield batch
            batch = []
            total_len = 0

        batch.append(record)
        total_len += length

    if batch:
        yield batch


class ResidueEmbedder:
    """ESM-based protein residue embedder.
    
    Generates per-residue embeddings for protein sequences using ESM-1b
    language model, with support for long sequences via windowing.
    """
    
    def __init__(
        self,
        local_checkpoint: Optional[Path] = None,
        h5_path: Path = EMBED_H5_PATH,
        max_aa_per_batch: int = DEFAULT_MAX_AA_PER_BATCH,
        esm_batch_size: int = ESM_BATCH_SIZE,
    ):
        """Initialize ResidueEmbedder.
        
        Args:
            local_checkpoint: Optional path to local ESM model checkpoint
            h5_path: Path to output HDF5 file for embeddings
            max_aa_per_batch: Maximum total amino acids per batch
            esm_batch_size: Batch size for ESM model inference
        """
        self.h5_path = Path(h5_path)
        self.max_aa_per_batch = max(1, max_aa_per_batch)
        self.esm_batch_size = max(1, esm_batch_size)

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

    def embed_sequence(self, seq_id: str, sequence: str) -> torch.Tensor:
        """Generate embeddings for a single protein sequence.
        
        For long sequences, uses overlapping windows and averages embeddings.
        
        Args:
            seq_id: Sequence identifier
            sequence: Amino acid sequence string
            
        Returns:
            Tensor of per-residue embeddings, shape (length, embedding_dim)
        """
        length = len(sequence)
        if length == 0:
            raise ValueError(f"Sequence {seq_id} is empty")

        if not self.window_len or length <= self.window_len:
            batch = [(seq_id, sequence)]
            _, _, tokens = self.batch_converter(batch)
            tokens = tokens.to(self.device)
            with torch.no_grad():
                reps = self.model(
                    tokens, repr_layers=[ESM_LAYER], return_contacts=False
                )["representations"][ESM_LAYER]
            return reps[0, 1 : 1 + length].cpu().to(torch.float16).clone()

        starts = list(range(0, length, self.window_stride))
        if starts[-1] + self.window_len < length:
            starts.append(length - self.window_len)
        starts = sorted(set(starts))

        windows = [
            (
                start,
                min(start + self.window_len, length),
                sequence[start : min(start + self.window_len, length)],
            )
            for start in starts
        ]

        accumulator: Optional[np.ndarray] = None
        counts = np.zeros((length,), dtype=np.int32)

        for block_start in range(0, len(windows), self.esm_batch_size):
            batch_windows = windows[block_start : block_start + self.esm_batch_size]
            batch = [
                (f"{seq_id}_w{block_start + idx}", window_seq)
                for idx, (_, _, window_seq) in enumerate(batch_windows)
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

            for local_idx, (start, end, _) in enumerate(batch_windows):
                window_len = end - start
                sub_emb = reps[local_idx, 1 : 1 + window_len]
                if accumulator is None:
                    accumulator = np.zeros(
                        (length, sub_emb.shape[-1]), dtype=np.float64
                    )
                accumulator[start:end] += sub_emb
                counts[start:end] += 1

        counts[counts == 0] = 1
        averaged = (accumulator / counts[:, None]).astype(np.float32)
        return torch.from_numpy(averaged).clone()

    def embed_batch(self, records: List[SeqRecord]) -> Dict[str, torch.Tensor]:
        """Generate embeddings for a batch of sequences.
        
        Args:
            records: List of BioPython SeqRecord objects
            
        Returns:
            Dictionary mapping sequence IDs to embedding tensors
        """
        results: Dict[str, torch.Tensor] = {}
        long_sequences: List[Tuple[str, str]] = []
        short_sequences: List[Tuple[str, str]] = []

        for record in records:
            seq_id = record.id
            sequence = str(record.seq)
            if not sequence:
                raise ValueError(f"Sequence {seq_id} is empty")
            if self.window_len and len(sequence) > self.window_len:
                long_sequences.append((seq_id, sequence))
            else:
                short_sequences.append((seq_id, sequence))

        for seq_id, sequence in long_sequences:
            results[seq_id] = self.embed_sequence(seq_id, sequence)

        for start in range(0, len(short_sequences), self.esm_batch_size):
            chunk = short_sequences[start : start + self.esm_batch_size]
            batch = [(seq_id, sequence) for seq_id, sequence in chunk]
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
            for (seq_id, sequence), rep in zip(chunk, reps):
                length = len(sequence)
                results[seq_id] = rep[1 : 1 + length].to(torch.float16).clone()

        return results

    def process_fasta(
        self,
        fasta_path: Path,
        overwrite: bool = False,
    ) -> Tuple[int, int, int]:
        """Process a FASTA file and save embeddings to HDF5.
        
        Args:
            fasta_path: Path to input FASTA file
            overwrite: If True, recompute embeddings even if they exist
            
        Returns:
            Tuple of (total_sequences, written, skipped)
        """
        fasta_path = Path(fasta_path)
        if not fasta_path.exists():
            raise FileNotFoundError(f"FASTA not found: {fasta_path}")

        records = list(SeqIO.parse(str(fasta_path), "fasta"))
        if not records:
            logger.warning("No sequences found in FASTA %s", fasta_path)
            return 0, 0, 0

        self.h5_path.parent.mkdir(parents=True, exist_ok=True)

        written = 0
        skipped = 0
        failed = 0

        with h5py.File(self.h5_path, "a") as h5f, tqdm(
            total=len(records), desc="Embedding sequences", unit="protein"
        ) as progress:
            for batch in batched_by_length(records, self.max_aa_per_batch):
                todo: List[SeqRecord] = []
                for record in batch:
                    seq_id = record.id
                    if not overwrite and seq_id in h5f:
                        skipped += 1
                        progress.update(1)
                        continue
                    todo.append(record)

                if not todo:
                    continue

                try:
                    embeddings = self.embed_batch(todo)
                except Exception as exc:
                    failed += len(todo)
                    logger.error(
                        "Embedding failure for batch starting at %s: %s",
                        todo[0].id,
                        exc,
                    )
                    progress.update(len(todo))
                    continue

                for record in todo:
                    seq_id = record.id
                    sequence = str(record.seq)
                    embedding = embeddings.get(seq_id)
                    if embedding is None:
                        failed += 1
                        logger.error("Missing embeddings for %s", seq_id)
                        progress.update(1)
                        continue

                    emb_np = embedding.cpu().numpy()
                    if seq_id in h5f:
                        del h5f[seq_id]
                    group = h5f.create_group(seq_id)
                    group.create_dataset("embeddings", data=emb_np, compression="gzip")
                    group.attrs["sequence_length"] = len(sequence)
                    group.attrs["sequence"] = sequence
                    group.attrs["layer"] = ESM_LAYER

                    written += 1
                    progress.update(1)

        logger.info(
            "Finished embeddings: written=%d skipped=%d failed=%d",
            written,
            skipped,
            failed,
        )
        return written, skipped, failed


def main() -> None:
    """Main entry point for residue embedding generation.
    
    Processes a FASTA file and generates ESM embeddings for all sequences,
    storing results in an HDF5 file.
    """
    parser = argparse.ArgumentParser(
        description="Embed UniProt sequences into per-residue ESM1b representations."
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        default=FASTA_PATH,
        help="Input FASTA file (default: SwissProt 2024_01).",
    )
    parser.add_argument(
        "--local",
        type=Path,
        default=None,
        help="Optional path to a local ESM checkpoint.",
    )
    parser.add_argument(
        "--max-aa-per-batch",
        type=int,
        default=DEFAULT_MAX_AA_PER_BATCH,
        help="Maximum total amino acids per embedding batch.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute embeddings even if they already exist in the H5 file.",
    )
    args = parser.parse_args()

    embedder = ResidueEmbedder(
        local_checkpoint=args.local,
        max_aa_per_batch=args.max_aa_per_batch,
    )
    written, skipped, failed = embedder.process_fasta(
        args.fasta, overwrite=args.overwrite
    )
    logger.info(
        "Embedding complete - written=%d skipped=%d failed=%d",
        written,
        skipped,
        failed,
    )


if __name__ == "__main__":
    main()

# Usage example:
# python src/data/embed_residues.py --fasta data/swissprot/2024_01/swissprot_2024_01.fasta --h5 data/swissprot/2024_01/swissprot_esm1b_per_aa.h5 --local /path/to/esm1b_checkpoint.pt --max-aa-per-batch 4000 --overwrite
