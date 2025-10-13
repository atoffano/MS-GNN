import torch
import pickle
import pandas as pd
import numpy as np
import esm
from torch_geometric.data import HeteroData
from pathlib import Path
import logging
from tqdm.auto import tqdm
from Bio import SeqIO
import concurrent.futures

logger = logging.getLogger(__name__)


def _build_protein_graph(
    protein_id, embeddings, sequence, interpro_dict, go_annotations, interpro_vocab_size
):
    seq_len = len(sequence)
    if embeddings.shape[0] != seq_len:
        logger.warning(
            f"Embedding length mismatch for {protein_id}: {embeddings.shape[0]} vs {seq_len}"
        )
        return None
    data = HeteroData()
    data["aa"].x = embeddings
    data["aa"].num_nodes = seq_len
    data["protein"].num_nodes = 1
    if protein_id in interpro_dict:
        data["protein"].interpro = interpro_dict[protein_id].unsqueeze(0)
    else:
        data["protein"].interpro = torch.zeros(
            1, interpro_vocab_size, dtype=torch.float32
        )
    for subontology in ["BPO", "CCO", "MFO"]:
        data["protein"][f"go_terms_{subontology}"] = {}
        experimental_terms = []
        curated_terms = []
        if protein_id in go_annotations and subontology in go_annotations[protein_id]:
            experimental_terms = go_annotations[protein_id][subontology].get(
                "experimental", []
            )
            curated_terms = go_annotations[protein_id][subontology].get("curated", [])
        data["protein"][f"go_terms_{subontology}"]["experimental"] = experimental_terms
        data["protein"][f"go_terms_{subontology}"]["curated"] = curated_terms
    aa_indices = torch.arange(seq_len)
    protein_indices = torch.zeros(seq_len, dtype=torch.long)
    data["aa", "belongs_to", "protein"].edge_index = torch.stack(
        [aa_indices, protein_indices]
    )
    data["protein"].protein_id = protein_id
    data["protein"].sequence = sequence
    data["protein"].sequence_length = seq_len
    return data


class ProteinGraphPreprocessor:
    """Precomputes individual protein graphs from embeddings and annotations."""

    def __init__(self):
        self.output_dir = Path("./data/swissprot/2024_01/protein_graphs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.fasta_path = Path("./data/swissprot/2024_01/swissprot_2024_01.fasta")
        self.esm_checkpoint = Path("../torch_cache/esm1b_t33_650M_UR50S.pt")
        # self.esm_checkpoint = Path(
        #     "/home/atoffano/.cache/torch/hub/checkpoints/esm1b_t33_650M_UR50S.pt"
        # )
        self.esm_layer = 33
        self.batch_size = 8

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model, self.alphabet = self._load_esm_model()
        self.model = self.model.to(self.device).eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.max_model_tokens = getattr(
            getattr(self.model, "args", None), "max_positions", None
        )
        self.window_seq_len = (
            (self.max_model_tokens - 2) if self.max_model_tokens else None
        )
        self.window_stride = (
            max(1, self.window_seq_len // 2) if self.window_seq_len else None
        )

        self.interpro_dict, self.interpro_vocab_size = self._load_interpro_annotations()
        self.go_annotations = self._load_all_go_annotations()
        self.go_vocab_sizes = self._load_go_vocab_from_obo()

        logger.info(f"Loaded {len(self.interpro_dict)} InterPro annotations")
        logger.info(f"Loaded GO annotations for {len(self.go_annotations)} proteins")
        logger.info(f"GO vocab sizes: {self.go_vocab_sizes}")
        logger.info(f"Using device {self.device}")
        logger.info(
            f"ESM model window length: {self.window_seq_len if self.window_seq_len else 'unbounded'}"
        )

    def _load_esm_model(self):
        """Try downloading the model first; fall back to local weights after 15 s."""
        model_name = "esm1b_t33_650M_UR50S"

        def load_online():
            logger.info(
                f"Attempting to download ESM model '{model_name}' from remote source"
            )
            return esm.pretrained.esm1b_t33_650M_UR50S()

        if self.esm_checkpoint.exists():
            local_loader = lambda: esm.pretrained.load_model_and_alphabet_local(
                self.esm_checkpoint
            )
            logger.info(f"Loading ESM model from checkpoint: {self.esm_checkpoint}")
            return local_loader()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(load_online)
            try:
                model, alphabet = future.result(timeout=15)
                logger.info("Loaded ESM model from online source")
                return model, alphabet
            except concurrent.futures.TimeoutError:
                logger.warning("Online model load timed out after 15 s")
            except Exception as exc:
                logger.warning(f"Online model load failed ({exc})")

        raise RuntimeError(
            "Failed to load ESM model online within timeout and no local checkpoint available."
            "Please provide a valid local checkpoint path that contains esm1b_t33_650M_UR50S-contact-regression.pt and esm1b_t33_650M_UR50S.pt."
        )

    def _load_interpro_annotations(self):
        """Load InterPro annotations."""
        interpro_path = "./data/swissprot/2024_01/swissprot_interpro_106_0.tsv"
        df = pd.read_csv(interpro_path, sep="\t")
        ipr_ids = sorted(df["IPR"].unique())
        ipr_to_idx = {ipr: i for i, ipr in enumerate(ipr_ids)}

        grouped = df.groupby("ID")["IPR"].apply(list)
        interpro_dict = {}
        for pid, ipr_list in grouped.items():
            vec = torch.zeros(len(ipr_ids), dtype=torch.float32)
            for ipr in ipr_list:
                vec[ipr_to_idx[ipr]] = 1.0
            interpro_dict[pid] = vec

        with open(self.output_dir / "interpro_vocab.pkl", "wb") as f:
            pickle.dump({"ipr_to_idx": ipr_to_idx, "vocab_size": len(ipr_ids)}, f)

        return interpro_dict, len(ipr_ids)

    def _load_go_vocab_from_obo(self):
        """Load GO vocabulary from go.obo file and determine sizes per subontology."""
        obo_path = "./data/go.obo"

        go_terms_by_namespace = {
            "BPO": set(),
            "CCO": set(),
            "MFO": set(),
        }

        namespace_mapping = {
            "biological_process": "BPO",
            "cellular_component": "CCO",
            "molecular_function": "MFO",
        }

        current_term = None
        current_namespace = None

        with open(obo_path, "r") as f:
            for line in f:
                line = line.strip()

                if line == "[Term]":
                    current_term = None
                    current_namespace = None
                elif line.startswith("id: GO:"):
                    current_term = line.split("id: ")[1]
                elif line.startswith("namespace: "):
                    namespace = line.split("namespace: ")[1]
                    current_namespace = namespace_mapping.get(namespace)
                elif line == "" and current_term and current_namespace:
                    go_terms_by_namespace[current_namespace].add(current_term)

        go_vocab_info = {}
        for onto in ["BPO", "CCO", "MFO"]:
            terms = sorted(go_terms_by_namespace[onto])
            go_to_idx = {go: i for i, go in enumerate(terms)}
            go_vocab_info[onto] = {
                "terms": terms,
                "go_to_idx": go_to_idx,
                "vocab_size": len(terms),
            }

        with open(self.output_dir / "go_vocab.pkl", "wb") as f:
            pickle.dump(go_vocab_info, f)

        return {onto: info["vocab_size"] for onto, info in go_vocab_info.items()}

    def _load_all_go_annotations(self):
        """Load all GO annotations for each subontology (both experimental and curated)."""
        go_annotations = {}

        for subontology in ["BPO", "CCO", "MFO"]:
            exp_tsv_path = f"./data/swissprot/2024_01/swissprot_2024_01_{subontology}_exp_annotations.tsv"

            if Path(exp_tsv_path).exists():
                exp_df = pd.read_csv(exp_tsv_path, sep="\t")

                for _, row in exp_df.iterrows():
                    pid = row["EntryID"]
                    terms = [t.strip() for t in row["term"].split(";") if t.strip()]

                    if pid not in go_annotations:
                        go_annotations[pid] = {}
                    if subontology not in go_annotations[pid]:
                        go_annotations[pid][subontology] = {}

                    go_annotations[pid][subontology]["experimental"] = terms
            else:
                logger.warning(
                    f"Experimental GO annotation file not found: {exp_tsv_path}"
                )

            curated_tsv_path = f"./data/swissprot/2024_01/swissprot_2024_01_{subontology}_annotations.tsv"

            if Path(curated_tsv_path).exists():
                curated_df = pd.read_csv(curated_tsv_path, sep="\t")

                for _, row in curated_df.iterrows():
                    pid = row["EntryID"]
                    terms = [t.strip() for t in row["term"].split(";") if t.strip()]

                    if pid not in go_annotations:
                        go_annotations[pid] = {}
                    if subontology not in go_annotations[pid]:
                        go_annotations[pid][subontology] = {}

                    go_annotations[pid][subontology]["curated"] = terms
            else:
                logger.warning(
                    f"Curated GO annotation file not found: {curated_tsv_path}"
                )

        return go_annotations

    def _embed_sequence(self, seq_id, seq):
        L = len(seq)
        if L == 0:
            raise ValueError(f"Sequence {seq_id} is empty")

        if not self.window_seq_len or L <= self.window_seq_len:
            batch = [(seq_id, seq)]
            _, _, tokens = self.batch_converter(batch)
            tokens = tokens.to(self.device)

            with torch.no_grad():
                results = self.model(
                    tokens, repr_layers=[self.esm_layer], return_contacts=False
                )
                reps = results["representations"][self.esm_layer]

            emb = reps[0, 1 : 1 + L, :].detach().cpu().to(torch.float32)
            return emb

        starts = list(range(0, L, self.window_stride))
        if starts[-1] + self.window_seq_len < L:
            starts.append(L - self.window_seq_len)
        starts = sorted(set(s for s in starts if s >= 0))

        windows = [
            (
                s,
                min(s + self.window_seq_len, L),
                seq[s : min(s + self.window_seq_len, L)],
            )
            for s in starts
        ]
        acc = None
        counts = np.zeros((L,), dtype=np.int32)

        for i in range(0, len(windows), self.batch_size):
            batch_windows = windows[i : i + self.batch_size]
            batch = [
                (f"{seq_id}_w{i+j}", window_seq)
                for j, (_, _, window_seq) in enumerate(batch_windows)
            ]

            _, _, tokens = self.batch_converter(batch)
            tokens = tokens.to(self.device)

            with torch.no_grad():
                results = self.model(
                    tokens, repr_layers=[self.esm_layer], return_contacts=False
                )
                reps = results["representations"][self.esm_layer].detach().cpu().numpy()

            for j, (start, end, _) in enumerate(batch_windows):
                sub_len = end - start
                sub_emb = reps[j, 1 : 1 + sub_len, :]

                if acc is None:
                    acc = np.zeros((L, sub_emb.shape[-1]), dtype=np.float64)

                acc[start:end, :] += sub_emb
                counts[start:end] += 1

        counts = counts.astype(np.float32)
        counts[counts == 0] = 1.0
        emb = (acc / counts[:, None]).astype(np.float32)
        return torch.from_numpy(emb)

    def process_fasta(self):
        if not self.fasta_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {self.fasta_path}")

        successful = 0
        failed = 0

        logger.info(f"Processing FASTA: {self.fasta_path}")

        with open(self.fasta_path, "r") as handle:
            records = list(SeqIO.parse(handle, "fasta"))

        for record in tqdm(records, desc="Embedding proteins", unit="protein"):
            protein_id = record.id
            sequence = str(record.seq)

            try:
                embeddings = self._embed_sequence(protein_id, sequence)
                protein_graph = _build_protein_graph(
                    protein_id,
                    embeddings,
                    sequence,
                    self.interpro_dict,
                    self.go_annotations,
                    self.interpro_vocab_size,
                )
                if protein_graph is None:
                    failed += 1
                    continue

                output_path = self.output_dir / f"{protein_id}.pt"
                torch.save(protein_graph, output_path)
                successful += 1
            except Exception as exc:
                failed += 1
                logger.error(f"Failed to process {protein_id}: {exc}")

        logger.info(f"Successfully processed {successful} proteins, failed: {failed}")
        return successful, failed


def main():
    """Main preprocessing function."""

    logger.info("Processing both experimental and curated annotations...")
    preprocessor = ProteinGraphPreprocessor()
    successful, failed = preprocessor.process_fasta()

    logger.info(f"Processing complete - Successful: {successful}, Failed: {failed}")

    print("Preprocessing complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
