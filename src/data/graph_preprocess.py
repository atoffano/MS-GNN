import torch
import h5py
import pickle
import pandas as pd
from torch_geometric.data import HeteroData
from pathlib import Path
import logging
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count

logger = logging.getLogger(__name__)

_WORKER_STATE = {}


def _init_worker(
    interpro_dict, go_annotations, interpro_vocab_size, output_dir, h5_path
):
    global _WORKER_STATE
    _WORKER_STATE = {
        "interpro_dict": interpro_dict,
        "go_annotations": go_annotations,
        "interpro_vocab_size": interpro_vocab_size,
        "output_dir": Path(output_dir),
        "h5_path": h5_path,
    }


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
    data["aa"].x = torch.tensor(embeddings, dtype=torch.float32)
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


def _process_protein(protein_id):
    try:
        with h5py.File(_WORKER_STATE["h5_path"], "r") as h5f:
            group = h5f[protein_id]
            embeddings = group["embeddings"][:]
            sequence = group.attrs["sequence"]
        protein_graph = _build_protein_graph(
            protein_id,
            embeddings,
            sequence,
            _WORKER_STATE["interpro_dict"],
            _WORKER_STATE["go_annotations"],
            _WORKER_STATE["interpro_vocab_size"],
        )
        if protein_graph is None:
            return False, protein_id, None, None
        output_path = _WORKER_STATE["output_dir"] / f"{protein_id}.pt"
        torch.save(protein_graph, output_path)
        return True, protein_id, None, embeddings.shape[1]
    except Exception as exc:
        logger.error(f"Failed to process {protein_id}: {exc}")
        return False, protein_id, exc, None


class ProteinGraphPreprocessor:
    """Precomputes individual protein graphs from embeddings and annotations."""

    def __init__(self):
        self.output_dir = Path("./data/swissprot/2024_01/protein_graphs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load annotations
        self.interpro_dict, self.interpro_vocab_size = self._load_interpro_annotations()
        self.go_annotations = self._load_all_go_annotations()
        self.go_vocab_sizes = self._load_go_vocab_from_obo()

        logger.info(f"Loaded {len(self.interpro_dict)} InterPro annotations")
        logger.info(f"Loaded GO annotations for {len(self.go_annotations)} proteins")
        logger.info(f"GO vocab sizes: {self.go_vocab_sizes}")

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

        # Save vocabulary for later use
        with open(self.output_dir / "interpro_vocab.pkl", "wb") as f:
            pickle.dump({"ipr_to_idx": ipr_to_idx, "vocab_size": len(ipr_ids)}, f)

        return interpro_dict, len(ipr_ids)

    def _load_go_vocab_from_obo(self):
        """Load GO vocabulary from go.obo file and determine sizes per subontology."""
        obo_path = "./data/go.obo"

        # Parse GO terms by namespace (subontology)
        go_terms_by_namespace = {
            "BPO": set(),  # biological_process
            "CCO": set(),  # cellular_component
            "MFO": set(),  # molecular_function
        }

        # Mapping from obo namespace to our naming convention
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
                # elif line.startswith("is_obsolete: true"):
                #     # Skip obsolete terms
                #     current_term = None
                #     current_namespace = None
                elif line == "" and current_term and current_namespace:
                    # End of term entry
                    go_terms_by_namespace[current_namespace].add(current_term)

        # Convert to sorted lists and create mappings
        go_vocab_info = {}
        for onto in ["BPO", "CCO", "MFO"]:
            terms = sorted(go_terms_by_namespace[onto])
            go_to_idx = {go: i for i, go in enumerate(terms)}
            go_vocab_info[onto] = {
                "terms": terms,
                "go_to_idx": go_to_idx,
                "vocab_size": len(terms),
            }

        # Save GO vocabulary
        with open(self.output_dir / "go_vocab.pkl", "wb") as f:
            pickle.dump(go_vocab_info, f)

        return {onto: info["vocab_size"] for onto, info in go_vocab_info.items()}

    def _load_all_go_annotations(self):
        """Load all GO annotations for each subontology (both experimental and curated)."""
        go_annotations = {}

        for subontology in ["BPO", "CCO", "MFO"]:
            # Load experimental annotations
            exp_tsv_path = f"./data/swissprot/2024_01/swissprot_2024_01_{subontology}_exp_annotations.tsv"

            if Path(exp_tsv_path).exists():
                exp_df = pd.read_csv(exp_tsv_path, sep="\t")

                for _, row in exp_df.iterrows():
                    pid = row["EntryID"]
                    # Split terms and clean whitespace
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

            # Load all (curated) annotations
            curated_tsv_path = f"./data/swissprot/2024_01/swissprot_2024_01_{subontology}_annotations.tsv"

            if Path(curated_tsv_path).exists():
                curated_df = pd.read_csv(curated_tsv_path, sep="\t")

                for _, row in curated_df.iterrows():
                    pid = row["EntryID"]
                    # Split terms and clean whitespace
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

    def process_embeddings_h5(self):
        """Process all proteins from H5 embeddings file."""
        h5_path = "./data/swissprot/2024_01/swissprot_esm1b_per_aa.h5"
        with h5py.File(h5_path, "r") as h5f:
            protein_ids = list(h5f.keys())

        successful = 0
        failed = 0
        embedding_dim = None
        num_workers = max(1, min(cpu_count(), len(protein_ids)))

        with Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(
                self.interpro_dict,
                self.go_annotations,
                self.interpro_vocab_size,
                str(self.output_dir),
                h5_path,
            ),
        ) as pool:
            for success, _, _, emb_dim in tqdm(
                pool.imap_unordered(_process_protein, protein_ids),
                total=len(protein_ids),
                desc="Processing proteins",
            ):
                if success:
                    successful += 1
                    if embedding_dim is None and emb_dim is not None:
                        embedding_dim = emb_dim
                else:
                    failed += 1

        logger.info(f"Successfully processed {successful} proteins, failed: {failed}")

        return successful, failed


def main():
    """Main preprocessing function."""

    # Process both experimental and curated annotations
    logger.info("Processing both experimental and curated annotations...")
    preprocessor = ProteinGraphPreprocessor()
    successful, failed = preprocessor.process_embeddings_h5()

    logger.info(f"Processing complete - Successful: {successful}, Failed: {failed}")

    print("Preprocessing complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
