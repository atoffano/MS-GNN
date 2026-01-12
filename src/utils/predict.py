import argparse
import os
import sys
import yaml
import torch
import logging
from datetime import datetime

# Add project root to path to allow imports if run directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.dataloading import SwissProtDataset, define_loaders, make_batch_transform
from torch_geometric.loader import NeighborLoader
from src.models.gnn_model import ProteinGNN
from src.utils.evaluation import save_predictions

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def predict_from_directory(
    results_dir,
    splits=["test"],
    device_name="cuda",
    checkpoint_path=None,
    proteins=None,
    tau=None,
    name=None,
    subontology=None,
):
    """
    Loads model and config from results_dir and generates predictions.

    Args:
        results_dir (str): Path to the directory containing cfg.yaml and model checkpoints.
        splits (list or str): Dataset split(s) to predict on ('val', 'test', or both).
        device_name (str): Device to use ('cuda' or 'cpu').
        checkpoint_path (str, optional): Path to a specific model checkpoint to use.
    """
    config_path = os.path.join(results_dir, "cfg.yaml")
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at {config_path}")
        return

    logger.info(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["run"]["results_dir"] = results_dir

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading dataset...")
    config["data"]["subontology"] = subontology
    dataset = SwissProtDataset(config)

    logger.info("Creating data loaders...")
    _, val_loader, test_loader = define_loaders(config, dataset)
    loaders = {"val": val_loader, "test": test_loader}

    if proteins:
        # Create a custom loader for specified proteins
        if len(proteins) == 1 and os.path.isfile(proteins[0]):
            if proteins[0].endswith(".tsv"):
                protein_list = []
                with open(proteins[0], "r") as f:
                    header = f.readline().strip().split("\t")
                    try:
                        col_idx = header.index("EntryID")
                    except ValueError:  # 'EntryID' not found; default to first column
                        col_idx = 0
                    for line in f:
                        if line.strip():
                            parts = line.strip().split("\t")
                            if len(parts) > col_idx:
                                protein_list.append(parts[col_idx])
            else:
                with open(proteins[0], "r") as f:
                    protein_list = [line.strip() for line in f if line.strip()]
        else:
            protein_list = proteins
        if dataset.uses_entryid:
            mapped_proteins = []
            for pid in protein_list:
                if "_" not in pid and pid in dataset.rev_pid_mapping:
                    mapped_proteins.append(dataset.rev_pid_mapping[pid])
                else:
                    mapped_proteins.append(pid)
            protein_list = mapped_proteins
        mask = torch.zeros(len(dataset.proteins), dtype=torch.bool)
        protein_idx = [
            dataset.protein_to_idx[pid]
            for pid in protein_list
            if pid in dataset.protein_to_idx
        ]
        mask[protein_idx] = True

        num_neighbors = {}
        for edge_type_str, num_samples in config["model"]["sampled_edges"].items():
            edge_type_tuple = tuple(edge_type_str.split("__"))
            num_neighbors[edge_type_tuple] = [num_samples]

        custom_loader = NeighborLoader(
            dataset.data,
            num_neighbors=num_neighbors,
            # batch_size=config["model"]["batch_size"],
            batch_size=8,
            input_nodes=("protein", mask),
            transform=make_batch_transform(dataset, mode="predict"),
            shuffle=False,
            num_workers=config["trainer"]["num_workers"],
        )
        split_name = (
            f"{name if name else 'custom'}_{datetime.now().strftime('%y%m%d_%H%M%S')}"
        )
        splits = [split_name]
        loaders[split_name] = custom_loader

    if isinstance(splits, str):
        splits = [splits]

    logger.info(f"Processing subontology: {subontology}")
    model = ProteinGNN(config, dataset)
    model = model.to(device)

    model_path = os.path.join(results_dir, checkpoint_path)
    if os.path.exists(model_path):
        logger.info(f"Loading model from {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            logger.warning(f"Failed to load {model_path}: {e}")
            exit(1)

    for split in splits:
        loader = loaders.get(split)
        logger.info(f"Generating predictions for {split} set...")
        save_predictions(config, model, loader, device, dataset, split, tau)

        output_file = f"{config['run']['results_dir']}/predictions/predictions_{split}_{dataset.subontology}.tsv"
        logger.info(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run predictions using a trained model from a results directory."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the results directory containing cfg.yaml and model checkpoints.",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        default=["test"],
        choices=["val", "test"],
        help="Dataset split(s) to predict on (default: test). Can specify multiple (e.g. --split val test)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a specific model checkpoint to use (optional, default is best model).",
    )
    parser.add_argument(
        "--proteins",
        nargs="+",
        default=None,
        help="Protein IDs to predict on as either a list of protein IDs, a file with one protein per line, or a .tsv with a column named 'EntryID'.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Threshold tau for filtering predictions.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name for the prediction output (optional).",
    )
    parser.add_argument(
        "--subontology",
        type=str,
        default=None,
        required=True,
        help="Which subontology to use.",
    )

    args = parser.parse_args()

    predict_from_directory(
        args.input_dir,
        args.split,
        args.device,
        args.checkpoint,
        args.proteins,
        args.tau,
        args.name,
        args.subontology,
    )
