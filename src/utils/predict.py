import argparse
import os
import sys
import yaml
import torch
import logging

# Add project root to path to allow imports if run directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.dataloading import SwissProtDataset, define_loaders
from src.models.gnn_model import ProteinGNN
from src.utils.evaluation import save_predictions

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def predict_from_directory(results_dir, splits=["test"], device_name="cuda"):
    """
    Loads model and config from results_dir and generates predictions.

    Args:
        results_dir (str): Path to the directory containing cfg.yaml and model checkpoints.
        splits (list or str): Dataset split(s) to predict on ('val', 'test', or both).
        device_name (str): Device to use ('cuda' or 'cpu').
    """
    # 1. Load Config
    config_path = os.path.join(results_dir, "cfg.yaml")
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at {config_path}")
        return

    logger.info(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override results_dir in config to match the input directory
    # This ensures predictions are saved in the correct place
    config["run"]["results_dir"] = results_dir

    # 2. Setup Device
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 3. Load Dataset
    logger.info("Loading dataset...")
    # The dataset initialization might take some time
    dataset = SwissProtDataset(config)

    # 4. Create Loaders
    logger.info("Creating data loaders...")
    _, val_loader, test_loader = define_loaders(config, dataset)

    # Normalize splits to list
    if isinstance(splits, str):
        splits = [splits]

    # 5. Iterate over subontologies defined in config
    subontologies = config["data"]["subontology"]
    if isinstance(subontologies, str):
        subontologies = [subontologies]

    for subontology in subontologies:
        logger.info(f"Processing subontology: {subontology}")

        # Update dataset subontology context
        # This is crucial for save_predictions to use the correct vocabulary and filename
        dataset.subontology = subontology

        # 6. Initialize Model
        # We re-initialize or re-load for each subontology to ensure correct state
        model = ProteinGNN(config, dataset)
        model = model.to(device)

        # 7. Load Best Model Weights
        # Priority: best_model_{subontology}.pth -> best_model.pth -> model.pth
        model_files = [f"best_model_{subontology}.pth", "best_model.pth", "model.pth"]

        checkpoint_loaded = False
        for model_file in model_files:
            model_path = os.path.join(results_dir, model_file)
            if os.path.exists(model_path):
                logger.info(f"Loading model from {model_path}")
                try:
                    checkpoint = torch.load(model_path, map_location=device)
                    if (
                        isinstance(checkpoint, dict)
                        and "model_state_dict" in checkpoint
                    ):
                        model.load_state_dict(checkpoint["model_state_dict"])
                    else:
                        model.load_state_dict(checkpoint)
                    checkpoint_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {model_path}: {e}")

        if not checkpoint_loaded:
            logger.warning(
                f"No valid model checkpoint found for {subontology} in {results_dir}. Skipping."
            )
            continue

        # 8. Make Predictions for each requested split
        for split in splits:
            if split == "val":
                loader = val_loader
            elif split == "test":
                loader = test_loader
            else:
                logger.warning(f"Invalid split: {split}. Skipping.")
                continue

            if loader is None:
                logger.warning(
                    f"Loader for split '{split}' is None. Check your data configuration."
                )
                continue

            logger.info(f"Generating predictions for {split} set...")
            save_predictions(config, model, loader, device, dataset, split)

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

    args = parser.parse_args()

    predict_from_directory(args.input_dir, args.split, args.device)
