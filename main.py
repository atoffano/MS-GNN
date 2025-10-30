"""Main training script for PFP_layer protein function prediction model.

This script provides the main training loop for the heterogeneous Graph Neural Network
model used for protein function prediction. It handles dataset loading, model initialization,
training, validation, and evaluation across Gene Ontology subontologies.

The script supports:
- Training on SwissProt or custom datasets
- Multi-ontology prediction (MFO, BPO, CCO)
- Model checkpointing and prediction saving
- Weights & Biases integration for experiment tracking
"""

import argparse
import tqdm
import torch
import yaml
import logging
import wandb
import os
import datetime
from src.data.dataloading import SwissProtDataset, define_loaders
from src.models.gnn_model import ProteinGNN
from src.utils.evaluation import save_predictions, evaluate, compute_metrics, plot_aupr
from src.utils.helpers import timeit

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train(config, model, dataset, train_loader, val_loader, test_loader, device):
    """Train the protein function prediction model.

    Args:
        config: Configuration dictionary with training parameters
        model: ProteinGNN model instance
        dataset: SwissProtDataset instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        device: torch.device for computation (CPU or CUDA)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["scheduler"]["step_size"],
        gamma=config["scheduler"]["gamma"],
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=dataset.pos_weights.to(device))

    # Training loop
    for epoch in range(1, config["trainer"]["epochs"] + 1):
        model.train()
        train_loss_sum = 0
        for i, batch in tqdm.tqdm(
            enumerate(train_loader, start=1), desc=f"Training Epoch {epoch}"
        ):
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(
                batch.x_dict,
                batch.edge_index_dict,
                batch,
            )

            loss = criterion(
                out,
                batch["protein"].y,
            )
            loss.backward()
            optimizer.step()
            wandb.log({"train_loss": loss.item() / batch["protein"].batch_size})
            train_loss_sum += loss.item() / batch["protein"].batch_size

            if i % 50 == 0:
                avg_val_loss, val_aupr, val_fmax = run_intermediate_validation(
                    model, val_loader, criterion, device
                )
                wandb.log(
                    {
                        "intermediate_val_loss": avg_val_loss,
                        "intermediate_val_aupr": val_aupr,
                        "intermediate_val_fmax": val_fmax,
                    }
                )
                logger.info(
                    f"Epoch {epoch}, Batch {i}, Train Loss: {train_loss_sum / 50}, Intermediate Val Loss: {avg_val_loss}"
                )
                train_loss_sum = 0
        scheduler.step()
        val_loss, val_aupr, val_fmax, val_pr_plot = run_intermediate_validation(
            model, val_loader, criterion, device, num_batches=len(val_loader)
        )
        test_loss, test_aupr, test_fmax, test_pr_plot = run_intermediate_validation(
            model, test_loader, criterion, device, num_batches=len(test_loader)
        )
        logger.info(
            f"End of epoch: Validation Loss: {val_loss}, Test Loss: {test_loss}\n"
            f"Validation AUPR: {val_aupr}, Val F-max: {val_fmax}\n"
            f"Test AUPR: {test_aupr}, Test F-max: {test_fmax}"
        )
        wandb.log(
            {
                "end_epoch_val_loss": val_loss,
                "end_epoch_val_aupr": val_aupr,
                "end_epoch_val_fmax": val_fmax,
                "end_epoch_val_pr_curve": val_pr_plot,
                "end_epoch_test_loss": test_loss,
                "end_epoch_test_aupr": test_aupr,
                "end_epoch_test_fmax": test_fmax,
                "end_epoch_test_pr_curve": test_pr_plot,
            }
        )


def run_intermediate_validation(model, val_loader, criterion, device, num_batches=5):
    """Run validation on a subset of batches during training.

    Args:
        model: ProteinGNN model instance
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: torch.device for computation
        num_batches: Number of batches to validate on (default: 5)

    Returns:
        tuple: (avg_loss, aupr, fmax) or (avg_loss, aupr, fmax, pr_plot)
               if num_batches equals full validation set
    """
    if num_batches >= len(val_loader):
        num_batches = len(val_loader)
    model.eval()
    val_loss_sum = 0
    all_scores, all_targets = [], []
    with torch.no_grad():
        for _ in range(num_batches):
            val_batch = next(iter(val_loader))
            val_batch = val_batch.to(device)
            val_out = model(
                val_batch.x_dict,
                val_batch.edge_index_dict,
                val_batch,
            )
            val_loss = criterion(
                val_out,
                val_batch["protein"].y,
            )
            val_loss_sum += val_loss.item() / val_batch["protein"].batch_size
            all_scores.append(val_out.cpu().numpy())
            all_targets.append(val_batch["protein"].y.cpu().numpy())

    precision, recall, aupr, fmax = compute_metrics(all_scores, all_targets)
    val_loss = val_loss_sum / num_batches if num_batches > 0 else val_loss_sum
    model.train()

    if num_batches == len(val_loader):
        precision, recall, aupr, fmax = compute_metrics(all_scores, all_targets)
        pr_plot = plot_aupr(precision, recall, aupr)
        return val_loss, aupr, fmax, pr_plot
    return val_loss, aupr, fmax


def main():
    """Main entry point for training the protein function prediction model.

    Parses command-line arguments, loads configuration, initializes datasets
    and models, runs training, and performs evaluation across GO subontologies.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/toy_cfg.yaml")
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = config["data"]["dataset"]
    run_id = f"{time_str}_{dataset_name}_{config['run']['qualifier']}"

    # Create results directory
    config["run"]["results_dir"] = f"./results/{config['data']['dataset']}/{run_id}"
    os.makedirs(config["run"]["results_dir"], exist_ok=True)
    logger.info(
        "Config:\n" + yaml.dump(config, sort_keys=False, default_flow_style=False)
    )

    for subontology in config["data"]["subontology"]:
        # Initialize wandb
        wandb.init(
            project="PFP_layer",
            config=config,
            name=f"{run_id}_{dataset_name}_{subontology}",
        )
        wandb.run.log_code(".")

        # Set up logging to file
        log_file = os.path.join(config["run"]["results_dir"], f"{subontology}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

        device = torch.device(
            config["trainer"]["device"] if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {device}")
        logger.info(f"Results will be saved to: {config['run']['results_dir']}")

        # Create efficient dataset
        logger.info("Creating dataset...")
        dataset = SwissProtDataset(config)

        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader = define_loaders(config, dataset)

        # Get vocab size for current subontology
        go_vocab_size = dataset.go_vocab_sizes[subontology]

        logger.info(f"Dataset loaded with {len(dataset.proteins)} total proteins")
        logger.info(f"GO vocabulary size for {subontology}: {go_vocab_size}")
        logger.info(f"InterPro vocabulary size: {dataset.ipr_vocab_size}")
        logger.info(f"Number of training proteins: {len(dataset.train_proteins)}")
        logger.info(f"Number of validation proteins: {len(dataset.val_proteins)}")
        logger.info(f"Number of test proteins: {len(dataset.test_proteins)}")

        # Instantiate model with config values
        model = ProteinGNN(
            config,
            dataset,
        )
        model = model.to(device)
        if config["trainer"].get("compile", False):
            model = torch.compile(model)
            logger.info("Model compiled with torch.compile()")
        else:
            logger.info("Model not compiled; running in standard mode.")
        logger.info(f"Model: {model}")

        # Training
        train(config, model, dataset, train_loader, val_loader, test_loader, device)

        # Predictions
        splits = ["test"] if config["run"]["test_only"] else ["val", "test"]
        for split in splits:
            loader = val_loader if split == "val" else test_loader
            # Save predictions
            if config["run"]["save_predictions"][split]:
                save_predictions(config, model, loader, device, dataset, split)
                logger.info(
                    f"Saved predictions to {config['run']['results_dir']}/predictions_{split}_{dataset.subontology}.tsv"
                )

        if config["run"].get("save_model"):
            model_path = os.path.join(config["run"]["results_dir"], f"model.pth")
            torch.save(model.state_dict(), model_path)
            with open(
                os.path.join(config["run"]["results_dir"], f"cfg.yaml"), "w"
            ) as f:
                yaml.dump(config, f)
            logger.info(f"Model saved to {model_path}")
            # wandb.save(model_path)

        # Compute evaluation metrics
        logger.info("Starting evaluation...")
        if config["run"]["save_predictions"]["val"]:
            evaluate(
                logger,
                config["data"]["dataset"],
                config["run"]["results_dir"],
                subontology,
                split="val",
            )
        if config["run"]["save_predictions"]["test"] or not config["run"]["test_only"]:
            evaluate(
                logger,
                config["data"]["dataset"],
                config["run"]["results_dir"],
                subontology,
                split="test",
            )

        logger.info(f"Evaluation completed")
        wandb.finish()

    logger.info("Done!")


if __name__ == "__main__":
    main()
