"""Main training script for PFP_layer protein function prediction model.

This script provides the main training loop for the heterogeneous Graph Neural Network
model used for protein function prediction. It handles dataset loading, model initialization,
training, validation, and evaluation across Gene Ontology subontologies.

The script supports:
- Training on SwissProt or custom datasets
- Multi-ontology prediction (MFO, BPO, CCO)
- Model checkpointing and prediction saving
- Weights & Biases integration for experiment tracking
- Resume training from checkpoints
"""

import argparse
import gc
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
from src.utils.train_utils import save_checkpoint, load_checkpoint
from src.utils.helpers import timeit, MemoryTracker, log_process_tree_memory

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
import time


def train(
    config,
    model,
    dataset,
    train_loader,
    val_loader,
    test_loader,
    device,
    subontology,
    start_epoch=1,
    best_val_aupr=0.0,
):
    """Train the protein function prediction model.

    Args:
        config: Configuration dictionary with training parameters
        model: ProteinGNN model instance
        dataset: SwissProtDataset instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        device: torch.device for computation (CPU or CUDA)
        subontology: Current GO subontology being trained
        start_epoch: Epoch to start/resume training from (default: 1)
        best_val_aupr: Best validation AUPR achieved so far (default: 0.0)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["scheduler"]["step_size"],
        gamma=config["scheduler"]["gamma"],
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=config["trainer"]["epochs"]
    # )

    # Load checkpoint if resuming
    if start_epoch > 1:
        checkpoint_path = os.path.join(
            config["run"]["results_dir"],
            "checkpoints",
            f"checkpoint_latest_{subontology}.pth",
        )
        if os.path.exists(checkpoint_path):
            start_epoch, _, best_val_aupr = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler, device
            )

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=dataset.pos_weights.to(device))

    # Training loop
    for epoch in range(start_epoch, config["trainer"]["epochs"] + 1):
        model.train()
        train_loss_sum = 0

        # Log memory before starting epoch
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Epoch {epoch}")
        # log_process_tree_memory()
        for i, batch in tqdm.tqdm(
            enumerate(train_loader, start=1), desc=f"Training Epoch {epoch}"
        ):
            try:
                nb_prots = len(batch["protein"].n_id)
                logger.info(f"Number of proteins in batch: {nb_prots}")
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

            # Handle OOM errors gracefully
            except RuntimeError as e:
                logger.error(
                    f"Error occurred {e}, Batch {i}. Total proteins in batch: {nb_prots}. Skipping batch."
                )
                torch.cuda.empty_cache()
                continue

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
                "epoch": epoch,
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

        # Save checkpoint at end of each epoch
        save_checkpoint(
            config,
            model,
            optimizer,
            scheduler,
            epoch,
            subontology,
            val_aupr,
        )

        # Save best model
        if val_aupr > best_val_aupr:
            best_val_aupr = val_aupr
            best_model_path = os.path.join(
                config["run"]["results_dir"], f"best_model.pth"
            )
            torch.save(model.state_dict(), best_model_path)
            logger.info(
                f"Saved best model with AUPR {best_val_aupr:.4f} to {best_model_path}"
            )
            wandb.log({"best_val_aupr": best_val_aupr})

        # free memory, both CPU and GPU
        torch.cuda.empty_cache()
        gc.collect()


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
    tracker = MemoryTracker(log_prefix="[Main] ")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/toy_cfg.yaml")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from",
    )
    args = parser.parse_args()
    config_path = args.config

    tracker.set_baseline()
    tracker.log_memory("After imports and argument parsing")

    if args.resume:
        # Load config from checkpoint directory
        checkpoint_config_path = os.path.join(args.resume, "cfg.yaml")
        if os.path.exists(checkpoint_config_path):
            with open(checkpoint_config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from checkpoint: {checkpoint_config_path}")
            config["run"]["results_dir"] = args.resume
        else:
            logger.error(
                f"Config file not found in checkpoint directory: {checkpoint_config_path}"
            )
            return
    else:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = config["data"]["dataset"]
        run_id = f"{time_str}_{dataset_name}_{config['run']['qualifier']}"

        # Create results directory
        config["run"]["results_dir"] = f"./results/{config['data']['dataset']}/{run_id}"
        os.makedirs(config["run"]["results_dir"], exist_ok=True)

        # Save config to results directory
        with open(os.path.join(config["run"]["results_dir"], "cfg.yaml"), "w") as f:
            yaml.dump(config, f)

    logger.info(
        "Config:\n" + yaml.dump(config, sort_keys=False, default_flow_style=False)
    )

    for subontology in config["data"]["subontology"]:
        # Check if resuming from checkpoint
        start_epoch = 1
        best_val_aupr = 0.0
        checkpoint_path = os.path.join(
            config["run"]["results_dir"],
            "checkpoints",
            f"checkpoint_latest_{subontology}.pth",
        )

        # Initialize wandb
        wandb_id = None
        if args.resume and os.path.exists(checkpoint_path):
            # Try to resume wandb run
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            start_epoch = checkpoint["epoch"] + 1
            best_val_aupr = checkpoint.get("best_val_aupr", 0.0)

        run_name = f"{os.path.basename(config['run']['results_dir'])}_{subontology}"
        wandb.init(
            project="PFP_layer",
            config=config,
            name=run_name,
            id=wandb_id,
            resume="allow" if wandb_id else None,
        )
        wandb.run.log_code(".")

        # Set up logging to file
        log_file = os.path.join(config["run"]["results_dir"], f"{subontology}.log")
        file_handler = logging.FileHandler(log_file, mode="a" if args.resume else "w")
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

        # Load model checkpoint if resuming
        if args.resume and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded model from checkpoint at epoch {checkpoint['epoch']}")

        if config["trainer"].get("compile", False):
            model = torch.compile(model)
            logger.info("Model compiled with torch.compile()")
        else:
            logger.info("Model not compiled; running in standard mode.")
        logger.info(f"Model: {model}")

        # Training
        train(
            config,
            model,
            dataset,
            train_loader,
            val_loader,
            test_loader,
            device,
            subontology,
            start_epoch,
            best_val_aupr,
        )

        # Load best model for evaluation
        best_model_path = os.path.join(
            config["run"]["results_dir"], f"best_model_{subontology}.pth"
        )
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            logger.info(f"Loaded best model for evaluation from {best_model_path}")

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
            logger.info(f"Model saved to {model_path}")

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

        # Remove file handler to avoid duplicate logs
        logger.removeHandler(file_handler)
        file_handler.close()

    logger.info("Done!")


if __name__ == "__main__":
    main()
