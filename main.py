import tqdm
import torch
import yaml
import logging
import wandb
import os
import datetime
import time
from torch_geometric.explain import Explainer, CaptumExplainer, AttentionExplainer
from src.utils.visualize import visualize_graph_via_networkx, plot_aa_edge_histogram
from src.data.dataloading import SwissProtDataset, define_loaders
from src.models.gnn_model import ProteinGNN
from src.utils.evaluation import save_predictions, evaluate
from src.utils.helpers import timeit

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_intermediate_validation(model, val_loader, criterion, device, num_batches=5):
    if num_batches > len(val_loader):
        num_batches = len(val_loader)
    model.eval()
    val_loss_sum = 0
    val_batches_run = 0
    with torch.no_grad():
        for _ in range(num_batches):
            val_batch = next(iter(val_loader))
            val_batch = val_batch.to(device)
            val_out = model(val_batch.x_dict, val_batch.edge_index_dict, val_batch)
            val_loss = criterion(
                val_out,
                val_batch["protein"].go[: val_batch["protein"].batch_size],
            )
            val_loss_sum += val_loss.item()
            val_batches_run += 1
    avg_val_loss = val_loss_sum / val_batches_run
    model.train()
    return avg_val_loss


def train(config, model, train_loader, val_loader, test_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["scheduler"]["step_size"],
        gamma=config["scheduler"]["gamma"],
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(1, config["trainer"]["epochs"] + 1):
        model.train()
        batch_count = 0
        for batch in tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x_dict, batch.edge_index_dict, batch)

            loss = criterion(
                out,
                batch["protein"].go[: batch["protein"].batch_size],
            )
            loss.backward()
            optimizer.step()
            wandb.log({"train_loss": loss.item()})
            batch_count += 1

            if batch_count % 50 == 0:
                avg_val_loss = run_intermediate_validation(
                    model, val_loader, criterion, device
                )
                wandb.log(
                    {
                        "intermediate_val_loss": avg_val_loss,
                        "lr": scheduler.get_last_lr()[0],
                    }
                )
                batch_count = 0
                logger.info(
                    f"Epoch {epoch}, Batch {batch_count}, Train Loss: {loss.item():.4f}, Intermediate Val Loss: {avg_val_loss:.4f}"
                )
        scheduler.step()
    val_loss = run_intermediate_validation(
        model, val_loader, criterion, device, num_batches=len(val_loader)
    )
    test_loss = run_intermediate_validation(
        model, test_loader, criterion, device, num_batches=len(test_loader)
    )
    logger.info(f"Final Validation Loss: {val_loss:.4f}")
    logger.info(f"Final Test Loss: {test_loss:.4f}")
    wandb.log(
        {
            "final_val_loss": val_loss,
            "final_test_loss": test_loss,
        }
    )


def main():
    config_path = "src/configs/toy_cfg.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = config["data"]["dataset"]
    run_id = f"{time_str}_{dataset_name}_{config['run']['qualifier']}"

    # Create results directory
    config["run"]["results_dir"] = f"./results/{config['data']['dataset']}/{run_id}"
    os.makedirs(config["run"]["results_dir"], exist_ok=True)

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
        dataset = SwissProtDataset(config, split="train")

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
            hidden_channels=config["model"]["hidden_channels"],
            out_channels=go_vocab_size,
        )
        model = model.to(device)
        if config["trainer"].get("compile", False):
            model = torch.compile(model)
            logger.info("Model compiled with torch.compile()")
        else:
            logger.info("Model not compiled; running in standard mode.")
        logger.info(f"Model: {model}")

        # Training loop
        train(config, model, train_loader, val_loader, test_loader, device)
        splits = ["test"] if config["run"]["test_only"] else ["val", "test"]
        for split in splits:
            loader = val_loader if split == "val" else test_loader
            # Save predictions
            if config["run"]["save_predictions"][split]:
                save_predictions(config, model, loader, device, dataset, split)
                logger.info(
                    f"Saved predictions to {config['run']['results_dir']}/predictions_{split}_{dataset.subontology}.tsv"
                )

        # Save model
        if config["trainer"].get("save_model"):
            model_path = os.path.join(
                config["run"]["results_dir"], f"model_{subontology}.pth"
            )
            torch.save(model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")
            # wandb.save(model_path)

        # Compute evaluation metrics
        logger.info("Starting evaluation...")
        if config["run"]["save_predictions"]["val"]:
            evaluate(
                config, logger, config["run"]["results_dir"], subontology, split="val"
            )
        if config["run"]["save_predictions"]["test"] or not config["run"]["test_only"]:
            evaluate(
                config, logger, config["run"]["results_dir"], subontology, split="test"
            )

        logger.info(f"Evaluation completed")
        wandb.finish()

    logger.info("Done!")


if __name__ == "__main__":
    main()
