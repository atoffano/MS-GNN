import tqdm
import torch
import h5py
import yaml
import logging
import wandb
import os
import datetime
import torch_geometric.transforms as T
from torch_geometric.explain import Explainer, CaptumExplainer, AttentionExplainer
from src.utils.visualize import visualize_graph_via_networkx, plot_aa_edge_histogram
from src.data.dataloading import SwissProtDataset, define_loaders
from src.models.gnn_model import ProteinGNN
from src.utils.helpers import timeit
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@timeit
def train_epoch(model, optimizer, loader, device, epoch):
    model.train()
    total_loss = 0
    for batch in tqdm.tqdm(loader, desc=f"Training Epoch {epoch}"):
        batch = batch.to(device)

        optimizer.zero_grad()
        t1 = time.time()
        out = model(batch.x_dict, batch.edge_index_dict, batch)
        t2 = time.time()
        logger.debug(f"Forward pass time: {t2 - t1:.4f} seconds")

        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(
            out,
            batch["protein"].go[: batch["protein"].batch_size],
        )
        t1 = time.time()
        loss.backward()
        t2 = time.time()
        logger.debug(f"Backward pass time: {t2 - t1:.4f} seconds")
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    wandb.log({"train_loss": avg_loss, "epoch": epoch})
    logger.info(f"Epoch {epoch} - Train loss: {avg_loss:.4f}")
    return avg_loss


def validation(model, loader, device, epoch):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Validation"):
            if batch is None:
                continue

            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict, batch)

            criterion = torch.nn.BCEWithLogitsLoss()
            loss = criterion(
                out,
                batch["protein"].go[: batch["protein"].batch_size],
            )
            total_loss += loss.item()

        #     # Explanation generation
        #     explainer = Explainer(
        #         model,  # It is assumed that model outputs a single tensor.
        #         algorithm=CaptumExplainer("IntegratedGradients"),
        #         explanation_type="model",
        #         node_mask_type="attributes",
        #         edge_mask_type="object",
        #         model_config=dict(
        #             mode="multiclass_classification",
        #             task_level="node",
        #             return_type="raw",  # Model returns probabilities.
        #         ),
        #     )

        #     hetero_explanation = explainer(
        #         batch.x_dict,
        #         batch.edge_index_dict,
        #         batch=batch,
        #         target=None,
        #         # index=None,
        #         index=torch.arange(batch["protein"].batch_size),
        #         # batch["protein"].go[: batch["protein"].batch_size]
        #     )
        #     logger.info(
        #         f"Generated explanations in {hetero_explanation.available_explanations}"
        #     )

        # path = os.path.join(results_dir, "feature_importance.png")
        # hetero_explanation.visualize_feature_importance(path, top_k=10)
        # logger.info(f"Feature importance plot has been saved to '{path}'")
        # wandb.log({"feature_importance_plot": wandb.Image(path)})

        # # Visualize graph via NetworkX for specified edge types
        # graph_path = os.path.join(results_dir, "graph_explanation")
        # plots = visualize_graph_via_networkx(
        #     hetero_explanation,
        #     path=graph_path,  # Base path; function will append suffix for each edge type
        #     cutoff_edge=0.00001,
        #     edge_types_to_plot=[("aa", "aa2protein", "protein")],
        # )
        # # Log each saved plot to wandb
        # for edge_type, fig in plots.items():
        #     suffix = "_".join(edge_type)
        #     plot_path = f"{graph_path}_{suffix}.png"
        #     wandb.log({f"graph_explanation_plot_{suffix}": wandb.Image(plot_path)})

        # # Plot AA edge histogram
        # path = os.path.join(results_dir, "aa_edge_histogram.png")
        # ret = plot_aa_edge_histogram(
        #     hetero_explanation,
        #     edge_type=("aa", "aa2protein", "protein"),
        #     path=path,
        # )
        # wandb.log({"aa_edge_histogram_plot": wandb.Image(ret["fig"])})

    avg_loss = total_loss / len(loader)
    wandb.log({"val_loss": avg_loss, "epoch": epoch})
    logger.info(f"Validation loss: {avg_loss:.4f}")
    return avg_loss


def main():
    # Load config from YAML
    config_path = "src/configs/cfg.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = config["data"]["dataset"]
    run_id = f"{time_str}_{dataset_name}_{config['run']['qualifier']}_efficient"

    # Create results directory
    results_dir = f"./results/{config['data']['dataset']}/{run_id}"
    os.makedirs(results_dir, exist_ok=True)

    for subontology in config["data"]["subontology"]:
        # Initialize wandb
        wandb.init(
            project="PFP_layer",
            config=config,
            name=f"{run_id}_{dataset_name}_{subontology}",
        )
        wandb.run.log_code(".")

        # Set up logging to file
        log_file = os.path.join(results_dir, f"{subontology}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

        device = torch.device(
            config["trainer"]["device"] if torch.cuda.is_available() else "cpu"
        )

        # Update config for current subontology
        config["data"]["subontology"] = [subontology]  # Set current subontology

        # Create efficient dataset
        logger.info("Creating efficient dataset...")
        dataset = SwissProtDataset(config, split="train")

        # Create efficient loaders
        logger.info("Creating efficient data loaders...")
        train_loader, val_loader, test_loader = define_loaders(config, dataset)

        # Get vocab size for current subontology
        go_vocab_size = dataset.go_vocab_sizes[subontology]

        logger.info(
            f"Dataset loaded with {len(dataset.available_proteins)} total proteins"
        )
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

        optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["lr"])

        # Scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["scheduler"]["step_size"],
            gamma=config["scheduler"]["gamma"],
        )

        # Training loop
        for epoch in range(1, config["trainer"]["epochs"] + 1):
            train_loss = train_epoch(model, optimizer, train_loader, device, epoch)
            val_loss = validation(model, val_loader, device, epoch)
            scheduler.step()

        # Save model
        model_path = os.path.join(results_dir, f"model_{subontology}.pth")
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        wandb.save(model_path)

        wandb.finish()

    logger.info("Training complete for all subontologies!")


if __name__ == "__main__":
    main()
