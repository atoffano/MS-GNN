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
from src.data.dataloading import SwissProtHeteroDataset, define_loaders
from src.models.gnn_model import HeteroProteinGNN
from src.utils.helpers import timeit

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@timeit
def train_epoch(model, optimizer, loader, dataset, device, epoch):
    model.train()
    total_loss = 0
    with h5py.File(dataset.prot_emb_path, "r") as h5_file:
        for batch in loader:
            batch = dataset.get_batch_features(batch, h5_file)
            batch = batch.to(device)

            optimizer.zero_grad()
            out = model(batch.x_dict, batch.edge_index_dict, batch)

            criterion = torch.nn.BCEWithLogitsLoss()
            loss = criterion(
                out,
                batch["protein"].go[: batch["protein"].batch_size],
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    wandb.log({"train_loss": avg_loss, "epoch": epoch})
    logger.info(f"Epoch {epoch} - Train loss: {avg_loss:.4f}")
    return avg_loss


def validation(model, loader, dataset, device, epoch, results_dir):
    model.eval()
    total_loss = 0
    with h5py.File(dataset.prot_emb_path, "r") as h5_file:
        with torch.no_grad():
            for batch in loader:
                batch = dataset.get_batch_features(batch, h5_file)
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
    run_qualifier = config["run"]["qualifier"]
    dataset = config["data"]["dataset"]
    run_id = f"{time_str}_{dataset}_{run_qualifier}"

    # Create results directory
    results_dir = f"./results/{config['data']['dataset']}/{run_id}"
    os.makedirs(results_dir, exist_ok=True)

    for subontology in config["data"]["subontology"]:
        # Initialize wandb
        wandb.init(
            project="PFP_layer",
            config=config,
            name=f"{run_id}_{dataset}_{subontology}",
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

        # Paths from config
        datapath = {
            "interpro": f"{config['constants']['project_path']}{config['data']['interpro_path'][1:]}",
            "prot_emb": f"{config['constants']['project_path']}{config['data']['prot_emb_path'][1:]}",
            "alignments": f"{config['constants']['project_path']}{config['data']['alignment_path'][1:]}",
            "train": f"{config['constants']['project_path']}/data/{dataset}/{dataset}_{subontology}_train_annotations.tsv",
            "val": f"{config['constants']['project_path']}/data/{dataset}/{dataset}_{subontology}_val_annotations.tsv",
            "test": f"{config['constants']['project_path']}/data/{dataset}/{dataset}_{subontology}_test_annotations.tsv",
        }
        # Optionally add SwissProt training data
        if config["data"].get("train_on_swissprot", True):
            exp = "exp_" if config["data"].get("swissprot_exp_only", True) else ""
            datapath["train"] = (
                f"{config['constants']['project_path']}/data/swissprot/2024_01/swissprot_2024_01_{subontology}_{exp}annotations.tsv"
            )

        dataset = SwissProtHeteroDataset(datapath, split="train")

        transform = T.Compose(
            [
                T.ToUndirected(reduce="mean", merge=True),
                T.RemoveDuplicatedEdges(reduce="mean"),
            ]
        )
        dataset.data = transform(dataset.data)
        dataset.data.validate(raise_on_error=True)
        logger.info(f"Dataset: {dataset.data}")
        assert dataset.data.validate()

        train_loader, val_loader, test_loader = define_loaders(config, dataset)

        logger.info(f"Dataset has {dataset.data['protein'].num_nodes} protein nodes")
        logger.info(f"GO vocabulary size: {dataset.go_vocab_size}")
        logger.info(f"InterPro vocabulary size: {dataset.ipr_vocab_size}")
        logger.info(f"Number of training proteins: {dataset.train_mask.sum().item()}")
        logger.info(f"Number of validation proteins: {dataset.val_mask.sum().item()}")
        logger.info(f"Number of test proteins: {dataset.test_mask.sum().item()}")

        # Instantiate model with config values
        model = HeteroProteinGNN(
            hidden_channels=config["model"]["hidden_channels"],
            out_channels=dataset.go_vocab_size,
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

        for epoch in range(1, config["trainer"]["epochs"] + 1):
            loss = train_epoch(model, optimizer, train_loader, dataset, device, epoch)
            scheduler.step()

        # Save model
        model_path = os.path.join(results_dir, "model.pth")
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        wandb.save(model_path)

        # Validation after training
        val_loss = validation(model, val_loader, dataset, device, epoch, results_dir)
        wandb.finish()


if __name__ == "__main__":
    main()
