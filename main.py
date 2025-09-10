import tqdm
import torch
import h5py
import yaml
import torch_geometric.transforms as T
from torch_geometric.explain import Explainer, CaptumExplainer, AttentionExplainer
from src.data.dataloading import SwissProtHeteroDataset, define_loaders
from src.models.gnn_model import HeteroProteinGNN
from src.utils.helpers import timeit
from src.utils.visualize import visualize_graph_via_networkx, plot_aa_edge_histogram


@timeit
def train_epoch(model, optimizer, loader, dataset, h5_path, device):
    model.train()
    total_loss = 0
    with h5py.File(h5_path, "r") as h5_file:
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
    return total_loss / len(loader)


def validation(model, loader, dataset, h5_path, device):
    model.eval()
    total_loss = 0
    with h5py.File(h5_path, "r") as h5_file:
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

                # Explanation generation
                explainer = Explainer(
                    model,  # It is assumed that model outputs a single tensor.
                    algorithm=CaptumExplainer("IntegratedGradients"),
                    explanation_type="model",
                    node_mask_type="attributes",
                    edge_mask_type="object",
                    model_config=dict(
                        mode="multiclass_classification",
                        task_level="node",
                        return_type="raw",  # Model returns probabilities.
                    ),
                )

                hetero_explanation = explainer(
                    batch.x_dict,
                    batch.edge_index_dict,
                    batch=batch,
                    target=None,
                    # index=None,
                    index=torch.arange(batch["protein"].batch_size),
                    # batch["protein"].go[: batch["protein"].batch_size]
                )
                print(
                    f"Generated explanations in {hetero_explanation.available_explanations}"
                )

                path = "feature_importance.png"
                hetero_explanation.visualize_feature_importance(path, top_k=10)
                print(f"Feature importance plot has been saved to '{path}'")
                visualize_graph_via_networkx(
                    hetero_explanation,
                    path="graph_explanation.png",
                    cutoff_edge=0.00001,
                )
                plot_aa_edge_histogram(
                    hetero_explanation,
                    edge_type=("aa", "aa2protein", "protein"),
                    path="aa_edge_histogram.png",
                )
    return total_loss / len(loader)


def main():
    # Load config from YAML
    config_path = "src/configs/cfg.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        config["trainer"]["device"] if torch.cuda.is_available() else "cpu"
    )

    # Paths from config
    tsv_paths = {
        "InterPro": f"{config['constants']['project_path']}{config['data']['interpro_path'][1:]}",  # Remove leading ./ if present
    }
    go_split_paths = {
        "train": f"{config['constants']['project_path']}{config['data']['go_train_path'][1:]}",
        "val": f"{config['constants']['project_path']}{config['data']['go_val_path'][1:]}",
        "test": f"{config['constants']['project_path']}{config['data']['go_test_path'][1:]}",
    }
    alignment_path = (
        f"{config['constants']['project_path']}{config['data']['alignment_path'][1:]}"
    )
    h5_path = f"{config['constants']['project_path']}{config['data']['h5_path'][1:]}"

    # Initialize dataset
    dataset = SwissProtHeteroDataset(
        h5_path, tsv_paths, go_split_paths, alignment_path, split="train"
    )

    transform = T.Compose(
        [
            T.ToUndirected(reduce="mean", merge=True),
            T.RemoveDuplicatedEdges(reduce="mean"),
        ]
    )
    dataset.data = transform(dataset.data)
    dataset.data.validate(raise_on_error=True)
    print(dataset.data)
    assert dataset.data.validate()

    train_loader, val_loader, test_loader = define_loaders(config, dataset)

    print(f"Dataset has {dataset.data['protein'].num_nodes} protein nodes")
    print(f"GO vocabulary size: {dataset.go_vocab_size}")
    print(f"InterPro vocabulary size: {dataset.ipr_vocab_size}")
    print(f"Number of training proteins: {dataset.train_mask.sum().item()}")
    print(f"Number of validation proteins: {dataset.val_mask.sum().item()}")
    print(f"Number of test proteins: {dataset.test_mask.sum().item()}")

    # Instantiate model with config values
    model = HeteroProteinGNN(
        hidden_channels=config["model"]["hidden_channels"],
        out_channels=dataset.go_vocab_size,
    )
    model = model.to(device)
    if config["trainer"].get("compile", False):
        model = torch.compile(model)
        print("Model compiled with torch.compile()")
    else:
        print("Model not compiled; running in standard mode.")
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["lr"])

    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["scheduler"]["step_size"],
        gamma=config["scheduler"]["gamma"],
    )

    for epoch in range(1, config["trainer"]["epochs"] + 1):
        loss = train_epoch(model, optimizer, train_loader, dataset, h5_path, device)
        print(f"Epoch {epoch} - Train loss: {loss:.4f}")
        scheduler.step()  # Step scheduler if needed

        # Add validation logic here if desired

    # Validation after training
    val_loss = validation(model, val_loader, dataset, h5_path, device)
    print(f"Validation loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
