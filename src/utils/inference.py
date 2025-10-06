import torch
import yaml
import logging
import argparse
from torch_geometric.explain import (
    Explainer,
    CaptumExplainer,
)
from torch_geometric.loader import NeighborLoader
from src.data.dataloading import SwissProtDataset, make_batch_transform
from src.models.gnn_model import ProteinGNN
from src.utils.visualize import (
    plot_systemic_explanation,
    plot_protein_explanation,
    plot_systemic_attention,
    plot_protein_attention,
)
from src.utils.helpers import timeit
import pickle

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def zscore(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.float()
    if tensor.numel() == 0:
        return tensor
    mean = tensor.mean()
    std = tensor.std(unbiased=False)
    if std.item() < 1e-12:
        return torch.zeros_like(tensor)
    return (tensor - mean) / std


def load_model_and_config(model_path, device):
    """Load the pretrained model and configuration."""
    # Load config
    with open(f"{model_path}/cfg.yaml", "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loading dataset..")
    dataset = SwissProtDataset(config)
    logger.info(f"Loaded dataset.\n Loading model..")
    model = ProteinGNN(config, dataset)
    state_dict = torch.load(
        f"{model_path}/model.pth", map_location=device, weights_only=True
    )
    for key in list(state_dict.keys()):
        if key.startswith("_orig_mod."):
            state_dict[key.replace("_orig_mod.", "")] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded model from {model_path}")
    logger.info(f"Model moved to device: {device}")

    return config, model, dataset


def get_loader(dataset, protein_names=None):
    """Get a batch for explanation generation."""
    # Create mask for specific proteins
    protein_indices = [dataset.protein_to_idx[pid] for pid in protein_names]
    mask = torch.zeros(len(dataset.proteins), dtype=torch.bool)
    mask[protein_indices] = True

    loader = NeighborLoader(
        dataset.data,
        num_neighbors={("protein", "aligned_with", "protein"): [-1]},
        batch_size=1,
        input_nodes=("protein", mask),
        transform=make_batch_transform(dataset, mode="predict"),
        shuffle=False,
        num_workers=0,
    )

    return loader


@timeit
def generate_explanations(model, batch, device):
    """Generate explanations using the model and batch."""
    batch = batch.to(device)

    # Create explainer
    explainer = Explainer(
        model,
        algorithm=CaptumExplainer("IntegratedGradients"),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        ),
    )

    # Generate explanations
    logger.info("Generating explanations...")
    hetero_explanation = explainer(
        batch.x_dict,
        batch.edge_index_dict,
        batch=batch,
        target=None,
        index=torch.arange(batch["protein"].batch_size),
    )

    logger.info(
        f"Generated explanations in {hetero_explanation.available_explanations}"
    )

    return hetero_explanation


def main():
    parser = argparse.ArgumentParser(
        description="Generate explanations for pretrained protein function prediction model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to pretrained model weights (.pth file)",
    )
    parser.add_argument(
        "--proteins",
        nargs="*",
        default=None,
        help="Specific protein indices to explain (if None, uses test proteins)",
    )

    args = parser.parse_args()

    # load /home/atoffano/PFP_layer/results/D1/20251002_001234_D1_2layerGNN/explanations/hetero_explanation.pkl and /home/atoffano/PFP_layer/results/D1/20251002_001234_D1_2layerGNN/explanations/dataset.pkl
    # with open(f"{args.model_path}/explanations/hetero_explanation.pkl", "rb") as f:
    #     hetero_explanation = pickle.load(f)
    # with open(f"{args.model_path}/explanations/dataset.pkl", "rb") as f:
    #     dataset = pickle.load(f)

    # plot_systemic_explanation(args.model_path, hetero_explanation, dataset)
    # plot_protein_explanation(args.model_path, hetero_explanation, dataset)

    # # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model and config
    config, model, dataset = load_model_and_config(args.model_path, device)
    loader = get_loader(dataset, args.proteins)

    for batch in loader:
        batch = batch.to(device)
        # Get attention scores
        _, attn = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch=batch,
            return_attention_weights=True,
        )
        for idx, layer_attention in enumerate(attn, start=1):
            plot_systemic_attention(
                args.model_path, layer_attention, dataset, batch, idx
            )
            plot_protein_attention(
                args.model_path, layer_attention, dataset, batch, idx
            )

        logger.info(f"Generating explanations..")
        hetero_explanation = generate_explanations(model, batch, device)

        # # Save hetero_explanation and dataset as pkl
        # os.makedirs(f"{args.model_path}/explanations", exist_ok=True)
        # with open(f"{args.model_path}/explanations/hetero_explanation.pkl", "wb") as f:
        #     pickle.dump(hetero_explanation, f)
        # with open(f"{args.model_path}/explanations/dataset.pkl", "wb") as f:
        #     pickle.dump(dataset, f)

        # Plot protein-protein explanation
        logger.info(f"Plotting explanations..")
        plot_systemic_explanation(args.model_path, hetero_explanation, dataset)
        plot_protein_explanation(args.model_path, hetero_explanation, dataset)

    logger.info(
        f"Explanation generation completed! Results saved to: {args.model_path}"
    )


if __name__ == "__main__":
    main()
