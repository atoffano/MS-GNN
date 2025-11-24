"""Model inference and interpretability utilities."""

import torch
import yaml
import logging
import argparse
import go3
import tqdm
from typing import Optional, Dict
from torch_geometric.explain import Explainer, CaptumExplainer
from torch_geometric.loader import NeighborLoader

from src.data.dataloading import SwissProtDataset, make_batch_transform
from src.models.gnn_model import ProteinGNN
from src.utils.constants import SUPPORTED_CAPTUM_METHODS, GO_OBO_PATH
from src.utils.visualize import (
    plot_systemic_explanation,
    plot_protein_explanation,
    plot_protein_explanation_msa,
    plot_systemic_attention,
    plot_protein_attention,
    plot_protein_attention_msa,
    ensure_structure,
    analyze_attention_captum_correlation,
    build_plot_context,
)
from src.utils.structure_renderer import export_captum_3d, export_layer_attention_3d
from src.utils.helpers import timeit

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model_and_config(model_path: str, device: torch.device):
    """Load pretrained model, config, and dataset."""
    with open(f"{model_path}/cfg.yaml", "r") as f:
        config = yaml.safe_load(f)

    logger.info("Loading dataset...")
    dataset = SwissProtDataset(config)

    logger.info("Loading model...")
    model = ProteinGNN(config, dataset)
    state_dict = torch.load(
        f"{model_path}/best_model.pth", map_location=device, weights_only=False
    )
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    logger.info(f"Model loaded and moved to {device}")
    return config, model, dataset


class GOTermMapper:
    """Manages GO term to index mapping and ontology operations."""

    def __init__(self, dataset, obo_path: Optional[str] = GO_OBO_PATH):
        self.dataset = dataset
        self.mapping = self._build_mapping(dataset)
        self.idx_to_go = {idx: term for term, idx in self.mapping.items()}
        go3.load_go_terms(str(obo_path))

    @staticmethod
    def _build_mapping(dataset):
        """Build complete mapping from GO term IDs to indices."""
        go_term_mapping = {}
        for vocab_info in dataset.go_vocab_info.values():
            go_term_mapping.update(vocab_info["go_to_idx"])
        logger.info(f"Built GO term mapping with {len(go_term_mapping)} terms")
        return go_term_mapping

    def get_index(self, go_term: str) -> Optional[int]:
        """Get index for a GO term."""
        return self.mapping.get(go_term)

    def get_name(self, go_term: str) -> str:
        """Get human-readable name for a GO term."""
        for vocab_info in self.dataset.go_vocab_info.values():
            if go_term in vocab_info.get("idx_to_name", {}):
                return vocab_info["idx_to_name"][go_term]
        return go_term

    def validate_terms(self, go_terms: list[str]) -> list[tuple[str, int]]:
        """Validate GO terms and return list of (term, index) tuples."""
        valid_terms = []
        for go_term in go_terms:
            idx = self.get_index(go_term)
            if idx is not None:
                valid_terms.append((go_term, idx))
            else:
                logger.warning(f"GO term {go_term} not found in vocabulary")
        return valid_terms

    def get_leaf_terms_from_predictions(
        self, predictions: torch.Tensor, threshold: float = 0.5
    ) -> list[tuple[str, int]]:
        """Get leaf GO terms from model predictions above threshold."""
        predicted_indices = (predictions > threshold).nonzero(as_tuple=True)[0]
        logger.info(f"Found {len(predicted_indices)} predicted terms above {threshold}")

        predicted_terms = [
            self.idx_to_go[idx.item()]
            for idx in predicted_indices
            if idx.item() in self.idx_to_go
        ]

        leaf_terms = []
        for go_term in predicted_terms:
            try:
                term = go3.get_term_by_id(go_term)
                if not term.children:
                    leaf_terms.append((go_term, self.mapping[go_term]))
            except ValueError:
                pass

        logger.info(f"Identified {len(leaf_terms)} leaf terms from predictions")
        return leaf_terms

    def get_term_with_ancestors_target(
        self, leaf_term_id: str, output_size: int
    ) -> torch.Tensor:
        """Create target tensor with leaf term and all its ancestors set to 1."""
        target = torch.zeros(output_size, dtype=torch.long)

        try:
            term = go3.get_term_by_id(leaf_term_id)
            ancestors = go3.ancestors(leaf_term_id)

            for term_id in [leaf_term_id] + list(ancestors):
                idx = self.mapping.get(term_id)
                if idx is not None:
                    target[idx] = 1

            num_active = target.sum().item()
            logger.info(
                f"Created target for {leaf_term_id} ({term.name}): "
                f"{num_active} active terms"
            )
        except ValueError as e:
            logger.error(f"Error creating target for {leaf_term_id}: {e}")

        return target


class ExplanationGenerator:
    """Generates model explanations using Captum."""

    def __init__(self, model, device: torch.device, captum_method: str):
        self.model = model
        self.device = device
        self.captum_method = captum_method

    @timeit
    def generate(
        self,
        batch,
        explanation_type: str = "model",
        target: Optional[torch.Tensor] = None,
    ):
        """Generate explanations for the batch."""
        batch = batch.to(self.device)

        explainer = Explainer(
            self.model,
            algorithm=CaptumExplainer(self.captum_method),
            explanation_type=explanation_type,
            node_mask_type=None,
            edge_mask_type="object",
            model_config=dict(
                mode="binary_classification",
                task_level="node",
                return_type="probs",
            ),
        )

        logger.info(
            f"Generating {explanation_type} explanations with {self.captum_method}..."
        )

        kwargs = {"batch": batch}
        if explanation_type == "phenomenon":
            if target is None:
                raise ValueError("target required for phenomenon explanations")
            kwargs.update({"target": target, "index": None})
            logger.info(
                f"Computing attributions with {target.sum().item()} active terms"
            )

        attributions = explainer(batch.x_dict, batch.edge_index_dict, **kwargs)
        # Apply absolute value to all returned masks
        for et in attributions.edge_types:
            attributions[et].edge_mask = attributions[et].edge_mask.abs()
        return attributions


class ExplanationExporter:
    """Exports explanations to various formats."""

    def __init__(self, output_dir: str, dataset, go_mapper: GOTermMapper):
        self.output_dir = output_dir
        self.dataset = dataset
        self.go_mapper = go_mapper
        self.structure_cache: Dict[str, str] = {}

    def _ensure_cache(self, batch):
        """Lazy-load structure cache when first needed."""
        if self.structure_cache:
            return

        logger.info("Loading protein structures...")
        protein_ids = batch["protein"].n_id.detach().cpu().tolist()
        context = build_plot_context(self.output_dir, self.dataset, batch)

        for protein_id in protein_ids:
            uniprot_id = self.dataset.idx_to_protein[protein_id]
            pdb_id = (
                self.dataset.pid_mapping.get(uniprot_id, uniprot_id)
                if self.dataset.uses_entryid
                else uniprot_id
            )
            try:
                self.structure_cache[uniprot_id] = ensure_structure(
                    pdb_id, self.output_dir
                )
            except FileNotFoundError as e:
                logger.warning(f"Could not load structure for {uniprot_id}: {e}")

    def export_global(self, batch, hetero_explanation, attentions=None):
        """Export global model explanations."""
        self._ensure_cache(batch)
        logger.info("Plotting global explanations...")

        plot_systemic_explanation(self.output_dir, hetero_explanation, self.dataset)
        plot_protein_explanation(self.output_dir, hetero_explanation, self.dataset)
        plot_protein_explanation_msa(
            self.output_dir, hetero_explanation, self.dataset, batch
        )
        export_captum_3d(
            self.output_dir,
            self.dataset,
            batch,
            hetero_explanation,
            structure_cache=self.structure_cache,
        )

        if attentions:
            self._export_attention(batch, attentions)
            analyze_attention_captum_correlation(
                self.output_dir, self.dataset, batch, attentions, hetero_explanation
            )

    def export_go_term(self, batch, hetero_explanation, go_term: str, attentions=None):
        """Export GO term-specific explanation."""
        self._ensure_cache(batch)
        logger.info(f"Plotting explanations for {go_term}...")
        go_name = self.go_mapper.get_name(go_term)
        title_suffix = f"GO: {go_name}"

        plot_systemic_explanation(
            self.output_dir,
            hetero_explanation,
            self.dataset,
            title_suffix=title_suffix,
            go_term=go_term,
        )
        plot_protein_explanation(
            self.output_dir,
            hetero_explanation,
            self.dataset,
            title_suffix=title_suffix,
            go_term=go_term,
        )
        plot_protein_explanation_msa(
            self.output_dir,
            hetero_explanation,
            self.dataset,
            batch,
            title_suffix=title_suffix,
            go_term=go_term,
        )
        export_captum_3d(
            self.output_dir,
            self.dataset,
            batch,
            hetero_explanation,
            go_term=go_term,
            structure_cache=self.structure_cache,
        )

        if attentions:
            self._export_attention(batch, attentions, go_term)

    def _export_attention(self, batch, attentions, go_term=None):
        """Export attention visualizations."""
        for idx, layer_attention in enumerate(attentions, start=1):
            if layer_attention is None:
                continue

            if go_term is None:
                plot_systemic_attention(
                    self.output_dir, layer_attention, self.dataset, batch, idx
                )

            plot_protein_attention(
                self.output_dir, layer_attention, self.dataset, batch, idx, go_term
            )
            plot_protein_attention_msa(
                self.output_dir, layer_attention, self.dataset, batch, idx, go_term
            )
            export_layer_attention_3d(
                self.output_dir,
                self.dataset,
                batch,
                idx,
                layer_attention,
                structure_cache=self.structure_cache,
                go_term=go_term,
            )


def create_data_loader(dataset, config, protein_names: list[str]) -> NeighborLoader:
    """Create data loader for specified proteins."""
    proteins = []
    for protein in protein_names:
        if "_" in protein:
            if dataset.uses_entryid:
                proteins.append(protein)
        else:
            if not dataset.uses_entryid:
                proteins.append(protein)
            else:
                proteins.append(dataset.rev_pid_mapping[protein])

    protein_ids = []
    for protein in proteins:
        if protein in dataset.protein_to_idx:
            protein_ids.append(dataset.protein_to_idx[protein])
        else:
            logger.warning(f"Protein {protein} not found in dataset, skipping.")

    mask = torch.zeros(len(dataset.proteins), dtype=torch.bool)
    mask[protein_ids] = True

    # Parse sampled_edges from config
    num_neighbors = {}
    for key, val in config["model"]["sampled_edges"].items():
        parts = key.split("__")
        if len(parts) == 3:
            num_neighbors[tuple(parts)] = [val]

    return NeighborLoader(
        dataset.data,
        num_neighbors=num_neighbors,
        batch_size=1,
        input_nodes=("protein", mask),
        transform=make_batch_transform(dataset, mode="predict", return_sequences=True),
        shuffle=False,
        num_workers=0,
    )


def process_batch(batch, model, device, exporter, generator, go_mapper, go_terms):
    """Process a single batch: run inference and generate explanations."""
    batch = batch.to(device)
    preds, attn = model(
        batch.x_dict, batch.edge_index_dict, batch=batch, return_attention_weights=True
    )

    # Global explanations
    logger.info("=" * 60)
    logger.info("Generating global model explanations...")
    logger.info("=" * 60)
    global_explanation = generator.generate(batch, "model")
    exporter.export_global(batch, global_explanation, attn)

    # GO term-specific explanations
    if not go_terms:
        return

    logger.info("=" * 60)
    logger.info("Generating GO term-specific explanations...")
    logger.info("=" * 60)

    # Determine leaf terms
    if go_terms == ["predicted"]:
        leaf_terms = go_mapper.get_leaf_terms_from_predictions(preds[0])
        if not leaf_terms:
            logger.warning("No leaf terms found in predictions")
            return
    else:
        leaf_terms = go_mapper.validate_terms(go_terms)

    # Generate per-term explanations
    for leaf_term_id, _ in tqdm.tqdm(leaf_terms, desc="Attribution per GO leaf term"):
        go_name = go_mapper.get_name(leaf_term_id)
        logger.info(f"\n--- Explaining: {leaf_term_id} ({go_name}) and ancestors ---")

        target = go_mapper.get_term_with_ancestors_target(
            leaf_term_id, preds.size(1)
        ).to(device)

        go_explanation = generator.generate(batch, "phenomenon", target)
        exporter.export_go_term(batch, go_explanation, leaf_term_id, attn)


def main():
    """Main entry point for explanation generation."""
    parser = argparse.ArgumentParser(
        description="Generate explanations for protein function prediction model"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--proteins", nargs="*", default=None)
    parser.add_argument("--go_terms", nargs="*", default=None)
    parser.add_argument(
        "--captum_method",
        type=str,
        default="IntegratedGradients",
        choices=SUPPORTED_CAPTUM_METHODS,
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model and dataset
    config, model, dataset = load_model_and_config(args.model_path, device)

    # Initialize components
    go_mapper = GOTermMapper(dataset, obo_path=GO_OBO_PATH)
    generator = ExplanationGenerator(model, device, args.captum_method)
    loader = create_data_loader(dataset, config, args.proteins)

    # Process each batch
    for batch in loader:
        exporter = ExplanationExporter(args.model_path, dataset, go_mapper)
        process_batch(
            batch, model, device, exporter, generator, go_mapper, args.go_terms
        )

    logger.info(
        f"Explanation generation completed! Results saved to: {args.model_path}"
    )


if __name__ == "__main__":
    main()
