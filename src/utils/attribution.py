"""Model inference and interpretability utilities."""

import torch
import yaml
import logging
import argparse
import go3
import tqdm
import pickle
import os
from typing import Optional, Dict
from torch_geometric.explain import Explainer, CaptumExplainer
from torch_geometric.loader import NeighborLoader

from src.data.dataloading import SwissProtDataset, make_batch_transform
from src.models.gnn_model import ProteinGNN
from src.utils.constants import (
    SUPPORTED_CAPTUM_METHODS,
    GO_OBO_PATH,
    GO_VOCAB,
    INTERPRO_VOCAB,
)
from src.utils.visualize import (
    plot_systemic_explanation,
    plot_protein_explanation,
    # plot_protein_explanation_msa,
    plot_systemic_attention,
    plot_protein_attention,
    # plot_protein_attention_msa,
    # plot_attn_seed_vs_neighbor_scatter,
    plot_attn_stringdb_vs_aligned_scatter,
    ensure_structure,
    analyze_attention_captum_correlation,
    # perform_msa_from_batch,
)
from src.utils.structure_renderer import export_captum_3d, export_layer_attention_3d
from src.utils.helpers import timeit

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model_and_config(
    model_path: str,
    device: torch.device,
    checkpoint_name: str,
):
    """Load pretrained model, config, and dataset."""
    checkpoint_path = f"{model_path}/checkpoints/{checkpoint_name}"
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "config" in checkpoint:
        config = checkpoint["config"]
        logger.info("Using configuration embedded in checkpoint")
    else:
        with open(f"{model_path}/cfg.yaml", "r") as f:
            config = yaml.safe_load(f)
        logger.info("Using configuration from cfg.yaml")

    # If subontology is a list, convert to str for consistency
    if isinstance(config.get("data", {}).get("subontology"), list):
        config["data"]["subontology"] = config["data"]["subontology"][0]

    logger.info("Loading dataset...")
    dataset = SwissProtDataset(config)

    logger.info("Loading model...")
    model = ProteinGNN(config, dataset)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
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

    def _get_explainer(
        self, explanation_type: str, attribution_level: str
    ) -> Explainer:
        if attribution_level == "node":
            return Explainer(
                self.model,
                algorithm=CaptumExplainer(self.captum_method),
                explanation_type=explanation_type,
                node_mask_type="attributes",
                edge_mask_type=None,
                model_config=dict(
                    mode="binary_classification",
                    task_level="node",
                    return_type="probs",
                ),
            )

        return Explainer(
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

    @timeit
    def generate(
        self,
        batch,
        explanation_type: str = "model",
        target: Optional[torch.Tensor] = None,
        attribution_level: str = "edge",
    ):
        """Generate explanations for the batch."""
        batch = batch.to(self.device)
        explainer = self._get_explainer(explanation_type, attribution_level)

        logger.info(
            f"Generating {explanation_type} {attribution_level} explanations with {self.captum_method}..."
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

        if attribution_level == "edge":
            for et in attributions.edge_types:
                attributions[et].edge_mask = attributions[et].edge_mask.abs()
        elif (
            "protein" in attributions.node_types
            and hasattr(attributions["protein"], "node_mask")
            and attributions["protein"].node_mask is not None
        ):
            attributions["protein"].node_mask = attributions["protein"].node_mask.abs()

        return attributions


class ExplanationExporter:
    """Exports explanations to various formats."""

    def __init__(
        self,
        output_dir: str,
        dataset,
        go_mapper: GOTermMapper,
        plot_neighbors: bool = False,
        save_scores: bool = False,
    ):
        self.output_dir = output_dir
        self.dataset = dataset
        self.go_mapper = go_mapper
        self.plot_neighbors = plot_neighbors
        self.save_scores = save_scores
        self.structure_cache: Dict[str, str] = {}
        self.aligned_seqs = None
        self.go_idx_to_term, self.ipr_idx_to_term = self._load_feature_vocab_maps()
        self.attr_abs_threshold = 1e-8  # skip numerical noise in dense masks

        # Feature families to export from protein node_mask
        # Allowed values: "ipr", "go"
        # self.save = ["ipr", "go"]
        self.save = ["go"]

        self.feature_names, self.feature_keep_mask = self._build_feature_name_lookup()

    def _build_feature_name_lookup(self):
        """Precompute feature_idx -> term name and export mask for protein.x layout."""
        use_ipr = self.dataset.config["model"]["interpro"]
        use_go = self.dataset.config["model"]["go_neighbors"]
        save_set = set(self.save)

        names = []
        keep = []

        if use_ipr:
            names.extend(
                self.ipr_idx_to_term.get(i, f"IPR_IDX_{i}")
                for i in range(self.dataset.ipr_vocab_size)
            )
            keep.extend(["ipr" in save_set] * self.dataset.ipr_vocab_size)

        if use_go:
            names.extend(
                self.go_idx_to_term.get(i, f"GO_IDX_{i}")
                for i in range(self.dataset.go_vocab_size)
            )
            keep.extend(["go" in save_set] * self.dataset.go_vocab_size)

        if not names:
            # fallback for zero-input mode
            total = self.dataset.ipr_vocab_size + self.dataset.go_vocab_size
            names = [f"FEATURE_IDX_{i}" for i in range(total)]
            keep = [False] * total

        return names, torch.tensor(keep, dtype=torch.bool)

    def _load_feature_vocab_maps(self):
        """Load idx->term maps for GO and InterPro."""
        go_idx_to_term = {}
        ipr_idx_to_term = {}

        try:
            with open(GO_VOCAB, "rb") as f:
                go_vocab = pickle.load(f)
            sub = self.dataset.subontology
            go_to_idx = go_vocab.get(sub, {}).get("go_to_idx", {})
            go_idx_to_term = {idx: term for term, idx in go_to_idx.items()}
        except Exception as e:
            logger.warning(f"Could not load GO vocab mapping: {e}")

        try:
            with open(INTERPRO_VOCAB, "rb") as f:
                ipr_vocab = pickle.load(f)
            ipr_to_idx = ipr_vocab.get("ipr_to_idx", {})
            ipr_idx_to_term = {idx: term for term, idx in ipr_to_idx.items()}
        except Exception as e:
            logger.warning(f"Could not load InterPro vocab mapping: {e}")

        return go_idx_to_term, ipr_idx_to_term

    def _feature_idx_to_term(self, feature_idx: int) -> str:
        """Convert protein feature position to InterPro/GO term based on model input layout."""
        use_ipr = self.dataset.config["model"]["interpro"]
        use_go = self.dataset.config["model"]["go_neighbors"]

        if use_ipr and use_go:
            if feature_idx < self.dataset.ipr_vocab_size:
                return self.ipr_idx_to_term.get(feature_idx, f"IPR_IDX_{feature_idx}")
            go_idx = feature_idx - self.dataset.ipr_vocab_size
            return self.go_idx_to_term.get(go_idx, f"GO_IDX_{go_idx}")

        if use_ipr and not use_go:
            return self.ipr_idx_to_term.get(feature_idx, f"IPR_IDX_{feature_idx}")

        if (not use_ipr) and use_go:
            return self.go_idx_to_term.get(feature_idx, f"GO_IDX_{feature_idx}")

        return f"FEATURE_IDX_{feature_idx}"

    # def _ensure_msa(self, batch):
    #     """Ensure MSA is computed for the current batch."""
    #     if self.aligned_seqs is None:
    #         self.aligned_seqs = perform_msa_from_batch(batch)

    def _ensure_cache(self, batch):
        """Lazy-load structure cache when first needed."""
        if self.structure_cache:
            return

        logger.info("Loading protein structures...")
        protein_ids = batch["protein"].n_id.detach().cpu().tolist()

        for protein_id in tqdm.tqdm(protein_ids, desc="Loading structures"):
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

    def _save_captum_attributions_pkl(
        self,
        batch,
        edge_attributions=None,
        node_attributions=None,
        go_term=None,
    ):
        """Save edge and protein feature attributions as a single pickle dict."""
        pkl_path = os.path.join(self.output_dir, "captum_attributions.pkl")
        seed_idx = batch["protein"].n_id[0].item()
        seed_name = self.dataset.idx_to_protein[seed_idx]
        term_key = go_term if go_term else "global"

        n_ids = batch["protein"].n_id.detach().cpu().tolist()
        local_idx_to_name = {
            i: self.dataset.idx_to_protein[nid] for i, nid in enumerate(n_ids)
        }

        payload = {"edge": {}, "node": {}}

        if edge_attributions is not None:
            # Restrict to protein-protein edges for compact/fast serialization
            for rel in edge_attributions.edge_types:
                if hasattr(edge_attributions[rel], "edge_mask"):
                    edge_index = edge_attributions[rel].edge_index.detach().cpu()
                    edge_mask = edge_attributions[rel].edge_mask.detach().cpu()
                    u = [
                        local_idx_to_name.get(i, str(i)) for i in edge_index[0].tolist()
                    ]
                    v = [
                        local_idx_to_name.get(i, str(i)) for i in edge_index[1].tolist()
                    ]
                    payload["edge"][rel] = {
                        "edge_index": (u, v),
                        "scores": edge_mask.tolist(),
                    }

        if (
            node_attributions is not None
            and "protein" in node_attributions.node_types
            and hasattr(node_attributions["protein"], "node_mask")
            and node_attributions["protein"].node_mask is not None
        ):
            protein_mask = node_attributions["protein"].node_mask.detach().abs().cpu()

            # Keep only selected feature families from self.save
            keep_mask = self.feature_keep_mask
            if keep_mask.numel() < protein_mask.shape[1]:
                pad = torch.zeros(
                    protein_mask.shape[1] - keep_mask.numel(), dtype=torch.bool
                )
                keep_mask = torch.cat([keep_mask, pad], dim=0)
            elif keep_mask.numel() > protein_mask.shape[1]:
                keep_mask = keep_mask[: protein_mask.shape[1]]

            selected = (protein_mask > self.attr_abs_threshold) & keep_mask.unsqueeze(0)
            nz = torch.nonzero(selected, as_tuple=False)

            if nz.numel() > 0:
                node_idx = nz[:, 0]
                feat_idx = nz[:, 1]
                values = protein_mask[node_idx, feat_idx]

                # Group contiguous rows by node after sorting
                order = torch.argsort(node_idx)
                node_idx = node_idx[order]
                feat_idx = feat_idx[order]
                values = values[order]

                unique_nodes, counts = torch.unique_consecutive(
                    node_idx, return_counts=True
                )
                offsets = torch.cumsum(counts, dim=0)
                start = 0

                for i, local_node_idx in enumerate(unique_nodes.tolist()):
                    end = offsets[i].item()
                    node_name = local_idx_to_name.get(
                        local_node_idx, str(local_node_idx)
                    )

                    f_idx = feat_idx[start:end].tolist()
                    vals = values[start:end].tolist()
                    terms = [
                        (
                            self.feature_names[j]
                            if j < len(self.feature_names)
                            else f"FEATURE_IDX_{j}"
                        )
                        for j in f_idx
                    ]

                    payload["node"][node_name] = dict(zip(terms, vals))
                    start = end

        full_data = {}
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, "rb") as f:
                    full_data = pickle.load(f)
            except Exception:
                full_data = {}

        if seed_name not in full_data:
            full_data[seed_name] = {}
        full_data[seed_name][term_key] = payload

        try:
            with open(pkl_path, "wb") as f:
                pickle.dump(full_data, f)
            logger.info(
                f"Updated Captum attributions for {seed_name} ({term_key}) in {pkl_path}"
            )
        except Exception as e:
            logger.error(f"Failed to save Captum attributions: {e}")

    def export_global(
        self, batch, hetero_explanation, attentions=None, node_attributions=None
    ):
        """Export global model explanations."""
        logger.info("Plotting global explanations...")

        if self.save_scores:
            self._save_captum_attributions_pkl(
                batch,
                edge_attributions=hetero_explanation,
                node_attributions=node_attributions,
                go_term=None,
            )

        plot_systemic_explanation(self.output_dir, hetero_explanation, self.dataset)
        plot_protein_explanation(
            self.output_dir,
            hetero_explanation,
            self.dataset,
            plot_neighbors=self.plot_neighbors,
        )
        export_captum_3d(
            self.output_dir,
            self.dataset,
            batch,
            hetero_explanation,
            structure_cache=self.structure_cache,
            plot_neighbors=self.plot_neighbors,
        )

        if attentions:
            self._export_attention(batch, attentions)
            # analyze_attention_captum_correlation(
            #     self.output_dir, self.dataset, batch, attentions, hetero_explanation
            # )

    def export_go_term(
        self,
        batch,
        hetero_explanation,
        go_term: str,
        attentions=None,
        node_attributions=None,
    ):
        """Export GO term-specific explanation."""
        logger.info(f"Plotting explanations for {go_term}...")
        go_name = self.go_mapper.get_name(go_term)
        title_suffix = f"GO: {go_name}"

        if attentions and self.save_scores:
            self._save_attention_pkl(batch, attentions, go_term)

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
            plot_neighbors=self.plot_neighbors,
        )
        export_captum_3d(
            self.output_dir,
            self.dataset,
            batch,
            hetero_explanation,
            go_term=go_term,
            structure_cache=self.structure_cache,
            plot_neighbors=self.plot_neighbors,
        )

        if attentions:
            self._export_attention(batch, attentions, go_term)

        if self.save_scores:
            self._save_captum_attributions_pkl(
                batch,
                edge_attributions=hetero_explanation,
                node_attributions=node_attributions,
                go_term=go_term,
            )

    def _save_attention_pkl(self, batch, attentions, go_term=None):
        """Save attention scores to a single pickle file."""
        pkl_path = os.path.join(self.output_dir, "attention_scores.pkl")

        seed_idx = batch["protein"].n_id[0].item()
        seed_name = self.dataset.idx_to_protein[seed_idx]
        term_key = go_term if go_term else "global"

        # Map local batch indices to protein names
        n_ids = batch["protein"].n_id.detach().cpu().tolist()
        local_idx_to_name = {
            i: self.dataset.idx_to_protein[nid] for i, nid in enumerate(n_ids)
        }

        # Prepare layer-wise data
        layers_data = []
        target_keys = [
            ("protein", "aligned_with", "protein"),
            ("protein", "stringdb", "protein"),
            #     ("protein", "rev_belongs_to", "aa"),
            #     ("aa", "belongs_to", "protein"),
            #     ("aa", "close_to", "aa"),
        ]

        if attentions:
            layer_attn = attentions[1]  # Use layer 2.
            layer_dict = {}
            if layer_attn:
                for key in batch.edge_index_dict.keys():
                    if key in target_keys:
                        edge_index, scores = layer_attn[key]
                        # Convert indices to names
                        u_indices = edge_index[0].detach().cpu().tolist()
                        v_indices = edge_index[1].detach().cpu().tolist()

                        u_names = [
                            local_idx_to_name.get(idx, str(idx)) for idx in u_indices
                        ]
                        v_names = [
                            local_idx_to_name.get(idx, str(idx)) for idx in v_indices
                        ]

                        layer_dict[key] = {
                            "edge_index": (u_names, v_names),
                            "scores": scores.detach().cpu().tolist(),
                        }
            layers_data.append(layer_dict)

        # Load existing pickle if available (to append/update)
        full_data = {}
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, "rb") as f:
                    full_data = pickle.load(f)
            except Exception:
                full_data = {}

        if seed_name not in full_data:
            full_data[seed_name] = {}

        full_data[seed_name][term_key] = layers_data

        try:
            with open(pkl_path, "wb") as f:
                pickle.dump(full_data, f)
            logger.info(
                f"Updated attention scores for {seed_name} ({term_key}) in {pkl_path}"
            )
        except Exception as e:
            logger.error(f"Failed to save attention scores: {e}")

    def _export_attention(self, batch, attentions, go_term=None):
        """Export attention visualizations."""
        if go_term is None and self.save_scores:
            self._save_attention_pkl(batch, attentions)

        for idx, layer_attention in enumerate(attentions, start=1):
            if layer_attention is None:
                continue

            if go_term is None:
                # Systemic level neighbor attribution
                plot_systemic_attention(
                    self.output_dir, layer_attention, self.dataset, batch, idx
                )

            # Protein level residue attribution
            plot_protein_attention(
                self.output_dir,
                layer_attention,
                self.dataset,
                batch,
                idx,
                go_term,
                plot_neighbors=self.plot_neighbors,
            )

            if self.plot_neighbors:
                # plot_attn_seed_vs_neighbor_scatter(
                #     self.output_dir,
                #     layer_attention,
                #     self.dataset,
                #     batch,
                #     idx,
                #     go_term,
                #     aligned_seqs=self.aligned_seqs,
                # )
                plot_attn_stringdb_vs_aligned_scatter(
                    self.output_dir,
                    layer_attention,
                    self.dataset,
                    batch,
                    idx,
                    go_term,
                )

            # 3D structure visualization with attention scores mapped onto residues
            export_layer_attention_3d(
                self.output_dir,
                self.dataset,
                batch,
                idx,
                layer_attention,
                structure_cache=self.structure_cache,
                go_term=go_term,
                plot_neighbors=self.plot_neighbors,
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


def process_batch(
    batch, model, device, exporter, generator, go_mapper, go_terms, threshold=0.5
):
    """Process a single batch: run inference and generate explanations."""
    batch = batch.to(device)
    preds, attn = model(
        batch.x_dict, batch.edge_index_dict, batch=batch, return_attention_weights=True
    )

    # Global explanations
    logger.info("=" * 60)
    logger.info("Generating global model explanations...")
    logger.info("=" * 60)
    global_explanation = generator.generate(batch, "model", attribution_level="edge")
    global_node_attributions = (
        generator.generate(batch, "model", attribution_level="node")
        if exporter.save_scores
        else None
    )
    exporter.export_global(
        batch,
        global_explanation,
        attn,
        node_attributions=global_node_attributions,
    )

    # GO term-specific explanations
    if not go_terms:
        return

    logger.info("=" * 60)
    logger.info("Generating GO term-specific explanations...")
    logger.info("=" * 60)

    # Determine leaf terms
    if go_terms == ["predicted"]:
        leaf_terms = go_mapper.get_leaf_terms_from_predictions(preds[0], threshold)
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

        go_explanation = generator.generate(
            batch, "phenomenon", target, attribution_level="edge"
        )
        go_node_attributions = (
            generator.generate(batch, "phenomenon", target, attribution_level="node")
            if exporter.save_scores
            else None
        )
        exporter.export_go_term(
            batch,
            go_explanation,
            leaf_term_id,
            attn,
            node_attributions=go_node_attributions,
        )


def main():
    """Main entry point for explanation generation."""
    parser = argparse.ArgumentParser(
        description="Generate explanations for protein function prediction model"
    )
    parser.add_argument("--model_path", type=str, required=True)
    # optional
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint to load (e.g. checkpoint_latest_MFO.pth).",
    )
    parser.add_argument("--proteins", nargs="*", default=None, required=True)
    parser.add_argument("--go_terms", nargs="*", default=None)
    parser.add_argument(
        "--threshold", type=float, default=0.5
    )  # Threshold to consider a GO term as predicted when evaluating a specific go term (should match best tau found during validation)
    parser.add_argument(
        "--captum_method",
        type=str,
        default="IntegratedGradients",
        choices=SUPPORTED_CAPTUM_METHODS,
    )
    parser.add_argument(
        "--plot_neighbors",
        action="store_true",
        help="Generate plots for neighbor proteins",
    )
    parser.add_argument(
        "--save_scores",
        action="store_true",
        help="Save attention scores as pkl object",
    )

    args = parser.parse_args()
    device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load model and dataset
    config, model, dataset = load_model_and_config(
        args.model_path, device, args.checkpoint
    )

    # Initialize components
    go_mapper = GOTermMapper(dataset, obo_path=GO_OBO_PATH)
    generator = ExplanationGenerator(model, device, args.captum_method)
    loader = create_data_loader(dataset, config, args.proteins)

    # Process each batch
    # try:
    for batch in loader:
        exporter = ExplanationExporter(
            args.model_path,
            dataset,
            go_mapper,
            plot_neighbors=args.plot_neighbors,
            save_scores=args.save_scores,
        )
        process_batch(
            batch,
            model,
            device,
            exporter,
            generator,
            go_mapper,
            args.go_terms,
            threshold=args.threshold,
        )
    # except Exception as e:
    #     logger.error(f"Error during explanation generation: {e}")
    logger.info(
        f"Explanation generation completed! Results saved to: {args.model_path}"
    )


if __name__ == "__main__":
    main()
