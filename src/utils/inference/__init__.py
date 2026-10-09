"""MS-GNN Inference Pipeline.

Provides modular, CLI-callable tools for predicting Gene Ontology annotations
for arbitrary proteins using pre-trained MS-GNN models. The pipeline downloads
required data programmatically, builds minimal inference graphs on-the-fly,
and runs pre-trained models without requiring the full SwissProt corpus.

Modules:
    resolve_ids         — UniProt ID resolution & sequence fetching
    preprocess_target   — Target protein preprocessing (ESM, structure, graph)
    prepare_neighbors   — Neighbor identification & .pt preparation
    inference_dataset   — Lightweight dataset class for inference
    batch_features      — Batch feature assembly for inference
    output              — Prediction formatting & saving
    infer               — Main CLI entry point & orchestrator
"""

from src.utils.inference.resolve_ids import resolve_proteins, ProteinInfo
from src.utils.inference.inference_dataset import InferenceDataset

__all__ = [
    "resolve_proteins",
    "ProteinInfo",
    "InferenceDataset",
]
