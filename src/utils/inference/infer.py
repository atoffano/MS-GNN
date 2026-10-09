"""Main CLI entry point and orchestrator for MS-GNN inference.

Usage examples:

    # Basic — single protein, single subontology
    python -m src.utils.inference.infer --proteins P01308 --subontology MFO

    # Multiple proteins and subontologies
    python -m src.utils.inference.infer --proteins P01308 Q9Y6K1 --subontology MFO BPO CCO

    # With threshold and barebones output (leaf terms only)
    python -m src.utils.inference.infer --proteins P01308 --subontology MFO --tau 0.5 --barebones

    # From a file (one UniProt ID per line)
    python -m src.utils.inference.infer --proteins-file queries.txt --subontology MFO BPO CCO

    # Skip preprocessing (assume .pt files already exist)
    python -m src.utils.inference.infer --proteins P01308 --subontology MFO --skip-preprocessing
"""

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import yaml
from torch_geometric.loader import NeighborLoader

from src.models.gnn_model import ProteinGNN
from src.utils.constants import (
    GO_ONTOLOGIES,
    INFERENCE_MODELS_DIR,
    INFERENCE_MODEL_MAP,
    INFERENCE_OUTPUT_DIR,
    PROTEIN_GRAPHS_DIR,
)
from src.utils.inference.batch_features import make_inference_batch_transform
from src.utils.inference.inference_dataset import (
    InferenceDataset,
    get_go_vocab_info,
    get_ipr_vocab_size,
    get_pid_mapping,
)
from src.utils.inference.output import print_summary, save_inference_predictions
from src.utils.inference.prepare_neighbors import identify_neighbors
from src.utils.inference.preprocess_target import preprocess_target
from src.utils.inference.resolve_ids import ProteinInfo, resolve_proteins

logger = logging.getLogger(__name__)


def set_seed(seed: Optional[int] = 53) -> None:
    """Set random seed for reproducibility across Python, NumPy, and PyTorch."""
    if seed is None:
        return
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_model_dir(subontology: str, models_dir: Path = INFERENCE_MODELS_DIR) -> Path:
    """Resolve the model directory for a given subontology.

    Args:
        subontology: GO subontology (MFO, BPO, CCO).
        models_dir: Root directory containing model symlinks/directories, or direct model dir.

    Returns:
        Resolved path to the model directory.

    Raises:
        FileNotFoundError: If the model directory does not exist.
    """
    models_dir = Path(models_dir)
    # Check if models_dir itself is the model directory
    if (models_dir / "cfg.yaml").exists() and (
        (models_dir / "checkpoints" / f"checkpoint_latest_{subontology}.pth").exists()
        or (models_dir / f"checkpoint_latest_{subontology}.pth").exists()
    ):
        return models_dir.resolve()

    model_name = INFERENCE_MODEL_MAP.get(subontology)
    if model_name is None:
        raise ValueError(
            f"No model configured for subontology '{subontology}'. "
            f"Available: {list(INFERENCE_MODEL_MAP.keys())}"
        )

    model_dir = models_dir / model_name
    # Resolve symlinks
    model_dir = model_dir.resolve()
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir} "
            f"(symlink target for {models_dir / model_name})"
        )
    return model_dir


def load_model_config(model_dir: Path) -> dict:
    """Load model configuration from cfg.yaml.

    Args:
        model_dir: Path to model directory.

    Returns:
        Configuration dictionary.
    """
    cfg_path = model_dir / "cfg.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found at {cfg_path}")

    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    # Override results_dir to point to actual model directory
    config["run"]["results_dir"] = str(model_dir)
    return config


def load_edge_normalization(model_dir: Path) -> dict:
    """Load precomputed edge normalization constants.

    Args:
        model_dir: Path to model directory.

    Returns:
        Dict like {"aligned_with": {"mean": ..., "std": ...}, "stringdb": {...}}.
    """
    norm_path = model_dir / "edge_normalization.json"
    if not norm_path.exists():
        logger.warning(
            "edge_normalization.json not found at %s. "
            "Using identity normalization (mean=0, std=1).",
            norm_path,
        )
        return {}

    with open(norm_path, "r") as f:
        norm = json.load(f)

    logger.info("Loaded edge normalization from %s: %s", norm_path, norm)
    return norm


def load_model_for_inference(
    config: dict,
    dataset: Union[InferenceDataset, int, Any],
    model_dir: Path,
    subontology: str,
    device: Optional[torch.device] = None,
) -> ProteinGNN:
    """Load a pre-trained ProteinGNN model for inference.

    Handles torch.compile checkpoint key prefix stripping.

    Args:
        config: Model configuration dictionary.
        dataset: InferenceDataset instance, or integer go_vocab_size.
        model_dir: Path to model directory.
        subontology: GO subontology name.
        device: Optional torch.device to place model on.

    Returns:
        ProteinGNN model in eval mode.
    """
    if isinstance(dataset, int):
        from types import SimpleNamespace
        ds = SimpleNamespace(go_vocab_size=dataset)
    else:
        ds = dataset
    model = ProteinGNN(config, ds)

    ckpt_path = model_dir / "checkpoints" / f"checkpoint_latest_{subontology}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger.info("Loading checkpoint from %s", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state_dict = checkpoint["model_state_dict"]
    # Strip torch.compile prefix if present
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned)

    if device is not None:
        model = model.to(device)

    model.eval()
    logger.info("Model loaded and set to eval mode on %s.", device if device else "CPU")
    return model


def resolve_protein_structure(
    protein_info: ProteinInfo,
    output_dir: Path,
    target_graphs_dir: Path,
) -> Optional[Path]:
    """Resolve or download a 3D structure (.pdb or .cif) for a protein.

    Searches in local directories first, then falls back to downloading
    from AlphaFold / PDB via ensure_structure.
    """
    struct_dir = output_dir / "structures"
    struct_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        struct_dir / f"{protein_info.accession}.pdb",
        struct_dir / f"{protein_info.accession}.cif",
        struct_dir / f"{protein_info.entry_name}.pdb",
        struct_dir / f"{protein_info.entry_name}.cif",
        target_graphs_dir / f"{protein_info.accession}.pdb",
        target_graphs_dir / f"{protein_info.accession}.cif",
        target_graphs_dir / f"{protein_info.entry_name}.pdb",
        target_graphs_dir / f"{protein_info.entry_name}.cif",
    ]
    # Check known model results dirs with structures
    from src.utils.constants import PROJECT_ROOT
    known_dirs = [
        PROJECT_ROOT / "results" / "D1" / "20260214_100017_D1_2024_01_GNN_CONTACT_ATTR_STRINGDB_CUR",
        PROJECT_ROOT / "data" / "swissprot" / "2024_01" / "alphafold_pdb",
    ]
    for kd in known_dirs:
        if kd.exists():
            candidates.extend([
                kd / f"{protein_info.accession}.pdb",
                kd / f"{protein_info.accession}.cif",
                kd / f"{protein_info.entry_name}.pdb",
                kd / f"{protein_info.entry_name}.cif",
            ])

    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            return c

    # Fallback: try ensure_structure
    try:
        from src.utils.visualize.context import ensure_structure
        s_path = ensure_structure(
            protein_info.accession,
            out_dir=str(struct_dir),
            global_dir=str(known_dirs[0]) if known_dirs[0].exists() else None,
        )
        return Path(s_path)
    except Exception as exc:
        logger.warning("ensure_structure failed for %s: %s", protein_info.accession, exc)

    return None


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_inference(
    protein_ids: List[str],
    subontologies: List[str],
    output_dir: Path,
    models_dir: Path = INFERENCE_MODELS_DIR,
    tau: Optional[float] = None,
    barebones: bool = False,
    skip_preprocessing: bool = False,
    force_preprocessing: bool = False,
    force_remote: bool = False,
    from_scratch: bool = False,
    max_neighbors: int = 500,
    use_entry_ids: bool = False,
    device: Optional[str] = None,
    batch_size: Optional[int] = 1,
    seed: Optional[int] = 53,
    attribute: bool = True,
    render_putty: bool = True,
    save_pkl: bool = False,
) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
    """Run the full inference pipeline.

    Args:
        protein_ids: List of UniProt identifiers (accession or entry name).
        subontologies: List of GO subontologies to predict (MFO, BPO, CCO).
        output_dir: Directory for output files.
        models_dir: Directory containing model symlinks.
        tau: Score threshold for filtering predictions.
        barebones: If True (requires tau), output leaf terms only.
        skip_preprocessing: If True, skip target preprocessing (assume .pt exists).
        force_preprocessing: If True, force target preprocessing from scratch.
        force_remote: If True, force remote UniProt API lookup for ID resolution.
        from_scratch: If True, both force_preprocessing and force_remote are enabled.
        max_neighbors: Maximum neighbors to include in the inference graph.
        use_entry_ids: If True, output predictions keyed by EntryID.
        device: Device to run on (e.g. cuda:1 or cpu).
        batch_size: Inference batch size (default 1 to avoid GPU OOM on large graphs).
        seed: Random seed for sampling reproducibility (default: 53). Set to None for random.
        attribute: If True (default), extract layer-2 residue attention attribution scores.
        render_putty: If True (and attribute=True), render 3D structure using PyMOL putty layout.
        save_pkl: If True, also save predictions as a pickled dictionary (.pkl). Default False.

    Returns:
        Dict mapping subontology → {protein_id: [(go_term, score), ...]}.
    """
    if seed is not None:
        set_seed(seed)

    if from_scratch:
        force_preprocessing = True
        force_remote = True
        skip_preprocessing = False

    output_dir.mkdir(parents=True, exist_ok=True)

    # Attach file handler so the complete inference log is saved to the output dir
    log_file = (output_dir / "inference.log").resolve()
    root_logger = logging.getLogger()
    existing_file_handlers = [
        h for h in root_logger.handlers
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename).resolve() == log_file
    ]
    if not existing_file_handlers:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        root_logger.addHandler(file_handler)
        added_handler = file_handler
    else:
        added_handler = None

    all_results: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
    torch_device = torch.device(device) if device else torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # ---- Step 1: Resolve protein IDs ----
    logger.info("Step 1: Resolving %d protein identifier(s)...", len(protein_ids))

    protein_infos = resolve_proteins(
        protein_ids,
        require_sequence=(not skip_preprocessing),
        force_remote=force_remote,
    )
    logger.info("Resolved %d proteins.", len(protein_infos))

    # ---- Step 2: Load vocabulary and reference config ----
    ref_model_dir = resolve_model_dir(subontologies[0], models_dir)
    ref_config = load_model_config(ref_model_dir)
    ipr_vocab_size = get_ipr_vocab_size()
    go_vocab_info = get_go_vocab_info()

    # ---- Step 3: Preprocess target proteins ----
    target_graphs_dir = output_dir / "target_graphs"
    if not skip_preprocessing:
        logger.info("Step 2: Preprocessing %d target protein(s)...", len(protein_infos))
        for pinfo in protein_infos:
            try:
                preprocess_target(
                    pinfo,
                    output_dir=target_graphs_dir,
                    ipr_vocab_size=ipr_vocab_size,
                    force=force_preprocessing,
                )
            except Exception as exc:
                logger.error(
                    "Failed to preprocess %s: %s. Continuing with remaining proteins.",
                    pinfo.accession,
                    exc,
                    exc_info=True,
                )
        from src.utils.inference.preprocess_target import unload_embedder
        unload_embedder()
    else:
        logger.info("Step 2: Skipping preprocessing (--skip-preprocessing).")

    # ---- Step 4: Discover neighborhoods ----
    logger.info("Step 3: Discovering neighborhoods...")
    workspace = output_dir / "workspace"

    # Fast batched DIAMOND alignment
    try:
        from src.utils.inference.prepare_neighbors import run_diamond_batch
        run_diamond_batch(protein_infos, workspace=workspace, force=force_preprocessing)
    except Exception as exc:
        logger.warning("Batched DIAMOND failed (%s), will proceed with per-protein.", exc)

    per_query_neighbors: Dict[str, Dict[str, dict]] = {}
    per_query_pairwise: Dict[str, List[Tuple[str, str, str, float]]] = {}
    all_subgraph_nodes: Set[str] = {p.accession for p in protein_infos}

    # Pre-build in-memory alignment index once for fast induced edge queries
    from src.utils.inference.prepare_neighbors import find_induced_alignment_edges, get_alignment_index
    get_alignment_index()

    for pinfo in protein_infos:
        try:
            res = identify_neighbors(
                pinfo,
                workspace=workspace,
                max_neighbors=max_neighbors,
                return_pairwise=True,
            )
            if isinstance(res, tuple):
                neighbors, pairwise_string = res
            else:
                neighbors, pairwise_string = res, []

            per_query_neighbors[pinfo.accession] = neighbors

            # Discover induced alignment and STRING-DB edges for localized neighborhood
            subgraph_nodes = {pinfo.accession} | {
                n_acc for n_acc, n_info in neighbors.items() if n_info.get("has_pt", False)
            }
            induced_aln = find_induced_alignment_edges(subgraph_nodes)

            query_pairwise = [
                (acc_a, acc_b, "stringdb", s_score)
                for acc_a, acc_b, s_score in pairwise_string
            ] + [
                (p1, p2, "aligned_with", bscore)
                for p1, p2, bscore in induced_aln
            ]
            per_query_pairwise[pinfo.accession] = query_pairwise

            for n_acc in subgraph_nodes:
                all_subgraph_nodes.add(n_acc)

        except Exception as exc:
            logger.warning(
                "Failed to discover neighbors for %s: %s", pinfo.accession, exc
            )
            per_query_neighbors[pinfo.accession] = {}
            per_query_pairwise[pinfo.accession] = []

    total_query_neighbors = sum(len(n) for n in per_query_neighbors.values())
    total_query_edges = sum(len(e) for e in per_query_pairwise.values())
    logger.info(
        "Discovered neighborhoods for %d proteins: %d total neighbor links, %d total induced pairwise edges.",
        len(protein_infos),
        total_query_neighbors,
        total_query_edges,
    )

    # ---- Step 5: Run inference per subontology ----
    protein_names = {p.accession: p.entry_name for p in protein_infos}
    id_mapping = get_pid_mapping()

    # Pre-build localized InferenceDataset for each query protein once.
    # Graph wiring and edge normalization are identical across all subontologies.
    base_edge_norm = load_edge_normalization(ref_model_dir)
    logger.info("Constructing localized graphs for %d query protein(s)...", len(protein_infos))
    query_datasets: Dict[str, InferenceDataset] = {}
    for pinfo in protein_infos:
        q_neighbors = per_query_neighbors.get(pinfo.accession, {})
        q_pairwise = per_query_pairwise.get(pinfo.accession, [])
        query_datasets[pinfo.accession] = InferenceDataset(
            query_proteins=[pinfo],
            config=ref_config,
            edge_norm=base_edge_norm,
            per_query_neighbors={pinfo.accession: q_neighbors},
            pairwise_edges=q_pairwise,
            target_graphs_dir=target_graphs_dir,
            external_annotations={},
            id_mapping=id_mapping,
        )

    for onto in subontologies:
        logger.info("=" * 60)
        logger.info("Running inference for subontology: %s", onto)
        logger.info("=" * 60)

        try:
            # Load model config
            model_dir = resolve_model_dir(onto, models_dir)
            config = load_model_config(model_dir)
            config["data"]["subontology"] = onto

            # Preload SwissProt GO annotations for this subontology
            # Priority: explicit external_annotations_path from config, then exp_only fallback
            ext_annot_path = config["data"].get("external_annotations_path")
            annot_path = None
            if ext_annot_path:
                ext_p = Path(str(ext_annot_path).format(onto=onto))
                if not ext_p.is_absolute():
                    from src.utils.constants import PROJECT_ROOT
                    ext_p = (PROJECT_ROOT / ext_p).resolve()
                if ext_p.exists():
                    annot_path = ext_p
                    logger.info("Using configured external_annotations_path: %s", annot_path)
            if annot_path is None:
                exp_only = config["data"].get("exp_only", True)
                annot_template = GO_EXP_ANNOTATION_TEMPLATE if exp_only else GO_ANNOTATION_TEMPLATE
                annot_path = annot_template.with_name(annot_template.name.format(onto=onto))

            external_annotations = {}
            if annot_path.exists():
                import pandas as pd
                annot_df = pd.read_csv(annot_path, sep="\t")
                pid_map, rev_pid_map = id_mapping
                for key_val, term_val in zip(annot_df["EntryID"].astype(str), annot_df["term"].astype(str)):
                    key = key_val.strip()
                    terms = [t.strip() for t in term_val.split(";") if t.strip()]
                    external_annotations[key] = terms
                    if key in rev_pid_map:
                        external_annotations[rev_pid_map[key]] = terms
                    if key in pid_map:
                        external_annotations[pid_map[key]] = terms
                logger.info("Preloaded %d SwissProt %s annotations (dual-indexed) from %s.", len(external_annotations), onto, annot_path)

            go_vocab_size = go_vocab_info[onto]["vocab_size"]
            go_idx_to_term = {
                v: k
                for k, v in go_vocab_info[onto]["go_to_idx"].items()
            }

            model = load_model_for_inference(config, go_vocab_size, model_dir, onto, device=torch_device)

            predictions: Dict[str, List[Tuple[str, float]]] = {}
            num_neighbors = {}
            for edge_type_str, num_samples in config["model"]["sampled_edges"].items():
                edge_type_tuple = tuple(edge_type_str.split("__"))
                num_neighbors[edge_type_tuple] = [num_samples]

            logger.info("Executing localized inference for %d query proteins...", len(protein_infos))
            for idx, pinfo in enumerate(protein_infos):
                dataset = query_datasets[pinfo.accession]
                dataset.subontology = onto
                dataset.go_vocab_size = go_vocab_size
                dataset.external_annotations = external_annotations
                dataset.config = config

                if seed is not None:
                    torch.manual_seed(seed + idx)

                loader = NeighborLoader(
                    dataset.data,
                    num_neighbors=num_neighbors,
                    batch_size=1,
                    input_nodes=("protein", dataset.query_mask),
                    transform=make_inference_batch_transform(dataset),
                    shuffle=False,
                    num_workers=0,
                )

                batch = next(iter(loader)).to(torch_device)
                with torch.no_grad():
                    if attribute:
                        out_tensor, (attn1, attn2) = model(
                            batch.x_dict, batch.edge_index_dict, batch, return_attention_weights=True
                        )
                        out = out_tensor[0].cpu().numpy()
                    else:
                        out = model(batch.x_dict, batch.edge_index_dict, batch)[0].cpu().numpy()

                out_key = pinfo.entry_name if use_entry_ids else pinfo.accession
                term_scores = []
                for j, score in enumerate(out):
                    if score > 0:
                        term_scores.append((go_idx_to_term[j], float(score)))
                term_scores.sort(key=lambda x: x[1], reverse=True)
                predictions[out_key] = term_scores

                if attribute:
                    et = ("aa", "belongs_to", "protein")
                    res_scores = []
                    if et in attn2:
                        edge_index, alpha = attn2[et]
                        if alpha.dim() > 1:
                            alpha = alpha.mean(dim=-1)
                        alpha = alpha.detach().cpu()
                        edge_index = edge_index.detach().cpu()

                        mask = (edge_index[1] == 0)
                        aa_indices = edge_index[0][mask].tolist()
                        aa_scores = alpha[mask].tolist()
                        res_scores = [(idx + 1, float(score)) for idx, score in zip(aa_indices, aa_scores)]
                        res_scores.sort(key=lambda x: x[0])

                    if res_scores:
                        attr_dir = output_dir / "attribution" / onto
                        attr_dir.mkdir(parents=True, exist_ok=True)
                        rendered_ok = False

                        if render_putty:
                            try:
                                struct_path = resolve_protein_structure(
                                    pinfo, output_dir=output_dir, target_graphs_dir=target_graphs_dir
                                )
                                if struct_path and struct_path.exists():
                                    from src.utils.visualize.rendering import render_putty_scene
                                    putty_png = attr_dir / f"{out_key}_putty.png"
                                    rendered_ok = render_putty_scene(
                                        structure_path=str(struct_path),
                                        residue_scores=res_scores,
                                        image_path=str(putty_png),
                                        title=f"{out_key} – {onto} Attribution (Putty)",
                                    )
                                    if rendered_ok:
                                        logger.info("  [%s] Saved putty 3D render to %s", onto, putty_png)
                                    else:
                                        logger.warning("  [%s] PyMOL rendering did not succeed for %s", onto, pinfo.accession)
                                else:
                                    logger.warning(
                                        "  [%s] Could not locate or download 3D structure for %s",
                                        onto, pinfo.accession
                                    )
                            except Exception as exc:
                                logger.warning(
                                    "  [%s] PyMOL putty rendering failed for %s: %s",
                                    onto, pinfo.accession, exc
                                )

                        # Save attribution TSV only if PyMOL rendering is unavailable or failed
                        if not rendered_ok:
                            attr_tsv = attr_dir / f"{out_key}_residue_attribution.tsv"
                            with open(attr_tsv, "w", encoding="utf-8") as f:
                                f.write("residue\tattribution_score\n")
                                for r_num, r_score in res_scores:
                                    f.write(f"{r_num}\t{r_score:.6f}\n")
                            logger.info("  [%s] PyMOL rendering unavailable; saved residue attribution fallback to %s", onto, attr_tsv)

                if (idx + 1) % 200 == 0 or (idx + 1) == len(protein_infos):
                    logger.info("  [%s] Completed %d / %d proteins...", onto, idx + 1, len(protein_infos))

            all_results[onto] = predictions

            # Save and display
            pred_path = save_inference_predictions(
                predictions,
                subontology=onto,
                output_dir=output_dir,
                tau=tau,
                barebones=barebones,
                save_pkl=save_pkl,
            )

            print_summary(
                predictions,
                subontology=onto,
                output_path=pred_path,
                protein_names=protein_names,
                tau=tau,
            )

            # Clean up GPU memory between subontologies
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as exc:
            logger.error(
                "Inference failed for %s: %s", onto, exc, exc_info=True
            )
            continue

    logger.info("Inference pipeline complete. Results in %s", output_dir)
    if added_handler is not None:
        added_handler.flush()
        logging.getLogger().removeHandler(added_handler)
        added_handler.close()
    else:
        for h in root_logger.handlers:
            if isinstance(h, logging.FileHandler):
                h.flush()
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MS-GNN Inference Pipeline — Predict GO annotations for arbitrary proteins.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.utils.inference.infer --proteins P01308 --subontology MFO
  python -m src.utils.inference.infer --proteins P01308 Q9Y6K1 --subontology MFO BPO CCO
  python -m src.utils.inference.infer --proteins P01308 --subontology MFO --tau 0.5 --barebones
  python -m src.utils.inference.infer --proteins-file queries.txt --subontology MFO
        """,
    )

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--proteins",
        nargs="+",
        help="One or more UniProt IDs (accession or entry name).",
    )
    input_group.add_argument(
        "--proteins-file",
        type=Path,
        help="File with one UniProt ID per line.",
    )

    # Subontology
    parser.add_argument(
        "--subontology",
        nargs="+",
        default=GO_ONTOLOGIES,
        choices=GO_ONTOLOGIES,
        help="GO subontology(ies) to predict (default: all three).",
    )

    # Output
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Score threshold for filtering predictions.",
    )
    parser.add_argument(
        "--barebones",
        action="store_true",
        help="Output leaf terms only (requires --tau).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: {INFERENCE_OUTPUT_DIR}/<timestamp>/).",
    )

    # Model
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=INFERENCE_MODELS_DIR,
        help=f"Directory containing model symlinks (default: {INFERENCE_MODELS_DIR}).",
    )

    # Preprocessing
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip target preprocessing (assume .pt files already exist).",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Run inference completely from scratch for target proteins (pull ID conversion & sequence from UniProt API, download structure from AlphaFold, compute ESM embeddings, and build graph).",
    )
    parser.add_argument(
        "--force-preprocessing",
        action="store_true",
        help="Force target preprocessing from scratch even if .pt graph exists in precomputed database.",
    )
    parser.add_argument(
        "--force-remote",
        action="store_true",
        help="Force remote UniProt API lookup for ID resolution and sequence fetching even if protein is in local SwissProt.",
    )
    parser.add_argument(
        "--max-neighbors",
        type=int,
        default=500,
        help="Maximum neighbors per protein (default: 500).",
    )
    parser.add_argument(
        "--use-entry-ids",
        action="store_true",
        help="Keep EntryIDs in output predictions instead of converting to accessions.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (e.g. cuda:1 or cpu).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=53,
        help="Random seed for sampling reproducibility (default: 53). Set to -1 for random.",
    )
    parser.add_argument(
        "--no-attribution",
        dest="attribute",
        action="store_false",
        default=True,
        help="Disable layer-2 residue attribution and 3D PyMOL putty rendering (enabled by default).",
    )
    parser.add_argument(
        "--save-pkl",
        action="store_true",
        default=False,
        help="Save binary pickled predictions (.pkl) in addition to .tsv (default: False).",
    )

    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    # Validate barebones
    if args.barebones and args.tau is None:
        logger.error("--barebones requires --tau.")
        sys.exit(1)

    # Resolve protein IDs from file if needed
    if args.proteins_file:
        if not args.proteins_file.exists():
            logger.error("Proteins file not found: %s", args.proteins_file)
            sys.exit(1)
        protein_ids = [
            line.strip()
            for line in args.proteins_file.read_text().strip().split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
    else:
        protein_ids = args.proteins

    if not protein_ids:
        logger.error("No protein IDs provided.")
        sys.exit(1)

    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = INFERENCE_OUTPUT_DIR / timestamp
    else:
        output_dir = args.output_dir

    seed = None if args.seed is not None and args.seed < 0 else args.seed

    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "inference.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    logger.info("MS-GNN Inference Pipeline")
    logger.info("Proteins: %s", ", ".join(protein_ids))
    logger.info("Subontologies: %s", ", ".join(args.subontology))
    logger.info("Output directory: %s", output_dir)
    logger.info("Seed: %s", seed)
    logger.info("Attribution & Putty 3D: %s", args.attribute)

    run_inference(
        protein_ids=protein_ids,
        subontologies=args.subontology,
        output_dir=output_dir,
        models_dir=args.models_dir,
        tau=args.tau,
        barebones=args.barebones,
        skip_preprocessing=args.skip_preprocessing,
        force_preprocessing=args.force_preprocessing,
        force_remote=args.force_remote,
        from_scratch=args.from_scratch,
        max_neighbors=args.max_neighbors,
        use_entry_ids=args.use_entry_ids,
        device=args.device,
        batch_size=args.batch_size,
        seed=seed,
        attribute=args.attribute,
        render_putty=True,
        save_pkl=args.save_pkl,
    )


if __name__ == "__main__":
    main()
