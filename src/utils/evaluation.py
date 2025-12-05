"""Evaluation utilities for model predictions and metrics computation.

This module provides functions for evaluating protein function predictions, including:
- Computing precision-recall metrics (AUPR, F-max)
- Saving model predictions to files
- Generating evaluation plots and visualizations
- Integration with BEPROF and CAFA evaluation frameworks
"""

import pandas as pd
import os
import sys
import subprocess
import pickle
import tqdm
import argparse
import logging
import torch
from src.utils.helpers import timeit
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import wandb

# Import BEPROF evaluation functions
from src.utils.beprof_eval import (
    run_beprof_evaluation as run_beprof,
    gt_convert,
    convert_predictions,
)

# Import CAFA evaluation functions
from src.utils.cafa_evaluation import (
    run_cafa_evaluation as run_cafa,
)


def compute_metrics(all_scores, all_targets):
    """
    Compute Area Under the Precision-Recall Curve (AUPR) and F-max.
    Precision and recall are computed on the global pool of predictions (Micro).
    """
    if isinstance(all_scores, (list, tuple)):
        all_scores = np.concatenate(all_scores, axis=0).flatten()
    else:
        all_scores = np.array(all_scores).flatten()
    if isinstance(all_targets, (list, tuple)):
        all_targets = np.concatenate(all_targets, axis=0).flatten()
    else:
        all_targets = np.array(all_targets).flatten()

    precision, recall, _ = precision_recall_curve(all_targets.astype(int), all_scores)
    aupr = auc(recall, precision)

    # Fmax calculation
    f_scores = 2 * precision * recall / (precision + recall + 1e-8)
    fmax = np.max(f_scores)

    return precision, recall, aupr, fmax


def plot_aupr(precision, recall, aupr=None):
    """
    Compute Area Under the Precision-Recall Curve (AUPR).
    """
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AUPR={aupr:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Intermediate Validation PR Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    pr_plot = wandb.Image(plt)
    plt.close()

    return pr_plot


@timeit
def save_predictions(config, model, loader, device, dataset, split=None):
    """Save model predictions to a TSV file.

    Args:
        config: Configuration dictionary
        model: Trained ProteinGNN model
        loader: DataLoader for the dataset split
        device: torch.device for computation
        dataset: SwissProtDataset instance
        split: Dataset split name ('train', 'val', or 'test')
    """
    model.eval()
    go_idx_to_term = {
        v: k for k, v in dataset.go_vocab_info[dataset.subontology]["go_to_idx"].items()
    }
    pred_path = f"{config['run']['results_dir']}/predictions/predictions_{split}_{dataset.subontology}.tsv"
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    with open(pred_path, "w") as f:
        f.write("target_ID\tterm_ID\tscore\n")
    batch_buffer = []
    with torch.no_grad():
        for batch_count, batch in tqdm.tqdm(
            enumerate(loader), desc=f"Predicting on {split} proteins"
        ):
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict, batch)
            batch_size = batch["protein"].batch_size
            protein_ids = batch["protein"].protein_ids[:batch_size]
            if dataset.uses_entryid:
                protein_ids = [dataset.rev_pid_mapping[idx] for idx in protein_ids]
            scores = out[:batch_size].cpu().numpy()
            for i, pid in enumerate(protein_ids):
                for j, score in enumerate(scores[i]):
                    if score > 0:
                        term_id = go_idx_to_term[j]
                        batch_buffer.append(f"{pid}\t{term_id}\t{float(score)}\n")
            if batch_count % 50 == 0:
                with open(pred_path, "a") as f:
                    f.writelines(batch_buffer)
                batch_buffer = []
        if batch_buffer:
            with open(pred_path, "a") as f:
                f.writelines(batch_buffer)


@timeit
def evaluate(
    logger,
    dataset,
    output_dir,
    subontology,
    split,
    run_cafa_eval=True,
    run_beprof_eval=True,
    ia_file=None,
):
    """
    Evaluate the predictions using both CAFA and BeProf evaluation methods.

    Args:
        logger: Logger instance for output
        dataset: Dataset name (e.g., "swissprot")
        output_dir: Base output directory containing predictions
        subontology: GO subontology (BPO, CCO, MFO)
        split: Data split (train, val, test)
        run_cafa_eval: Whether to run CAFA evaluation (default: True)
        run_beprof_eval: Whether to run BEPROF evaluation (default: True)
    """
    go_obo_file = "./data/go.obo"
    gt_tsv = f"./data/{dataset}/{dataset}_{subontology}_{split}_annotations.tsv"
    pred_file = f"{output_dir}/predictions/predictions_{split}_{subontology}.tsv"
    logger.info(f"Starting evaluation for {subontology} on {split} split")

    # Run BEPROF evaluation
    if run_beprof_eval:
        background_pkl = f"./data/{dataset}/background_{dataset}_{split}.pkl"
        if os.path.exists(background_pkl):
            logger.info(f"Using existing background file: {background_pkl}")
        background_pkl = f"./data/{dataset}/background_{dataset}_{split}.pkl"
        # Gets IC values from training set annotations
        if not os.path.exists(background_pkl):
            logger.info(
                f"Background file {background_pkl} does not exist. Creating it..."
            )
            background_cmd = [
                sys.executable,
                "src/utils/background.py",
                "--cco",
                f"./data/{dataset}/{dataset}_CCO_train_annotations.tsv",
                "--bpo",
                f"./data/{dataset}/{dataset}_BPO_train_annotations.tsv",
                "--mfo",
                f"./data/{dataset}/{dataset}_MFO_train_annotations.tsv",
                "--test_cco",
                f"./data/{dataset}/{dataset}_CCO_{split}_annotations.tsv",
                "--test_bpo",
                f"./data/{dataset}/{dataset}_BPO_{split}_annotations.tsv",
                "--test_mfo",
                f"./data/{dataset}/{dataset}_MFO_{split}_annotations.tsv",
                "--output",
                background_pkl,
            ]
            try:
                subprocess.run(
                    background_cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logger.info(f"Background file created at {background_pkl}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to create background file: {e.stderr}")
                raise e

        # BEPROF requires ground truth in pkl format
        gt_pkl = f"./data/{dataset}/{dataset}_{subontology}_{split}_annotations.pkl"
        if not os.path.exists(gt_pkl):
            if os.path.exists(gt_tsv):
                logger.info(f"Converting Ground Truth TSV {gt_tsv} to pkl format")
                gt_convert(gt_tsv)
            else:
                logger.error(f"Ground Truth TSV file {gt_tsv} does not exist.")
                raise FileNotFoundError(
                    f"Ground Truth TSV file {gt_tsv} does not exist."
                )

        # Evaluate predictions
        pred_pkl = f"{output_dir}/predictions/predictions_{split}_{subontology}.pkl"

        if not os.path.exists(pred_file):
            logger.warning(f"Predictions file {pred_file} does not exist.")
            return

        logger.info(f"Evaluating predictions for {subontology} on {split} split")

        # Convert predictions to pkl for BEPROF evaluation
        pred_dict = convert_predictions(pred_file, subontology)
        with open(pred_pkl, "wb") as f:
            pickle.dump(pred_dict, f)
        logger.info(f"Converted predictions saved to {pred_pkl}")

        logger.info("Running BEPROF evaluation...")
        beprof_output_dir = f"{output_dir}/evaluation/{split}_{subontology}/beprof-eval"
        try:
            run_beprof(
                predictions_file=pred_pkl,
                gt_file=gt_pkl,
                background_file=background_pkl,
                ontology_file=go_obo_file,
                output_dir=beprof_output_dir,
                subontology=subontology,
                metrics="0,1,2,3,4,5",
            )
            logger.info(
                f"BEPROF evaluation completed. Results saved to: {beprof_output_dir}"
            )
        except Exception as e:
            logger.error(f"BEPROF evaluation failed: {e}")
            raise

    # Run CAFA evaluation
    if run_cafa_eval:
        logger.info("Running CAFA evaluation...")
        cafa_output_dir = f"{output_dir}/evaluation/{split}_{subontology}/cafa-eval"
        try:
            run_cafa(
                predictions_file=pred_file,
                gt_file=gt_tsv,
                ontology_file=go_obo_file,
                output_dir=cafa_output_dir,
                ia_file=ia_file,
            )
            logger.info(
                f"CAFA evaluation completed. Results saved to: {cafa_output_dir}"
            )
        except ImportError as e:
            logger.warning(f"CAFA evaluation skipped (cafaeval not installed): {e}")
        except Exception as e:
            logger.error(f"CAFA evaluation failed: {e}")

    logger.info(f"Evaluation completed for {subontology} on {split} split")


def setup_logging(output_dir, subontology):
    """Set up logging configuration for evaluation.

    Args:
        output_dir: Directory to store log files
        subontology: GO subontology name (MFO, BPO, or CCO)
    """
    import logging

    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{subontology}.log")

    logger = logging.getLogger(f"{subontology}")
    logger.setLevel(logging.INFO)

    # Create file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    if not logger.hasHandlers():
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate predictions using BEPROF and/or CAFA evaluators."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing predictions and evaluation results. Will also serve as output",
    )
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument(
        "--subontology", required=True, help="Ontology aspect (BPO, CCO, MFO)."
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Data split to evaluate on (train, val, test). Default: test",
    )
    parser.add_argument(
        "--no-cafa",
        action="store_true",
        help="Skip CAFA evaluation",
    )
    parser.add_argument(
        "--cafa_ia_file",
        help="CAFA evaluation IA file path",
        default=None,
    )

    parser.add_argument(
        "--no-beprof",
        action="store_true",
        help="Skip BEPROF evaluation",
    )
    args = parser.parse_args()
    eval_logger = setup_logging(args.input_dir, args.subontology)
    evaluate(
        eval_logger,
        args.dataset,
        args.input_dir,
        args.subontology,
        args.split,
        run_cafa_eval=not args.no_cafa,
        run_beprof_eval=not args.no_beprof,
        ia_file=args.cafa_ia_file,
    )
