import pandas as pd
import os
import sys
import subprocess
import pickle
from collections import defaultdict
import tqdm
import argparse
import torch


def save_predictions(config, model, loader, device, dataset, split=None):
    model.eval()
    predictions = []
    # Reverse mapping from index to GO term
    go_idx_to_term = {
        v: k for k, v in dataset.go_vocab_info[dataset.subontology]["go_to_idx"].items()
    }
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc=f"Predicting on {split} proteins"):
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict, batch)
            batch_size = batch["protein"].batch_size
            protein_ids = batch["protein"].protein_ids[:batch_size]
            scores = torch.sigmoid(out[:batch_size]).cpu().numpy()
            for i, pid in enumerate(protein_ids):
                for j, score in enumerate(scores[i]):
                    if score > 0:  # Save all nonzero scores
                        term_id = go_idx_to_term[j]
                        predictions.append([pid, term_id, float(score)])
    pred_path = f"{config['run']['results_dir']}/predictions/predictions_{split}_{dataset.subontology}.tsv"
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    df = pd.DataFrame(predictions, columns=["target_ID", "term_ID", "score"])
    df.to_csv(pred_path, sep="\t", index=False)


def evaluate(config, logger, output_dir, subontology, split):
    """
    Evaluate the predictions using the ground truth (GT) annotations and the BeProf evaluation method.
    """
    dataset = config["data"]["dataset"]
    background_pkl = f"./data/{dataset}/background_{dataset}_{split}.pkl"
    if os.path.exists(background_pkl):
        logger.info(f"Using existing background file: {background_pkl}")
    background_pkl = f"./data/{dataset}/background_{dataset}_{split}.pkl"
    # If background file does not exist, run background.py to create it
    if not os.path.exists(background_pkl):
        logger.info(f"Background file {background_pkl} does not exist. Creating it...")
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
    go_obo_file = "./data/go.obo"

    # Check if GT exists in pkl format. If not, convert GT TSV to pkl using gt_convert
    gt_pkl = f"./data/{dataset}/{dataset}_{subontology}_{split}_annotations.pkl"
    if not os.path.exists(gt_pkl):
        gt_tsv = f"./data/{dataset}/{dataset}_{subontology}_{split}_annotations.tsv"
        if os.path.exists(gt_tsv):
            logger.info(f"Converting Ground Truth TSV {gt_tsv} to pkl format")
            gt_convert(gt_tsv)
        else:
            logger.error(f"Ground Truth TSV file {gt_tsv} does not exist.")
            raise FileNotFoundError(f"Ground Truth TSV file {gt_tsv} does not exist.")

    # Evaluate predictions
    logger.info(f"Evaluating predictions")
    # Convert predictions to pkl
    pred_file = f"{output_dir}/predictions/predictions_{split}_{subontology}.tsv"
    pred_pkl = f"{output_dir}/predictions/predictions_{split}_{subontology}.pkl"
    if os.path.exists(pred_file):
        pred_dict = convert_predictions(pred_file, subontology)
        with open(pred_pkl, "wb") as f:
            pickle.dump(pred_dict, f)

        run_beprof_evaluation(
            logger,
            pred_pkl,
            gt_pkl,
            background_pkl,
            go_obo_file,
            f"{output_dir}/evaluation/{split}_{subontology}",
        )
    else:
        logger.warning(f"Predictions file {pred_file} does not exist.")


def run_beprof_evaluation(
    logger, pred_pkl, gt_pkl, background_pkl, go_obo_file, eval_output_dir
):
    """
    Run beprof_eval.py as a subprocess.
    """
    os.makedirs(eval_output_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "src/utils/beprof_eval.py",
        "--predict",
        pred_pkl,
        "--output_path",
        eval_output_dir,
        "--true",
        gt_pkl,
        "--background",
        background_pkl,
        "--go",
        go_obo_file,
        "--metrics",
        "0,1,2,3,4,5",  # All metrics: F_max, Smin, Aupr, ICAupr, DPAupr, threshold
    ]

    logger.info(f"Running BeProf evaluation: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )

        logger.info(f"BeProf evaluation completed successfully")
        logger.info(f"Results saved to: {eval_output_dir}")

        # Log stdout and stderr
        if result.stdout:
            logger.info(f"BeProf stdout:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"BeProf stderr:\n{result.stderr}")

    except subprocess.CalledProcessError as e:
        logger.error(f"BeProf evaluation failed with return code {e.returncode}")
        if e.stdout:
            logger.error(f"BeProf stdout:\n{e.stdout}")
        if e.stderr:
            logger.error(f"BeProf stderr:\n{e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Error running BeProf evaluation: {str(e)}")
        raise


def gt_convert(gt_tsv):
    """
    Convert GT to pkl.
    """
    # Extract the ontology type (BPO, CCO, MFO) from the filename
    base_name = os.path.basename(gt_tsv)
    ontology_type = base_name.split("_")[1]  # Extract BPO, CCO, or MFO
    ontology_mapping = {"BPO": "all_bp", "CCO": "all_cc", "MFO": "all_mf"}
    ontology_key = ontology_mapping[ontology_type]

    # Output filename: same as input but with .pkl extension
    gt_pkl = os.path.splitext(gt_tsv)[0] + ".pkl"

    print(f"Processing {base_name}...")

    # Read the input file
    df = pd.read_csv(gt_tsv, sep="\t")
    df["term"] = df["term"].str.split("; ")
    df = df.explode("term")

    # Build the nested dictionary for pickling
    protein_go_terms = defaultdict(
        lambda: {"all_bp": set(), "all_cc": set(), "all_mf": set()}
    )
    for _, row in df.iterrows():
        protein_id = row["EntryID"]
        term_id = row["term"]
        protein_go_terms[protein_id][ontology_key].add(term_id)

    # Convert defaultdict to regular dict for pickling
    protein_go_terms_dict = dict(protein_go_terms)

    # Save to pickle file
    with open(gt_pkl, "wb") as f:
        pickle.dump(protein_go_terms_dict, f)
    print(f"Saved pickle file: {gt_pkl}")

    # # Print sample data for verification
    # if protein_go_terms_dict:
    #     sample_protein = next(iter(protein_go_terms_dict))
    #     print(f"Sample data for {sample_protein} in {ontology_type}:")
    #     print(protein_go_terms_dict[sample_protein])


def convert_predictions(pred_file, aspect):
    """
    Converts a TSV prediction file with columns: target_ID, term_ID
    into a dictionary where each protein gets a 'bp' dictionary of GO term predictions.
    """
    subontology = aspect[:2].lower()
    df = pd.read_csv(pred_file, sep="\t")
    # Split term column by '; ' and explode
    df["term_ID"] = df["term_ID"].str.split("; ")
    df = df.explode("term_ID")
    pred_dict = {}
    for prot, group in tqdm.tqdm(df.groupby("target_ID")):
        # if subontology == "all":
        #     for sub in ["cc", "mf", "bp"]:
        #         pred_dict[prot] = {
        #             f"{sub}": dict(zip(group["term_ID"], [1] * len(group["term_ID"])))
        #         }
        pred_dict[prot] = {
            f"{subontology}": dict(zip(group["term_ID"], group["score"]))
        }
    return pred_dict


def setup_logging(output_dir, subontology):
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
    parser = argparse.ArgumentParser(description="Evaluate predictions using BeProf.")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing predictions and evaluation results. Will also serve as output",
    )
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument(
        "--subontology", required=True, help="Ontology aspect (BPO, CCO, MFO)."
    )
    args = parser.parse_args()
    logger = setup_logging(args.input_dir, args.subontology)
    evaluate(logger, args.input_dir, args.dataset, args.subontology)
