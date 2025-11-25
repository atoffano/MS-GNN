"""CAFA evaluation script using the CAFA-evaluator library.

This module provides functionality for evaluating protein function predictions
using the CAFA-evaluator (https://github.com/BioComputingUP/CAFA-evaluator).

It supports:
- Reading predictions from TSV files (target_ID, term_ID, score)
- Converting ground truth annotations for CAFA-evaluator
- Computing IA (Information Accretion) weights
- Running full CAFA evaluation with metrics like F-max, S-min, AUPR
"""

import sys
import os
import argparse
import tempfile
import shutil
import logging
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args(argv):
    """Parse command line arguments.

    Args:
        argv: Command line arguments list

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Evaluate protein function predictions using CAFA-evaluator."
    )

    parser.add_argument(
        "--predictions",
        "-p",
        required=True,
        help="Path to prediction TSV file with columns: target_ID, term_ID, score"
    )

    parser.add_argument(
        "--ground_truth",
        "-gt",
        required=True,
        help="Path to ground truth annotation TSV file"
    )

    parser.add_argument(
        "--ontology",
        "-go",
        default=None,
        help="Path to OBO ontology file. If empty, uses ./data/go.obo"
    )

    parser.add_argument(
        "--ia",
        "-ia",
        default=None,
        help="Path to Information Accretion file (optional). If provided, weighted metrics will be computed."
    )

    parser.add_argument(
        "--output",
        "-o",
        default="./cafa_results",
        help="Output directory for evaluation results. Default: ./cafa_results"
    )

    parser.add_argument(
        "--th_step",
        type=float,
        default=0.01,
        help="Threshold step size for PR curve calculation. Default: 0.01"
    )

    parser.add_argument(
        "--prop",
        choices=["max", "fill"],
        default="max",
        help="Propagation strategy: 'max' or 'fill'. Default: max"
    )

    parser.add_argument(
        "--norm",
        choices=["cafa", "pred", "gt"],
        default="cafa",
        help="Normalization strategy. Default: cafa"
    )

    parser.add_argument(
        "--no_orphans",
        action="store_true",
        help="Exclude orphan nodes (e.g. roots) from the calculation"
    )

    parser.add_argument(
        "--max_terms",
        type=int,
        default=None,
        help="Maximum number of terms per protein to consider"
    )

    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=1,
        help="Number of parallel threads. Default: 1"
    )

    return parser.parse_args(argv)


def convert_predictions_to_cafa_format(predictions_file, output_dir):
    """Convert prediction TSV to CAFA-evaluator format.

    The CAFA-evaluator expects prediction files with format:
    target_ID<tab>term_ID<tab>score

    Args:
        predictions_file: Path to prediction TSV file
        output_dir: Directory to save converted prediction file

    Returns:
        Path to the predictions directory
    """
    logger.info(f"Loading predictions from {predictions_file}")

    df = pd.read_csv(predictions_file, sep="\t")

    # Normalize column names (handle variations)
    column_mapping = {}
    for col in df.columns:
        lower_col = col.lower().replace(" ", "_")
        if "target" in lower_col:
            column_mapping[col] = "target_ID"
        elif "term" in lower_col:
            column_mapping[col] = "term_ID"
        elif "score" in lower_col:
            column_mapping[col] = "score"

    if column_mapping:
        df = df.rename(columns=column_mapping)

    # Ensure required columns exist
    required_cols = ["target_ID", "term_ID", "score"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}. Found columns: {list(df.columns)}")

    # Handle cases where term_ID might contain multiple terms separated by "; "
    if df["term_ID"].str.contains("; ", regex=False).any():
        # Split and explode the terms
        df["term_ID"] = df["term_ID"].str.split("; ")
        df = df.explode("term_ID")

    # Keep only required columns
    df = df[["target_ID", "term_ID", "score"]]

    # Remove rows with missing or invalid scores
    df = df.dropna(subset=["score"])

    # Ensure score is numeric
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])

    # Create predictions directory
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    # Write to CAFA format (no header)
    output_file = os.path.join(pred_dir, "predictions.tsv")
    df.to_csv(output_file, sep="\t", header=False, index=False)

    logger.info(f"Saved {len(df)} predictions to {output_file}")
    return pred_dir


def convert_ground_truth_to_cafa_format(gt_file, output_file):
    """Convert ground truth annotation file to CAFA-evaluator format.

    CAFA-evaluator expects ground truth format:
    target_ID<tab>term_ID

    Args:
        gt_file: Path to ground truth TSV file
        output_file: Path to save converted ground truth file

    Returns:
        Path to the converted ground truth file
    """
    logger.info(f"Loading ground truth from {gt_file}")

    df = pd.read_csv(gt_file, sep="\t")

    # Normalize column names
    column_mapping = {}
    for col in df.columns:
        lower_col = col.lower().replace(" ", "_")
        if "entry" in lower_col or "target" in lower_col or col.lower() == "entryid":
            column_mapping[col] = "target_ID"
        elif "term" in lower_col:
            column_mapping[col] = "term_ID"

    if column_mapping:
        df = df.rename(columns=column_mapping)

    # Ensure required columns exist
    if "target_ID" not in df.columns:
        raise ValueError(f"Missing target/entry ID column. Found columns: {list(df.columns)}")
    if "term_ID" not in df.columns:
        raise ValueError(f"Missing term column. Found columns: {list(df.columns)}")

    # Handle cases where term_ID might contain multiple terms separated by "; "
    if df["term_ID"].str.contains("; ", regex=False).any():
        df["term_ID"] = df["term_ID"].str.split("; ")
        df = df.explode("term_ID")

    # Keep only required columns
    df = df[["target_ID", "term_ID"]]

    # Remove rows with missing values
    df = df.dropna()

    # Write to CAFA format (no header)
    df.to_csv(output_file, sep="\t", header=False, index=False)

    logger.info(f"Saved {len(df)} ground truth annotations to {output_file}")
    return output_file


def run_cafa_evaluation(
    predictions_file,
    gt_file,
    ontology_file,
    output_dir,
    ia_file=None,
    th_step=0.01,
    prop="max",
    norm="cafa",
    no_orphans=False,
    max_terms=None,
    n_threads=1
):
    """Run CAFA evaluation on predictions.

    Args:
        predictions_file: Path to prediction TSV file
        gt_file: Path to ground truth annotation file
        ontology_file: Path to GO ontology file (OBO format)
        output_dir: Output directory for results
        ia_file: Optional path to Information Accretion file
        th_step: Threshold step size
        prop: Propagation strategy ('max' or 'fill')
        norm: Normalization strategy ('cafa', 'pred', or 'gt')
        no_orphans: Whether to exclude orphan nodes
        max_terms: Maximum number of terms per protein
        n_threads: Number of threads for parallel processing

    Returns:
        Tuple of (evaluation_df, best_scores_dict)
    """
    try:
        from cafaeval.evaluation import cafa_eval, write_results
    except ImportError:
        raise ImportError(
            "CAFA-evaluator is not installed. Install it with: pip install cafaeval"
        )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create temp directory for converted files
    temp_dir = tempfile.mkdtemp(prefix="cafa_eval_")
    logger.info(f"Using temporary directory: {temp_dir}")

    try:
        # Convert predictions to CAFA format
        pred_dir = convert_predictions_to_cafa_format(predictions_file, temp_dir)

        # Convert ground truth to CAFA format
        gt_converted = os.path.join(temp_dir, "ground_truth.tsv")
        convert_ground_truth_to_cafa_format(gt_file, gt_converted)

        # Run CAFA evaluation
        logger.info("Running CAFA evaluation...")
        logger.info(f"  Ontology: {ontology_file}")
        logger.info(f"  Predictions: {pred_dir}")
        logger.info(f"  Ground Truth: {gt_converted}")
        if ia_file:
            logger.info(f"  IA File: {ia_file}")

        df, dfs_best = cafa_eval(
            obo_file=ontology_file,
            pred_dir=pred_dir,
            gt_file=gt_converted,
            ia=ia_file,
            no_orphans=no_orphans,
            norm=norm,
            prop=prop,
            max_terms=max_terms,
            th_step=th_step,
            n_cpu=n_threads
        )

        # Write results
        logger.info(f"Writing results to {output_dir}")
        write_results(df, dfs_best, out_dir=output_dir, th_step=th_step)

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("CAFA Evaluation Summary")
        logger.info("=" * 60)

        if dfs_best:
            for metric, metric_df in dfs_best.items():
                if not metric_df.empty:
                    logger.info(f"\n{metric}:")
                    logger.info(metric_df.to_string())

        return df, dfs_best

    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")


def main(argv=None):
    """Main entry point for CAFA evaluation.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:])
    """
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)

    # Determine ontology path
    if args.ontology:
        ontology_file = args.ontology
    else:
        # Default to project's go.obo file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        ontology_file = os.path.join(project_root, "data", "go.obo")

    if not os.path.exists(ontology_file):
        logger.error(f"Ontology file not found: {ontology_file}")
        sys.exit(1)

    if not os.path.exists(args.predictions):
        logger.error(f"Predictions file not found: {args.predictions}")
        sys.exit(1)

    if not os.path.exists(args.ground_truth):
        logger.error(f"Ground truth file not found: {args.ground_truth}")
        sys.exit(1)

    if args.ia and not os.path.exists(args.ia):
        logger.error(f"IA file not found: {args.ia}")
        sys.exit(1)

    # Run evaluation
    try:
        run_cafa_evaluation(
            predictions_file=args.predictions,
            gt_file=args.ground_truth,
            ontology_file=ontology_file,
            output_dir=args.output,
            ia_file=args.ia,
            th_step=args.th_step,
            prop=args.prop,
            norm=args.norm,
            no_orphans=args.no_orphans,
            max_terms=args.max_terms,
            n_threads=args.threads
        )
        logger.info("CAFA evaluation completed successfully!")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
