import os
import numpy as np
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from cafaeval.parser import obo_parser, gt_parser, pred_parser
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())


# Return a mask for all the predictions (matrix) >= tau
def solidify_prediction(pred, tau):
    return pred >= tau


# computes the f metric for each precision and recall in the input arrays
def compute_f(pr, rc):
    n = 2 * pr * rc
    d = pr + rc
    return np.divide(n, d, out=np.zeros_like(n, dtype=float), where=d != 0)


def compute_s(ru, mi):
    return np.sqrt(ru**2 + mi**2)
    # return np.where(np.isnan(ru), mi, np.sqrt(ru + np.nan_to_num(mi)))


def compute_confusion_matrix(tau_arr, g, pred, n_gt, ic_arr=None):
    """
    Perform the evaluation at the matrix level for all tau thresholds
    The calculation is
    """
    # n, tp, fp, fn, pr, rc (fp = misinformation, fn = remaining uncertainty)
    metrics = np.zeros((len(tau_arr), 6), dtype="float")

    for i, tau in enumerate(tau_arr):

        # Filter predictions based on tau threshold
        p = solidify_prediction(pred, tau)

        # Terms subsets
        intersection = np.logical_and(p, g)  # TP
        mis = np.logical_and(
            p, np.logical_not(g)
        )  # FP, predicted but not in the ground truth
        remaining = np.logical_and(
            np.logical_not(p), g
        )  # FN, not predicted but in the ground truth

        # Weighted evaluation
        if ic_arr is not None:
            p = p * ic_arr
            intersection = intersection * ic_arr  # TP
            mis = mis * ic_arr  # FP, predicted but not in the ground truth
            remaining = remaining * ic_arr  # FN, not predicted but in the ground truth

        n_pred = p.sum(axis=1)  # TP + FP
        n_intersection = intersection.sum(axis=1)  # TP

        # Number of proteins with at least one term predicted with score >= tau
        metrics[i, 0] = (p.sum(axis=1) > 0).sum()

        # Sum of confusion matrices
        metrics[i, 1] = n_intersection.sum()  # TP
        metrics[i, 2] = mis.sum(axis=1).sum()  # FP
        metrics[i, 3] = remaining.sum(axis=1).sum()  # FN

        # Macro-averaging
        metrics[i, 4] = np.divide(
            n_intersection,
            n_pred,
            out=np.zeros_like(n_intersection, dtype="float"),
            where=n_pred > 0,
        ).sum()  # Precision
        metrics[i, 5] = np.divide(
            n_intersection, n_gt, out=np.zeros_like(n_gt, dtype="float"), where=n_gt > 0
        ).sum()  # Recall

    return metrics


def compute_metrics(pred, gt, tau_arr, toi, ic_arr=None, n_cpu=0):
    """
    Takes the prediction and the ground truth and for each threshold in tau_arr
    calculates the confusion matrix and returns the coverage,
    precision, recall, remaining uncertainty and misinformation.
    Toi is the list of terms (indexes) to be considered
    """
    if n_cpu == 0:
        n_cpu = mp.cpu_count()

    columns = ["n", "tp", "fp", "fn", "pr", "rc"]

    # Slice once and reuse (shared by all threads)
    g = gt.matrix[:, toi]
    pred_sub = pred.matrix[:, toi]
    w = None if ic_arr is None else ic_arr[toi]

    # Precompute n_gt once
    n_gt = g.sum(axis=1) if w is None else (g * w).sum(axis=1)

    # Don’t start more workers than chunks
    n_workers = min(n_cpu, len(tau_arr))
    tau_chunks = np.array_split(tau_arr, n_workers)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        parts = list(
            ex.map(
                compute_confusion_matrix,
                tau_chunks,
                repeat(g),
                repeat(pred_sub),
                repeat(n_gt),
                repeat(w),
            )
        )

    metrics = np.concatenate(parts, axis=0)
    return pd.DataFrame(metrics, columns=columns)


def compute_per_protein_metrics(pred, gt, tau, toi, ic_arr=None, ns=None):
    """
    Compute precision and recall for each individual protein at a given tau threshold.

    Args:
        pred: Prediction object with matrix attribute
        gt: Ground truth object with matrix attribute
        tau: Score threshold (e.g., 0.5)
        toi: Terms of interest (indices)
        ic_arr: Information content array for weighted metrics (optional)
        ns: Namespace (optional)

    Returns:
        DataFrame with per-protein precision, recall, and F-score
    """
    # Filter predictions based on tau threshold
    p = pred.matrix[:, toi] >= tau
    g = gt.matrix[:, toi]

    # Per-protein calculations
    if ic_arr is not None:
        # Weighted metrics
        p_weighted = p * ic_arr[toi]
        g_weighted = g * ic_arr[toi]

        tp = (np.logical_and(p, g) * ic_arr[toi]).sum(axis=1)
        fp = (np.logical_and(p, np.logical_not(g)) * ic_arr[toi]).sum(axis=1)
        fn = (np.logical_and(np.logical_not(p), g) * ic_arr[toi]).sum(axis=1)

        n_pred = p_weighted.sum(axis=1)
        n_gt = g_weighted.sum(axis=1)
    else:
        # Unweighted metrics
        tp = np.logical_and(p, g).sum(axis=1)
        fp = np.logical_and(p, np.logical_not(g)).sum(axis=1)
        fn = np.logical_and(np.logical_not(p), g).sum(axis=1)

        n_pred = p.sum(axis=1)
        n_gt = g.sum(axis=1)

    # Calculate precision and recall per protein
    precision = np.divide(
        tp, n_pred, out=np.zeros_like(tp, dtype=float), where=n_pred > 0
    )
    recall = np.divide(tp, n_gt, out=np.zeros_like(tp, dtype=float), where=n_gt > 0)
    f_score = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )

    # Build DataFrame with protein indices
    df = pd.DataFrame(
        {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "n_pred": n_pred,
            "n_gt": n_gt,
            "precision": precision,
            "recall": recall,
            "f": f_score,
        }
    )

    # Add protein IDs if available
    if hasattr(pred, "ids"):
        df["protein_id"] = list(pred.ids.keys())

    # Add context columns to ensure data is identifiable
    if ns is not None:
        df["ns"] = ns
    df["tau"] = tau

    return df


def normalize(metrics, ns, tau_arr, ne, normalization):

    # Normalize columns
    for column in metrics.columns:
        if column != "n":
            # By default normalize by gt
            denominator = ne
            # Otherwise normalize by pred
            if normalization == "pred" or (normalization == "cafa" and column == "pr"):
                denominator = metrics["n"]
            metrics[column] = np.divide(
                metrics[column],
                denominator,
                out=np.zeros_like(metrics[column], dtype="float"),
                where=denominator > 0,
            )

    metrics["ns"] = [ns] * len(tau_arr)
    metrics["tau"] = tau_arr
    metrics["cov"] = metrics["n"] / ne
    metrics["mi"] = metrics["fp"]
    metrics["ru"] = metrics["fn"]

    metrics["f"] = compute_f(metrics["pr"], metrics["rc"])
    metrics["s"] = compute_s(metrics["ru"], metrics["mi"])

    # Micro-average, calculation is based on the average of the confusion matrices
    metrics["pr_micro"] = np.divide(
        metrics["tp"],
        metrics["tp"] + metrics["fp"],
        out=np.zeros_like(metrics["tp"], dtype="float"),
        where=(metrics["tp"] + metrics["fp"]) > 0,
    )
    metrics["rc_micro"] = np.divide(
        metrics["tp"],
        metrics["tp"] + metrics["fn"],
        out=np.zeros_like(metrics["tp"], dtype="float"),
        where=(metrics["tp"] + metrics["fn"]) > 0,
    )
    metrics["f_micro"] = compute_f(metrics["pr_micro"], metrics["rc_micro"])

    return metrics


def evaluate_prediction(
    prediction, gt, ontologies, tau_arr, normalization="cafa", n_cpu=0, compute_pp=False
):

    dfs = []
    dfs_w = []
    dfs_per_protein_unweighted = []
    dfs_per_protein_weighted = []

    for ns in prediction:
        ne = np.full(len(tau_arr), gt[ns].matrix[:, ontologies[ns].toi].shape[0])
        df_ns = normalize(
            compute_metrics(
                prediction[ns], gt[ns], tau_arr, ontologies[ns].toi, None, n_cpu
            ),
            ns,
            tau_arr,
            ne,
            normalization,
        )
        dfs.append(df_ns)

        if compute_pp:
            # Find best tau for unweighted f
            best_idx = df_ns["f"].idxmax()
            best_tau = df_ns.loc[best_idx, "tau"]

            dfs_per_protein_unweighted.append(
                compute_per_protein_metrics(
                    pred=prediction[ns],
                    gt=gt[ns],
                    tau=best_tau,
                    toi=ontologies[ns].toi,
                    ic_arr=None,
                    ns=ns,
                )
            )

        if ontologies[ns].ia is not None:
            ne = np.full(len(tau_arr), gt[ns].matrix[:, ontologies[ns].toi_ia].shape[0])
            df_ns_w = normalize(
                compute_metrics(
                    prediction[ns],
                    gt[ns],
                    tau_arr,
                    ontologies[ns].toi_ia,
                    ontologies[ns].ia,
                    n_cpu,
                ),
                ns,
                tau_arr,
                ne,
                normalization,
            )
            dfs_w.append(df_ns_w)

            if compute_pp:
                # Find best tau for weighted f
                best_idx_w = df_ns_w["f"].idxmax()
                best_tau_w = df_ns_w.loc[best_idx_w, "tau"]

                df_pp_w = compute_per_protein_metrics(
                    pred=prediction[ns],
                    gt=gt[ns],
                    tau=best_tau_w,
                    toi=ontologies[ns].toi_ia,
                    ic_arr=ontologies[ns].ia,
                    ns=ns,
                )
                # Mark these columns as weighted
                df_pp_w = df_pp_w.add_suffix("_w")
                # Restore key columns
                for col in ["protein_id", "ns", "tau"]:
                    if f"{col}_w" in df_pp_w.columns:
                        df_pp_w.rename(columns={f"{col}_w": col}, inplace=True)

                # Rename f_w to f_w (already done by suffix)
                # But we want to ensure tau is distinct if we merge?
                # Actually, we will merge on protein_id and ns.
                # So we should rename tau to tau_w in the weighted dataframe.
                df_pp_w.rename(columns={"tau": "tau_w"}, inplace=True)

                dfs_per_protein_weighted.append(df_pp_w)

    dfs = pd.concat(dfs)

    if dfs_w:
        dfs_w = pd.concat(dfs_w)
        dfs = pd.merge(dfs, dfs_w, on=["ns", "tau"], suffixes=("", "_w"))

    df_per_protein = pd.DataFrame()
    if compute_pp:
        if dfs_per_protein_unweighted:
            df_pp = pd.concat(dfs_per_protein_unweighted, ignore_index=True)

            if dfs_per_protein_weighted:
                df_pp_w = pd.concat(dfs_per_protein_weighted, ignore_index=True)
                # Merge on protein_id and ns
                # Note: unweighted has 'tau', weighted has 'tau_w'
                df_per_protein = pd.merge(
                    df_pp, df_pp_w, on=["protein_id", "ns"], how="outer"
                )
            else:
                df_per_protein = df_pp
        elif dfs_per_protein_weighted:
            df_per_protein = pd.concat(dfs_per_protein_weighted, ignore_index=True)

    return dfs, df_per_protein


def cafa_eval(
    obo_file,
    pred_dir,
    gt_file,
    ia=None,
    no_orphans=False,
    norm="cafa",
    prop="max",
    max_terms=None,
    th_step=0.01,
    n_cpu=1,
    compute_pp=False,
):

    # Tau array, used to compute metrics at different score thresholds
    tau_arr = np.arange(th_step, 1, th_step)

    # Parse the OBO file and creates a different graphs for each namespace
    ontologies = obo_parser(obo_file, ("is_a", "part_of"), ia, not no_orphans)

    # Parse ground truth file
    gt = gt_parser(gt_file, ontologies)

    # Set prediction files looking recursively in the prediction folder
    pred_folder = os.path.normpath(pred_dir) + "/"  # add the tailing "/"
    pred_files = []
    for root, dirs, files in os.walk(pred_folder):
        for file in files:
            pred_files.append(os.path.join(root, file))
    logging.debug("Prediction paths {}".format(pred_files))

    # Parse prediction files and perform evaluation
    dfs = []
    dfs_pp_all = []  # Store per-protein dataframes

    for file_name in pred_files:
        prediction = pred_parser(file_name, ontologies, gt, prop, max_terms)
        if not prediction:
            logging.warning("Prediction: {}, not evaluated".format(file_name))
        else:
            # Unpack the two return values
            df_pred, df_pp = evaluate_prediction(
                prediction,
                gt,
                ontologies,
                tau_arr,
                normalization=norm,
                n_cpu=n_cpu,
                compute_pp=compute_pp,
            )

            clean_filename = file_name.replace(pred_folder, "").replace("/", "_")

            df_pred["filename"] = clean_filename
            dfs.append(df_pred)

            if not df_pp.empty:
                df_pp["filename"] = clean_filename
                dfs_pp_all.append(df_pp)

            logging.info("Prediction: {}, evaluated".format(file_name))

    # Concatenate all dataframes and save them
    df = None
    df_per_protein_final = None  # Final per-protein dataframe
    dfs_best = {}

    if dfs:
        df = pd.concat(dfs)
        if dfs_pp_all:
            df_per_protein_final = pd.concat(dfs_pp_all, ignore_index=True)

        # Remove rows with no coverage
        df = df[df["cov"] > 0].reset_index(drop=True)
        df.set_index(["filename", "ns", "tau"], inplace=True)

        # Calculate the best index for each namespace and each evaluation metric
        for metric, cols in [
            ("f", ["rc", "pr"]),
            ("f_w", ["rc_w", "pr_w"]),
            ("s", ["ru", "mi"]),
            ("f_micro", ["rc_micro", "pr_micro"]),
            ("f_micro_w", ["rc_micro_w", "pr_micro_w"]),
        ]:
            if metric in df.columns:
                index_best = (
                    df.groupby(level=["filename", "ns"])[metric].idxmax()
                    if metric in ["f", "f_w", "f_micro", "f_micro_w"]
                    else df.groupby(["filename", "ns"])[metric].idxmin()
                )
                df_best = df.loc[index_best]
                if metric[-2:] != "_w":
                    df_best["cov_max"] = (
                        df.reset_index("tau")
                        .loc[[ele[:-1] for ele in index_best]]
                        .groupby(level=["filename", "ns"])["cov"]
                        .max()
                    )
                else:
                    df_best["cov_max"] = (
                        df.reset_index("tau")
                        .loc[[ele[:-1] for ele in index_best]]
                        .groupby(level=["filename", "ns"])["cov_w"]
                        .max()
                    )
                dfs_best[metric] = df_best
    else:
        logging.info("No predictions evaluated")

    # Return the per-protein dataframe as well
    return df, dfs_best, df_per_protein_final


def write_results(df, dfs_best, df_per_protein=None, out_dir="results", th_step=0.01):

    # Create output folder here in order to store the log file
    out_folder = os.path.normpath(out_dir) + "/"
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    # Set the number of decimals to write in the output files based on the threshold step size
    decimals = int(np.ceil(-np.log10(th_step))) + 1

    df.to_csv(
        "{}/evaluation_all.tsv".format(out_folder),
        float_format="%.{}f".format(decimals),
        sep="\t",
    )

    for metric in dfs_best:
        dfs_best[metric].to_csv(
            "{}/evaluation_best_{}.tsv".format(out_folder, metric),
            float_format="%.{}f".format(decimals),
            sep="\t",
        )

    if df_per_protein is not None:
        df_per_protein.to_csv(
            "{}/evaluation_per_protein.tsv".format(out_folder),
            float_format="%.{}f".format(decimals),
            sep="\t",
            index=False,
        )
