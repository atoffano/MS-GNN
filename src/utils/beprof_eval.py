"""BEPROF evaluation metrics for protein function prediction.

This module implements evaluation metrics compatible with the BEPROF (Benchmark of
Protein Function) framework. It includes functions for computing F-max, AUPR, and
other performance metrics for multi-label GO term prediction, along with utilities
for handling GO term hierarchies and information content.

It supports:
- Reading predictions from TSV or PKL files
- Converting ground truth annotations for BEPROF evaluation
- Computing IC (Information Content) weights
- Running full BEPROF evaluation with metrics like F-max, S-min, AUPR, ICAupr, DPAupr
"""

import warnings
import numpy as np
import scipy.sparse as ssp
import math
import pandas as pd
from collections import OrderedDict, deque, Counter, defaultdict
import pickle as pkl
import os
import sys
import argparse
import logging
import tqdm

from src.utils.constants import (
    GO_ROOT_TERMS,
    GO_BIOLOGICAL_PROCESS,
    GO_MOLECULAR_FUNCTION,
    GO_CELLULAR_COMPONENT,
    GO_FUNC_DICT,
    GO_NAMESPACES,
    GO_NAMESPACES_REVERT,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

EXP_CODES = {
    "EXP",
    "IDA",
    "IPI",
    "IMP",
    "IGI",
    "IEP",
    "TAS",
    "IC",
}

CAFA_TARGETS = {
    "10090",
    "223283",
    "273057",
    "559292",
    "85962",
    "10116",
    "224308",
    "284812",
    "7227",
    "9606",
    "160488",
    "237561",
    "321314",
    "7955",
    "99287",
    "170187",
    "243232",
    "3702",
    "83333",
    "208963",
    "243273",
    "44689",
    "8355",
}


def parse_args(argv=None):
    """Parse command-line arguments for BEPROF evaluation.

    Args:
        argv: Command line arguments list (defaults to sys.argv[1:])

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Evaluate protein function predictions using BEPROF metrics."
    )

    parser.add_argument(
        "--predictions",
        "-p",
        dest="predict",
        required=True,
        help="Path to prediction file (TSV or PKL format)",
    )

    parser.add_argument(
        "--ground_truth",
        "-gt",
        dest="true",
        required=True,
        help="Path to ground truth annotation file (TSV or PKL format)",
    )

    parser.add_argument(
        "--background",
        "-b",
        required=True,
        help="Path to background annotation file (PKL format)",
    )

    parser.add_argument(
        "--ontology",
        "-go",
        dest="go",
        default=None,
        help="Path to OBO ontology file. If empty, uses ./data/go.obo",
    )

    parser.add_argument(
        "--output",
        "-o",
        dest="output_path",
        default=None,
        help="Output directory for evaluation results. If not specified, will be derived from predictions path.",
    )

    parser.add_argument(
        "--metrics",
        "-m",
        default="0,1,2,3,4,5",
        help="Comma-separated list of metric indices: 0=F_max, 1=Smin, 2=Aupr, 3=ICAupr, 4=DPAupr, 5=threshold. Default: 0,1,2,3,4,5",
    )

    parser.add_argument(
        "--subontology",
        "-s",
        default=None,
        help="Subontology to evaluate (BPO, CCO, MFO). If not provided, inferred from prediction filename.",
    )

    if argv is None:
        return parser.parse_args()
    return parser.parse_args(argv)


__all__ = [
    "fmax",
    "aupr",
    "ROOT_GO_TERMS",
    "compute_performance",
    "run_beprof_evaluation",
    "gt_convert",
    "convert_predictions",
    "derive_output_dir_from_predictions",
    "read_pkl",
    "save_pkl",
]

# Re-export constants for backwards compatibility
ROOT_GO_TERMS = GO_ROOT_TERMS


def gt_convert(gt_tsv):
    """Convert ground truth TSV file to pickle format for BEPROF evaluation.

    Args:
        gt_tsv: Path to ground truth TSV file with columns: EntryID, term

    Returns:
        Path to the generated pickle file
    """
    # Extract the ontology type (BPO, CCO, MFO) from the filename
    base_name = os.path.basename(gt_tsv)
    ontology_type = base_name.split("_")[1]  # Extract BPO, CCO, or MFO
    ontology_mapping = {"BPO": "all_bp", "CCO": "all_cc", "MFO": "all_mf"}
    ontology_key = ontology_mapping[ontology_type]

    # Output filename: same as input but with .pkl extension
    gt_pkl = os.path.splitext(gt_tsv)[0] + ".pkl"

    logger.info(f"Converting {base_name} to pickle format...")

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

    protein_go_terms_dict = dict(protein_go_terms)

    # Save to pickle file
    with open(gt_pkl, "wb") as f:
        pkl.dump(protein_go_terms_dict, f)
    logger.info(f"Saved pickle file: {gt_pkl}")

    return gt_pkl


def convert_predictions(pred_file, subontology):
    """Convert a TSV prediction file to pickle format for BEPROF evaluation.

    Args:
        pred_file: Path to prediction TSV file with columns: target_ID, term_ID, score
        subontology: Subontology type (BPO, CCO, MFO)

    Returns:
        Dictionary with protein predictions in BEPROF format
    """
    subontology = subontology[:2].lower()
    df = pd.read_csv(pred_file, sep="\t")
    # Split term column by '; ' and explode
    df["term_ID"] = df["term_ID"].str.split("; ")
    df = df.explode("term_ID")
    pred_dict = {}
    for prot, group in tqdm.tqdm(df.groupby("target_ID"), desc="Converting predictions"):
        pred_dict[prot] = {
            f"{subontology}": dict(zip(group["term_ID"], group["score"]))
        }
    return pred_dict


def derive_output_dir_from_predictions(predictions_file, subontology=None, split=None):
    """Derive output directory from predictions file path.

    Args:
        predictions_file: Path to the predictions file (TSV or PKL)
        subontology: Optional subontology (MFO, BPO, CCO) - inferred from filename if not provided
        split: Optional data split (train, val, test) - inferred from filename if not provided

    Returns:
        str: Path to output directory for BEPROF evaluation results
    """
    predictions_path = os.path.abspath(predictions_file)
    predictions_dir = os.path.dirname(predictions_path)
    predictions_filename = os.path.basename(predictions_file)

    # Extract split and subontology from filename if not provided
    # Expected format: predictions_{split}_{subontology}.tsv or .pkl
    if split is None or subontology is None:
        try:
            parts = predictions_filename.replace(".tsv", "").replace(".pkl", "").split("_")
            if len(parts) >= 3:
                if split is None:
                    split = parts[1]  # e.g., "test"
                if subontology is None:
                    subontology = parts[2]  # e.g., "MFO"
        except (IndexError, AttributeError):
            # Filename doesn't match expected format, use fallback
            pass

    # Navigate up from predictions directory to results directory
    # Structure: {results_dir}/predictions/predictions_{split}_{subontology}.tsv
    if os.path.basename(predictions_dir) == "predictions":
        results_dir = os.path.dirname(predictions_dir)
    else:
        results_dir = predictions_dir

    if split is not None and subontology is not None:
        output_dir = os.path.join(
            results_dir, "evaluation", f"{split}_{subontology}", "beprof-eval"
        )
    else:
        output_dir = os.path.join(results_dir, "evaluation", "beprof-eval")

    return output_dir


def fmax(go, targets, scores, idx_goid):
    """Calculate F-max score with information content weighting.

    Args:
        go: GO ontology object with IC calculation methods
        targets: Target annotations matrix
        scores: Prediction scores matrix
        idx_goid: List mapping indices to GO term IDs

    Returns:
        Tuple of F-max metrics including IC-weighted and depth-weighted versions
    """
    targets = ssp.csr_matrix(targets)

    fmax_ = 0.0, 0.0, 0.0
    precisions = []
    recalls = []
    icprecisions = []
    icrecalls = []
    dpprecisions = []
    dprecalls = []
    mi_values = []
    ru_values = []
    goic_list = []
    godp_list = []
    for i in range(len(idx_goid)):
        goic_list.append(go.get_ic(idx_goid[i]))
    for i in range(len(idx_goid)):
        godp_list.append(go.get_icdepth(idx_goid[i]))
    goic_vector = np.array(goic_list).reshape(-1, 1)
    godp_vector = np.array(godp_list).reshape(-1, 1)

    for cut in (c / 100 for c in range(101)):
        cut_sc = ssp.csr_matrix((scores >= cut).astype(np.int32))
        correct = cut_sc.multiply(targets).sum(axis=1)
        correct_sc = cut_sc.multiply(targets)
        fp_sc = cut_sc - correct_sc
        fn_sc = targets - correct_sc

        correct_ic = ssp.csr_matrix(correct_sc.dot(goic_vector))
        cut_ic = ssp.csr_matrix(cut_sc.dot(goic_vector))
        targets_ic = ssp.csr_matrix(targets.dot(goic_vector))

        correct_dp = ssp.csr_matrix(correct_sc.dot(godp_vector))
        cut_dp = ssp.csr_matrix(cut_sc.dot(godp_vector))
        targets_dp = ssp.csr_matrix(targets.dot(godp_vector))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, r = correct / cut_sc.sum(axis=1), correct / targets.sum(axis=1)
            p, r = np.average(p[np.invert(np.isnan(p))]), np.average(r)

            mi = fp_sc.dot(goic_vector).sum(axis=0)
            ru = fn_sc.dot(goic_vector).sum(axis=0)
            mi /= len(targets.sum(axis=1))
            ru /= len(targets.sum(axis=1))

            # Store the mi and ru values for this threshold
            mi_values.append(float(mi))
            ru_values.append(float(ru))

            icp, icr = correct_ic / cut_ic, correct_ic / targets_ic
            icp, icr = np.average(icp[np.invert(np.isnan(icp))]), np.average(icr)

            dpp, dpr = correct_dp / cut_dp, correct_dp / targets_dp
            dpp, dpr = np.average(dpp[np.invert(np.isnan(dpp))]), np.average(dpr)

        if np.isnan(p):
            precisions.append(0.0)
            recalls.append(r)
        else:
            precisions.append(p)
            recalls.append(r)

        if np.isnan(icp):
            icprecisions.append(0.0)
            icrecalls.append(icr)
        else:
            icprecisions.append(icp)
            icrecalls.append(icr)

        if np.isnan(dpp):
            dpprecisions.append(0.0)
            dprecalls.append(dpr)
        else:
            dpprecisions.append(dpp)
            dprecalls.append(dpr)

        try:
            fmax_ = max(
                fmax_,
                (
                    2 * p * r / (p + r) if p + r > 0.0 else 0.0,
                    math.sqrt(ru * ru + mi * mi),
                    cut,
                ),
            )
        except ZeroDivisionError:
            pass

    # Add endpoints when dealing with undefined regions
    for rec, prec in [
        (recalls, precisions),
        (icrecalls, icprecisions),
        (dprecalls, dpprecisions),
    ]:
        if rec[0] > 0:
            rec = np.concatenate(([0], rec))
            prec = np.concatenate(([1], prec))
        if rec[-1] < 1:
            rec = np.concatenate((rec, [1.0]))
            prec = np.concatenate((prec, [0.0]))
        # Assign back to original variables
        if rec is recalls:
            recalls, precisions = rec, prec
        elif rec is icrecalls:
            icrecalls, icprecisions = rec, prec
        else:
            dprecalls, dpprecisions = rec, prec

    return (
        fmax_[0],
        fmax_[1],
        fmax_[2],
        precisions,
        recalls,
        icprecisions,
        icrecalls,
        dpprecisions,
        dprecalls,
        mi_values,
        ru_values,
        goic_vector,
        godp_vector,
    )


def read_pkl(pklfile):
    """Load data from a pickle file.

    Args:
        pklfile: Path to pickle file

    Returns:
        Unpickled data object
    """
    with open(pklfile, "rb") as fr:
        data = pkl.load(fr)
    return data


def save_pkl(pklfile, data):
    """Save data to a pickle file.

    Args:
        pklfile: Path to output pickle file
        data: Data object to pickle
    """
    with open(pklfile, "wb") as fw:
        pkl.dump(data, fw)


# Re-export constants for backwards compatibility
BIOLOGICAL_PROCESS = GO_BIOLOGICAL_PROCESS
MOLECULAR_FUNCTION = GO_MOLECULAR_FUNCTION
CELLULAR_COMPONENT = GO_CELLULAR_COMPONENT
FUNC_DICT = GO_FUNC_DICT
NAMESPACES = GO_NAMESPACES
NAMESPACES_REVERT = GO_NAMESPACES_REVERT


def is_cafa_target(org):
    return org in CAFA_TARGETS


def is_exp_code(code):
    return code in EXP_CODES


class Ontology(object):
    def __init__(self, filename, with_rels=False):
        self.ont = self.load(filename, with_rels)
        self.ic = None
        self.icdepth = None

    def has_term(self, term_id):
        return term_id in self.ont

    def calculate_ic(self, annots):
        cnt = Counter()
        for x in annots:
            cnt.update(x)
        self.ic = {}
        self.icdepth = {}
        for go_id, n in cnt.items():
            parents = self.get_parents(go_id)
            if len(parents) == 0:
                min_n = n
            else:
                min_n = min([cnt[x] for x in parents])
            self.ic[go_id] = math.log(min_n / n, 2)
            self.icdepth[go_id] = (
                math.log(
                    self.get_depth(go_id, NAMESPACES_REVERT[self.get_namespace(go_id)]),
                    2,
                )
                * self.ic[go_id]
            )

    def get_ic(self, go_id):
        if self.ic is None:
            raise Exception("Not yet calculated")
        if go_id not in self.ic:
            return 0.0
        return self.ic[go_id]

    def get_icdepth(self, go_id):
        if self.icdepth is None:
            raise Exception("Not yet calculated")
        if go_id not in self.icdepth:
            return 0.0
        return self.icdepth[go_id]

    def load(self, filename, with_rels):
        ont = dict()
        obj = None
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == "[Term]":
                    if obj is not None:
                        ont[obj["id"]] = obj
                    obj = dict()
                    obj["is_a"] = list()
                    obj["part_of"] = list()
                    obj["regulates"] = list()
                    obj["alt_ids"] = list()
                    obj["is_obsolete"] = False
                    continue
                elif line == "[Typedef]":
                    obj = None
                else:
                    if obj is None:
                        continue
                    l = line.split(": ")
                    if l[0] == "id":
                        obj["id"] = l[1]
                    elif l[0] == "alt_id":
                        obj["alt_ids"].append(l[1])
                    elif l[0] == "namespace":
                        obj["namespace"] = l[1]
                    elif l[0] == "is_a":
                        obj["is_a"].append(l[1].split(" ! ")[0])
                    elif with_rels and l[0] == "relationship":
                        it = l[1].split()
                        # add all types of relationships
                        if it[0] == "part_of":
                            obj["is_a"].append(it[1])

                    elif l[0] == "name":
                        obj["name"] = l[1]
                    elif l[0] == "is_obsolete" and l[1] == "true":
                        obj["is_obsolete"] = True
        if obj is not None:
            ont[obj["id"]] = obj
        for term_id in list(ont.keys()):
            for t_id in ont[term_id]["alt_ids"]:
                ont[t_id] = ont[term_id]
            if ont[term_id]["is_obsolete"]:
                del ont[term_id]
        for term_id, val in ont.items():
            if "children" not in val:
                val["children"] = set()
            for p_id in val["is_a"]:
                if p_id in ont:
                    if "children" not in ont[p_id]:
                        ont[p_id]["children"] = set()
                    ont[p_id]["children"].add(term_id)
        return ont

    def get_anchestors(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]["is_a"]:
                    if parent_id in self.ont:
                        q.append(parent_id)
        return term_set

    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]["is_a"]:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set

    def get_depth(self, term_id, ont):
        q = deque()
        q.append(term_id)
        layer = 1
        while len(q) > 0:
            all_p = set()
            while len(q) > 0:
                t_id = q.popleft()
                p_id = self.get_parents(t_id)
                all_p.update(p_id)
            if all_p:
                layer += 1
                for item in all_p:
                    if item == FUNC_DICT[ont]:
                        return layer
                    q.append(item)
        return layer

    def get_namespace_terms(self, namespace):
        terms = set()
        for go_id, obj in self.ont.items():
            if obj["namespace"] == namespace:
                terms.add(go_id)
        return terms

    def get_namespace(self, term_id):
        return self.ont[term_id]["namespace"]

    def get_term_set(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for ch_id in self.ont[t_id]["children"]:
                    q.append(ch_id)
        return term_set


def compute_performance(test_df, go, ont, output_path):

    go_set = go.get_namespace_terms(NAMESPACES[ont])
    go_set.remove(FUNC_DICT[ont])
    print(len(go_set))

    labels = list(go_set)
    goid_idx = {}
    idx_goid = {}
    for idx, goid in enumerate(labels):
        goid_idx[goid] = idx
        idx_goid[idx] = goid

    pred_scores = []
    true_scores = []
    # Annotations
    for i, row in enumerate(test_df.itertuples()):
        # true
        true_vals = [0] * len(labels)
        annots = set()
        for go_id in row.gos:
            if go.has_term(go_id):
                annots |= go.get_anchestors(go_id)
        for go_id in annots:
            if go_id in go_set:
                true_vals[goid_idx[go_id]] = 1

        # pred
        pred_vals = [-1] * len(labels)
        for items, score in row.predictions.items():
            if items in go_set:
                pred_vals[goid_idx[items]] = max(score, pred_vals[goid_idx[items]])
            go_parent = go.get_anchestors(items)
            for go_id in go_parent:
                if go_id in go_set:
                    pred_vals[goid_idx[go_id]] = max(pred_vals[goid_idx[go_id]], score)

        # Only keep proteins with at least one valid annotation
        if sum(true_vals) > 0:
            true_scores.append(true_vals)
            pred_scores.append(pred_vals)
        else:
            print(
                f"Skipping protein {row.protein_id}: no valid annotations in ontology."
            )

    pred_scores = np.array(pred_scores)
    true_scores = np.array(true_scores)
    # print(
    #     pred_scores.shape, true_scores.shape, sum(pred_scores < 0), sum(pred_scores > 0)
    # )

    (
        result_fmax,
        result_smin,
        result_t,
        precisions,
        recalls,
        icprecisions,
        icrecalls,
        dpprecisions,
        dprecalls,
        mi_values,
        ru_values,
        goic_vector,
        godp_vector,
    ) = fmax(go, true_scores, pred_scores, idx_goid)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    result_aupr = np.trapz(precisions, recalls)

    icprecisions = np.array(icprecisions)
    icrecalls = np.array(icrecalls)
    sorted_index = np.argsort(icrecalls)
    icrecalls = icrecalls[sorted_index]
    icprecisions = icprecisions[sorted_index]
    result_icaupr = np.trapz(icprecisions, icrecalls)

    dpprecisions = np.array(dpprecisions)
    dprecalls = np.array(dprecalls)
    sorted_index = np.argsort(dprecalls)
    dprecalls = dprecalls[sorted_index]
    dpprecisions = dpprecisions[sorted_index]
    result_dpaupr = np.trapz(dpprecisions, dprecalls)

    # Save recalls, precisions, icprecisions, icrecalls, dpprecisions, dprecalls, result_aupr, result_icaupr, result_dpaupr, ru_values, mi_values to dict and pickle
    save_dict = {
        "ontology": ont,
        "recalls": recalls,
        "precisions": precisions,
        "icprecisions": icprecisions,
        "icrecalls": icrecalls,
        "dpprecisions": dpprecisions,
        "dprecalls": dprecalls,
        "result_aupr": result_aupr,
        "result_icaupr": result_icaupr,
        "result_dpaupr": result_dpaupr,
        "ru_values": ru_values,
        "mi_values": mi_values,
        "goic_vector": goic_vector,
        "godp_vector": godp_vector,
        "result_fmax": result_fmax,
        "result_smin": result_smin,
        "result_t": result_t,
    }
    save_pkl(
        "{0}/beprof_eval_results.pkl".format(output_path),
        save_dict,
    )
    print(f"Saved detailed evaluation results to {output_path}/beprof_eval_results.pkl")
    print(f"F_max: {result_fmax:.2f}")
    print(f"S_min: {result_smin:.2f}")
    print(f"Aupr: {result_aupr:.2f}")
    print(f"ICAupr: {result_icaupr:.2f}")
    print(f"DPAupr: {result_dpaupr:.2f}")
    # return result_fmax, result_smin, result_aupr, result_icaupr, result_dpaupr, result_t


def generate_result(
    input_file,
    output_path,
    go_file,
    real_test_protein_mess,
    all_protein_information,
    metrics,
):
    all_files = {}
    all_files["Your_method"] = input_file
    go = Ontology(go_file, with_rels=True)

    all_annotations = []
    for prot, ann in tqdm.tqdm(
        all_protein_information.items(), desc="Computing ancestors"
    ):
        combined_terms = set()
        for terms in ann.values():
            combined_terms |= terms
        item_set = set()
        for item in combined_terms:
            if go.has_term(item):
                item_set |= go.get_anchestors(item)
        all_annotations.append(list(item_set))
    go.calculate_ic(all_annotations)

    if "CCO" in input_file:
        all_tags = ["cc"]
    elif "BPO" in input_file:
        all_tags = ["bp"]
    elif "MFO" in input_file:
        all_tags = ["mf"]
    else:
        all_tags = ["cc", "bp", "mf"]
    all_results = OrderedDict()
    all_results["methods"] = []
    all_results["methods"].append(math.nan)
    for m in all_files.keys():
        all_results["methods"].append(m)
    all_metrics = ["F_max", "Smin", "Aupr", "ICAupr", "DPAupr", "threadhold"]
    metric_list = []
    for metric in metrics:
        metric_list.append(all_metrics[int(metric)])

    for evas in metric_list:
        for num, tag in enumerate(all_tags):
            all_results["{0}_{1}".format(evas, num)] = []
            all_results["{0}_{1}".format(evas, num)].append(tag)

    for num, tag in enumerate(all_tags):
        for method, mfile in all_files.items():
            save_dict = {}
            save_dict["protein_id"] = []
            save_dict["gos"] = []
            save_dict["predictions"] = []

            with open(mfile, "rb") as fr:
                method_predict_result = pkl.load(fr)

            for protein, val in method_predict_result.items():
                if real_test_protein_mess[protein]["all_{0}".format(tag)] == set():
                    continue
                if tag not in method_predict_result[protein]:
                    method_predict_result[protein][tag] = {}

                save_dict["protein_id"].append(protein)
                save_dict["gos"].append(
                    real_test_protein_mess[protein]["all_{0}".format(tag)]
                )
                save_dict["predictions"].append(method_predict_result[protein][tag])

            df = pd.DataFrame(save_dict)
            compute_performance(df, go, tag, output_path)

    #         F_max, Smin, Aupr, ICAupr, DPAupr, threadhold = compute_performance(
    #             df, go, tag, output_path
    #         )

    #         if "F_max" in metric_list:
    #             all_results["{0}_{1}".format("F_max", num)].append(round(F_max, 5))
    #         if "Smin" in metric_list:
    #             all_results["{0}_{1}".format("Smin", num)].append(round(Smin, 5))
    #         if "Aupr" in metric_list:
    #             all_results["{0}_{1}".format("Aupr", num)].append(round(Aupr, 5))
    #         if "ICAupr" in metric_list:
    #             all_results["{0}_{1}".format("ICAupr", num)].append(round(ICAupr, 5))
    #         if "DPAupr" in metric_list:
    #             all_results["{0}_{1}".format("DPAupr", num)].append(round(DPAupr, 5))

    #         print(
    #             "Have done",
    #             method,
    #             tag,
    #             "F_max:",
    #             F_max,
    #             "Smin:",
    #             Smin,
    #             "Aupr:",
    #             Aupr,
    #             "ICAupr:",
    #             ICAupr,
    #             "DPAupr:",
    #             DPAupr,
    #             "threadhold:",
    #             threadhold,
    #         )

    # df = pd.DataFrame(all_results)
    # df.to_csv("{0}/eval_beprof_test_evaluation_results.csv".format(output_path))


def run_beprof_evaluation(
    predictions_file,
    gt_file,
    background_file,
    ontology_file,
    output_dir,
    subontology=None,
    metrics="0,1,2,3,4,5",
):
    """Run BEPROF evaluation on predictions.

    This function provides a programmatic interface for running BEPROF evaluation,
    similar to how run_cafa_evaluation works in cafa_evaluation.py.

    Args:
        predictions_file: Path to prediction file (TSV or PKL format)
        gt_file: Path to ground truth annotation file (TSV or PKL format)
        background_file: Path to background annotation file (PKL format)
        ontology_file: Path to GO ontology file (OBO format)
        output_dir: Output directory for results
        subontology: Subontology type (BPO, CCO, MFO). If None, inferred from filename.
        metrics: Comma-separated list of metric indices (default: "0,1,2,3,4,5")

    Returns:
        Path to output directory containing evaluation results
    """
    # Infer subontology from filename if not provided
    if subontology is None:
        base_name = os.path.basename(predictions_file)
        if "CCO" in base_name:
            subontology = "CCO"
        elif "BPO" in base_name:
            subontology = "BPO"
        elif "MFO" in base_name:
            subontology = "MFO"
        else:
            raise ValueError(
                f"Could not infer subontology from filename: {base_name}. "
                "Please provide subontology parameter."
            )

    logger.info(f"Running BEPROF evaluation for subontology: {subontology}")
    logger.info(f"  Predictions: {predictions_file}")
    logger.info(f"  Ground Truth: {gt_file}")
    logger.info(f"  Background: {background_file}")
    logger.info(f"  Ontology: {ontology_file}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert predictions to PKL if TSV
    if predictions_file.endswith(".tsv"):
        logger.info("Converting predictions TSV to PKL format...")
        pred_dict = convert_predictions(predictions_file, subontology)
        pred_pkl = os.path.join(output_dir, "predictions.pkl")
        with open(pred_pkl, "wb") as f:
            pkl.dump(pred_dict, f)
        predictions_file = pred_pkl
        logger.info(f"Converted predictions saved to {pred_pkl}")

    # Convert ground truth to PKL if TSV
    if gt_file.endswith(".tsv"):
        gt_pkl = os.path.splitext(gt_file)[0] + ".pkl"
        if not os.path.exists(gt_pkl):
            logger.info("Converting ground truth TSV to PKL format...")
            gt_convert(gt_file)
        gt_file = gt_pkl

    # Parse metrics
    if isinstance(metrics, str):
        metrics_list = metrics.strip().split(",")
    else:
        metrics_list = metrics

    # Load data
    logger.info("Loading ground truth and background data...")
    with open(gt_file, "rb") as f:
        test_data = pkl.load(f)
    with open(background_file, "rb") as f:
        all_protein_information = pkl.load(f)
    logger.info("Test data and all protein information loaded.")

    # Run evaluation
    generate_result(
        predictions_file,
        output_dir,
        ontology_file,
        test_data,
        all_protein_information,
        metrics_list,
    )

    logger.info(f"BEPROF evaluation completed. Results saved to: {output_dir}")
    return output_dir


def main_cli(
    input_file,
    output_path,
    test_data_file,
    all_protein_information_file,
    go_file,
    metrics,
):
    """Legacy main function for backward compatibility.

    Args:
        input_file: Path to prediction PKL file
        output_path: Output directory path
        test_data_file: Path to ground truth PKL file
        all_protein_information_file: Path to background PKL file
        go_file: Path to GO ontology file
        metrics: List of metric indices to compute
    """
    with open(test_data_file, "rb") as f:
        test_data = pkl.load(f)
    with open(all_protein_information_file, "rb") as f:
        all_protein_information = pkl.load(f)
    logger.info("Test data and all protein information loaded.")
    generate_result(
        input_file, output_path, go_file, test_data, all_protein_information, metrics
    )


def main(argv=None):
    """Main entry point for BEPROF evaluation CLI.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:])
    """
    args = parse_args(argv)

    # Determine ontology path
    if args.go:
        ontology_file = args.go
    else:
        # Default to project's go.obo file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        ontology_file = os.path.join(project_root, "data", "go.obo")

    if not os.path.exists(ontology_file):
        logger.error(f"Ontology file not found: {ontology_file}")
        sys.exit(1)

    if not os.path.exists(args.predict):
        logger.error(f"Predictions file not found: {args.predict}")
        sys.exit(1)

    if not os.path.exists(args.true):
        logger.error(f"Ground truth file not found: {args.true}")
        sys.exit(1)

    if not os.path.exists(args.background):
        logger.error(f"Background file not found: {args.background}")
        sys.exit(1)

    # Determine output directory
    if args.output_path:
        output_dir = args.output_path
    else:
        output_dir = derive_output_dir_from_predictions(
            args.predict, subontology=args.subontology
        )
        logger.info(f"Output directory derived from predictions path: {output_dir}")

    # Run evaluation
    try:
        run_beprof_evaluation(
            predictions_file=args.predict,
            gt_file=args.true,
            background_file=args.background,
            ontology_file=ontology_file,
            output_dir=output_dir,
            subontology=args.subontology,
            metrics=args.metrics,
        )
        logger.info("BEPROF evaluation completed successfully!")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
