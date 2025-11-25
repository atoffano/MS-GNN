"""Information Accretion (IA) computation for GO annotations.

This module computes Information Accretion values for Gene Ontology terms
based on annotation frequencies. IA values are used for weighted evaluation
metrics in CAFA-style assessments.

The IA value measures how much additional information a term provides
compared to its parent terms in the ontology.

Note: IA should be computed using the same ontology version that was used
for term propagation. Otherwise, this may result in negative IA values.
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
from scipy.sparse import dok_matrix
import tqdm

try:
    import obonet
except ImportError:
    print("Please install obonet: pip install obonet")
    sys.exit(1)


def obsolete_terms(ontology):
    """Returns obsolete terms info from an ontology.

    Args:
        ontology: Path to OBO file or obonet graph object

    Returns:
        Tuple of (obsolete_set, old_to_new_dict):
            - obsolete_set: Set of obsolete terms without replacements
            - old_to_new_dict: Dict mapping obsolete terms to their replacements
    """
    graph_with_obs = (
        obonet.read_obo(ontology, ignore_obsolete=False)
        if isinstance(ontology, str)
        else ontology
    )
    print(f"Number of terms: {len(graph_with_obs)}")
    old_to_new = dict()
    obsolete = set()
    for node, data in graph_with_obs.nodes(data=True):
        for replaced_by in data.get("replaced_by", []):
            old_to_new[node] = replaced_by
        if data.get("is_obsolete", False) and node not in old_to_new.keys():
            obsolete.add(node)

    return obsolete, old_to_new


def clean_ontology_edges(ontology):
    """Remove non-standard ontology edges.

    Keeps only "is_a" and "part_of" edges and removes cross-ontology edges.

    Args:
        ontology: Ontology graph (networkx DiGraph or MultiDiGraph)

    Returns:
        Cleaned ontology graph
    """
    # keep only "is_a" and "part_of" edges (All the "regulates" edges are in BPO)
    remove_edges = [
        (i, j, k) for i, j, k in ontology.edges if not (k == "is_a" or k == "part_of")
    ]

    ontology.remove_edges_from(remove_edges)

    # There should not be any cross-ontology edges, but we verify here
    crossont_edges = [
        (i, j, k)
        for i, j, k in ontology.edges
        if ontology.nodes[i]["namespace"] != ontology.nodes[j]["namespace"]
    ]
    if len(crossont_edges) > 0:
        ontology.remove_edges_from(crossont_edges)

    return ontology


def fetch_aspect(ontology, root: str):
    """Return a subgraph of an ontology for a specific aspect.

    Args:
        ontology: Ontology graph (networkx DiGraph or MultiDiGraph)
        root: Root GO term for the aspect (e.g., GO:0008150 for BPO)

    Returns:
        Subgraph containing only nodes of the specified namespace
    """
    namespace = ontology.nodes[root]["namespace"]
    aspect_nodes = [
        n for n, v in ontology.nodes(data=True) if v["namespace"] == namespace
    ]
    subont_ = ontology.subgraph(aspect_nodes)
    return subont_


def propagate_terms(terms_df, subontologies):
    """Propagate terms in DataFrame according to ontology structure.

    Args:
        terms_df: DataFrame with columns 'EntryID', 'term', 'aspect'
        subontologies: Dict of ontology aspects (networkx graphs)

    Returns:
        DataFrame with propagated terms
    """
    # Look up ancestors ahead of time for efficiency
    subont_terms = {
        aspect: set(terms_df[terms_df.aspect == aspect].term.values)
        for aspect in subontologies.keys()
    }
    ancestor_lookup = {
        aspect: {
            t: nx.descendants(subont, t) for t in subont_terms[aspect] if t in subont
        }
        for aspect, subont in subontologies.items()
    }

    propagated_terms = []
    grouped = terms_df.groupby(["EntryID", "aspect"])
    for (protein, aspect), entry_df in tqdm.tqdm(
        grouped,
        desc="Propagating terms",
        total=grouped.ngroups,
    ):
        # Only include terms that are in the ancestor lookup
        valid_terms = [t for t in set(entry_df.term.values) if t in ancestor_lookup[aspect]]
        if valid_terms:
            protein_terms = set().union(
                *[list(ancestor_lookup[aspect][t]) + [t] for t in valid_terms]
            )
            propagated_terms += [
                {"EntryID": protein, "term": t, "aspect": aspect} for t in protein_terms
            ]

    return pd.DataFrame(propagated_terms)


def term_counts(terms_df, term_indices):
    """Count instances of each term in the annotation data.

    Args:
        terms_df: DataFrame with propagated annotated terms
        term_indices: Dict mapping terms to column indices

    Returns:
        Sparse matrix of term counts per protein
    """
    num_proteins = len(terms_df.groupby("EntryID"))
    S = dok_matrix((num_proteins + 1, len(term_indices)), dtype=np.int32)
    S[-1, :] = 1  # dummy protein

    for i, (protein, protdf) in enumerate(terms_df.groupby("EntryID")):
        row_count = {term_indices[t]: c for t, c in Counter(protdf["term"]).items()}
        for col, count in row_count.items():
            S[i, col] = count

    return S


def calc_ia(term, count_matrix, ontology, terms_index):
    """Calculate Information Accretion for a single term.

    IA is computed as -log2(P(term) / P(parents)), which measures how much
    additional information a term provides compared to its parents.

    Args:
        term: GO term ID
        count_matrix: Sparse matrix of term counts
        ontology: Ontology subgraph
        terms_index: Dict mapping terms to column indices

    Returns:
        IA value for the term
    """
    parents = nx.descendants_at_distance(ontology, term, 1)

    # count of proteins with term
    prots_with_term = count_matrix[:, terms_index[term]].sum()

    # count of proteins with all parents - only consider parents in the index
    valid_parents = [p for p in parents if p in terms_index]
    num_parents = len(valid_parents)
    
    if num_parents == 0:
        # Root term or no valid parents
        return 0
    
    prots_with_parents = (
        count_matrix[:, [terms_index[p] for p in valid_parents]].sum(1) == num_parents
    ).sum()

    # avoid floating point errors by returning exactly zero
    if prots_with_term == prots_with_parents:
        return 0

    return -np.log2(prots_with_term / prots_with_parents)


def parse_inputs(argv):
    """Parse command line arguments.

    Args:
        argv: Command line arguments

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Compute Information Accretion of GO annotations. "
                    "Note: If annotations in input file have been propagated to ontology roots, "
                    "the input ontology graph should be the same as the one used to propagate terms."
    )

    parser.add_argument(
        "--annot", "-a",
        required=True,
        help="Path to annotation file"
    )

    parser.add_argument(
        "--dataset", "-d",
        default=None,
        help="Dataset name used to load test proteins. If empty, no filtering is applied."
    )

    parser.add_argument(
        "--ontology", "-go",
        default=None,
        help="Path to OBO ontology graph file. If empty, current OBO will be downloaded."
    )

    parser.add_argument(
        "--prop", "-p",
        action="store_true",
        help="Flag to propagate terms in annotation file according to the ontology graph"
    )

    parser.add_argument(
        "--aspect", "-asp",
        default=None,
        choices=["BPO", "CCO", "MFO"],
        help="Compute IA for terms in this aspect only. If empty, all aspects are computed."
    )

    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path. If empty, saves to data/<dataset>/IC_<dataset>[_<aspect>].tsv"
    )

    return parser.parse_args(argv)


def compute_ia(
    annotation_file,
    output_file=None,
    ontology_path=None,
    dataset=None,
    aspect=None,
    propagate=False
):
    """Compute Information Accretion values for GO terms.

    Args:
        annotation_file: Path to annotation TSV file with columns EntryID, term
        output_file: Path to save IA values (optional)
        ontology_path: Path to OBO file (optional, defaults to downloading)
        dataset: Dataset name for filtering test proteins (optional)
        aspect: Ontology aspect to compute IA for (optional, BPO/CCO/MFO)
        propagate: Whether to propagate terms (default: False)

    Returns:
        DataFrame with columns 'term' and 'ic'
    """
    # Load annotation data
    annotation_df = pd.read_csv(annotation_file, sep="\t")
    annotation_df = annotation_df[["EntryID", "term"]]

    # Load test proteins if dataset specified
    test_df = None
    if dataset and aspect:
        test_file = f"./data/{dataset}/{dataset}_{aspect}_test_annotations.tsv"
        if os.path.exists(test_file):
            test_df = pd.read_csv(
                test_file,
                sep="\t",
                header=None,
                names=["EntryID", "term"],
            )
            # Remove test proteins from IC computation
            annotation_df = annotation_df[
                ~annotation_df["EntryID"].isin(test_df["EntryID"])
            ]
        else:
            print(f"Warning: Test file not found: {test_file}")

    # Load ontology
    if ontology_path:
        obo_path = ontology_path
    else:
        obo_path = "http://purl.obolibrary.org/obo/go/go.obo"

    print(f"Loading ontology from {obo_path}...")
    ontology_graph = clean_ontology_edges(
        obonet.read_obo(obo_path, ignore_obsolete=False)
    )

    roots = {"BPO": "GO:0008150", "CCO": "GO:0005575", "MFO": "GO:0003674"}
    subontologies = {
        asp: fetch_aspect(ontology_graph, roots[asp]) for asp in roots
    }

    aspect_map = {
        "BPO": list(subontologies["BPO"].nodes),
        "CCO": list(subontologies["CCO"].nodes),
        "MFO": list(subontologies["MFO"].nodes),
    }
    obsolete, old_to_new = obsolete_terms(ontology_graph)

    # Reverse aspect dictionary
    term_to_aspect = {go: asp for asp, go_list in aspect_map.items() for go in go_list}

    annotation_df = annotation_df.dropna(subset=["term"])

    # Handle multiple terms per cell - ensure term is string before split
    annotation_df["term"] = annotation_df["term"].astype(str).apply(
        lambda x: x.split("; ") if x and x != "nan" else []
    )
    annotation_df = annotation_df[annotation_df["term"].apply(lambda x: len(x) > 0)]
    annotation_df = annotation_df[["EntryID", "term"]].explode("term")
    annotation_df["term"] = annotation_df["term"].map(lambda x: old_to_new.get(x, x))
    annotation_df = annotation_df[~annotation_df["term"].isin(obsolete)]
    annotation_df["aspect"] = annotation_df["term"].map(term_to_aspect)
    annotation_df = annotation_df.dropna(subset=["aspect"])

    print(f"Loaded {len(annotation_df)} annotations")

    # Filter by aspect if specified
    if aspect:
        subontologies = {aspect: subontologies[aspect]}
        print(f"Computing IA for aspect {aspect}")

    if propagate:
        print("Propagating Terms...")
        annotation_df = propagate_terms(annotation_df, subontologies)

    # Count term instances
    print("Counting Terms...")
    aspect_counts = dict()
    aspect_terms = dict()
    term_idx = dict()
    for asp, subont in subontologies.items():
        aspect_terms[asp] = sorted(subont.nodes)  # ensure same order
        term_idx[asp] = {t: i for i, t in enumerate(aspect_terms[asp])}
        aspect_counts[asp] = term_counts(
            annotation_df[annotation_df.aspect == asp], term_idx[asp]
        )

        assert aspect_counts[asp].sum() == len(
            annotation_df[annotation_df.aspect == asp]
        ) + len(aspect_terms[asp])

    # Convert to CSC format for efficient column indexing
    sp_matrix = {asp: dok.tocsc() for asp, dok in aspect_counts.items()}

    # Compute IA
    print("Computing Information Accretion...")
    aspect_ia = {
        asp: {t: 0 for t in aspect_terms[asp]} for asp in aspect_terms.keys()
    }
    for asp, subontology in subontologies.items():
        for term in tqdm.tqdm(aspect_ia[asp].keys(), desc=f"Computing IA for {asp}"):
            aspect_ia[asp][term] = calc_ia(
                term, sp_matrix[asp], subontology, term_idx[asp]
            )

    ia_df = pd.concat(
        [
            pd.DataFrame.from_dict(
                {
                    "term": aspect_ia[asp].keys(),
                    "ic": aspect_ia[asp].values(),
                    "aspect": asp,
                }
            )
            for asp in subontologies.keys()
        ]
    )

    negative_ia_terms = ia_df[ia_df["ic"] < 0]
    if not negative_ia_terms.empty:
        print("Warning: The following terms have negative IA values:")
        print(negative_ia_terms)
        print(
            "This usually happens when there is a mismatch between ontology versions "
            "used for propagation and IA computation."
        )

    # Verify all counts are non-negative
    if ia_df["ic"].min() < 0:
        print("Warning: Some IA values are negative. Check ontology version consistency.")

    # Save to file
    if output_file:
        out_path = output_file
    elif dataset:
        if aspect:
            out_path = f"./data/{dataset}/IC_{dataset}_{aspect}.tsv"
        else:
            out_path = f"./data/{dataset}/IC_{dataset}.tsv"
    else:
        out_path = "./ia_output.tsv"

    print(f"Saving to file {out_path}")
    ia_df[["term", "ic"]].to_csv(out_path, header=None, sep="\t", index=False)

    return ia_df


if __name__ == "__main__":
    args = parse_inputs(sys.argv[1:])
    compute_ia(
        annotation_file=args.annot,
        output_file=args.output,
        ontology_path=args.ontology,
        dataset=args.dataset,
        aspect=args.aspect,
        propagate=args.prop
    )


