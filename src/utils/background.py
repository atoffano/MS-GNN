import pandas as pd
import argparse
import pickle
import os


def parse_terms(terms_str):
    # Convert the string representation of a list to an actual list using ast.literal_eval
    try:
        return set(terms_str.split("; "))
    except Exception as e:
        print(f"Error parsing terms: {terms_str}\n{e}")
        return set()


def load_file(annot_path, test_path):
    """
    Loads a TSV file with columns: EntryID, term
    Returns a dictionary mapping protein IDs to a set of GO terms for the given ontology.
    """
    df = pd.read_csv(annot_path, sep="\t")
    test_proteins = pd.read_csv(test_path, sep="\t")
    test_proteins = set(test_proteins["EntryID"].tolist())
    prot_dict = {}
    for _, row in df.iterrows():
        prot = row["EntryID"]
        if (
            prot not in test_proteins
        ):  # Proteins waiting for annotation do not contribute to IC calculation
            terms = parse_terms(row["term"])
            prot_dict[prot] = terms
    return prot_dict


def main():
    parser = argparse.ArgumentParser(
        description="Merge CCO, BPO, MFO training files into background file format"
    )
    parser.add_argument(
        "--test_cco",
        default=None,
        help="Path to CCO test proteins file (format: tsv with EntryID column containing protein IDs)",
    )
    parser.add_argument(
        "--test_bpo",
        default=None,
        help="Path to BPO test proteins file (format: tsv with EntryID column containing protein IDs)",
    )
    parser.add_argument(
        "--test_mfo",
        default=None,
        help="Path to MFO test proteins file (format: tsv with EntryID column containing protein IDs)",
    )
    parser.add_argument("--cco", default=None, help="Path to CCO_train.tsv")
    parser.add_argument("--bpo", default=None, help="Path to BPO_train.tsv")
    parser.add_argument("--mfo", default=None, help="Path to MFO_train.tsv")
    parser.add_argument("--output", required=True, help="Output pickle file path")
    args = parser.parse_args()

    cco_dict, mfo_dict, bpo_dict = {}, {}, {}
    if args.cco is not None:
        print(f"Loading CCO file: {args.cco}")
        cco_dict = load_file(args.cco, args.test_cco)
    if args.bpo is not None:
        print(f"Loading BPO file: {args.bpo}")
        bpo_dict = load_file(args.bpo, args.test_bpo)
    if args.mfo is not None:
        print(f"Loading MFO file: {args.mfo}")
        mfo_dict = load_file(args.mfo, args.test_mfo)

    # Create a union of all protein IDs from the three files
    all_proteins = set(cco_dict.keys()) | set(bpo_dict.keys()) | set(mfo_dict.keys())
    result = {}
    for prot in all_proteins:
        result[prot] = {
            "all_bp": bpo_dict.get(prot, set()),
            "all_cc": cco_dict.get(prot, set()),
            "all_mf": mfo_dict.get(prot, set()),
        }

    output_dir = args.output.rsplit("/", 1)[0]
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    # Save the merged dictionary as pandas df
    with open(args.output, "wb") as f:
        pickle.dump(result, f)
    print(f"Merged background file saved to: {args.output}")


if __name__ == "__main__":
    main()

# Example usage:
# python background.py --cco ./data/swissprot/2024_01/swissprot_2024_01_CCO_annotations.tsv --bpo ./data/swissprot/2024_01/swissprot_2024_01_BPO_annotations.tsv --mfo ./data/swissprot/2024_01/swissprot_2024_01_MFO_annotations.tsv --output ./data/H30/background_H30.pkl --test_cco ./data/H30/H30_CCO_test_annotations.tsv --test_bpo ./data/H30/H30_BPO_test_annotations.tsv --test_mfo ./data/H30/H30_MFO_test_annotations.tsv
