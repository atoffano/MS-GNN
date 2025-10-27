import os
import gzip
import requests
import pandas as pd
from pathlib import Path
import tqdm


def get_stringdb(
    base_dir,
):
    stringdb_url = (
        "https://stringdb-downloads.org/download/stream/protein.links.detailed.v12.0.onlyAB.tsv.gz",
    )
    mapping_path = base_dir / "idmapping_swissprot_stringdb.tsv"
    output_path = base_dir / "swissprot_stringdb.tsv"
    # Download STRINGdb file
    gz_path = base_dir / "protein.links.detailed.v12.0.onlyAB.tsv.gz"
    # if not os.path.exists(gz_path):
    #     print("Downloading STRINGdb data...")
    #     with requests.get(stringdb_url, stream=True) as r:
    #         r.raise_for_status()
    #         with open(gz_path, "wb") as f:
    #             for chunk in r.iter_content(chunk_size=8192):
    #                 f.write(chunk)
    # else:
    #     print("STRINGdb data already downloaded.")

    # Load SwissProt-STRINGdb mapping
    print("Loading SwissProt-STRINGdb mapping...")
    mapping_df = pd.read_csv(mapping_path, sep="\t")
    stringdb_ids = set(mapping_df["To"].astype(str))

    # Prepare output
    print("Filtering STRINGdb data...")
    with gzip.open(gz_path, "rt") as fin, open(output_path, "w") as fout:
        header = fin.readline()
        fout.write(header)
        for line in tqdm.tqdm(fin, desc="Filtering lines"):
            cols = line.rstrip("\n").split("\t")
            if cols[0] in stringdb_ids or cols[1] in stringdb_ids:
                fout.write(line)
    print(f"Filtered data saved to {output_path}")


if __name__ == "__main__":
    get_stringdb(
        Path(
            "/lustre/fsn1/projects/rech/dqy/uki62ne/tempdata/PFP_layer/data/swissprot/2024_01"
        )
    )
# Example usage:
# get_stringdb()
