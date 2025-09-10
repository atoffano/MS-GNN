import pandas as pd
import requests
import gzip
import shutil
import os

# Download and extract InterPro data
ipr_gz_url = (
    "https://ftp.ebi.ac.uk/pub/databases/interpro/releases/106.0/protein2ipr.dat.gz"
)
ipr_gz_path = "data/swissprot/protein2ipr.dat.gz"
ipr_path = "data/swissprot/protein2ipr.dat"

if not os.path.exists(ipr_path):
    print("Downloading protein2ipr.dat.gz...")
    r = requests.get(ipr_gz_url, stream=True)
    with open(ipr_gz_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Extracting protein2ipr.dat...")
    with gzip.open(ipr_gz_path, "rb") as f_in, open(ipr_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    print("Done.")

# Load valid protein IDs from FASTA (lines starting with '>')
fasta_path = "data/swissprot/2024_01/swissprot_2024_01.fasta"
valid_ids = set()
with open(fasta_path) as f:
    for line in f:
        if line.startswith(">"):
            valid_ids.add(line.split()[0][1:])

# Parse protein2ipr.dat, keeping only rows with ID in valid_ids
cols = ["ID", "IPR", "desc", "db", "start", "end"]
ipr_df = pd.read_csv(ipr_path, sep="\t", header=None, names=cols, dtype=str)

filtered_ipr_df = ipr_df[ipr_df["ID"].isin(valid_ids)]

filtered_ipr_df.to_csv("data/swissprot/swissprot_interpro.tsv", sep="\t", index=False)
os.remove(ipr_gz_path)
os.remove(ipr_path)
