import os
import torch
import tqdm

protein_graphs_dir = "/lustre/fsn1/projects/rech/dqy/uki62ne/tempdata/PFP_layer/data/swissprot/2024_01/protein_graphs"
no_structure_fasta = "/lustre/fsn1/projects/rech/dqy/uki62ne/tempdata/PFP_layer/data/swissprot/2024_01/no_structure_in_graph.fasta"
could_not_load_fasta = "/lustre/fsn1/projects/rech/dqy/uki62ne/tempdata/PFP_layer/data/swissprot/2024_01/could_not_load.fasta"


def write_fasta(records, out_path):
    with open(out_path, "w") as f:
        for name, seq in records:
            f.write(f">{name}\n{seq}\n")


no_structure = []
could_not_load = []

for fname in tqdm.tqdm(os.listdir(protein_graphs_dir)):
    if not fname.endswith(".pt"):
        continue
    protein_name = fname.replace(".pt", "")
    fpath = os.path.join(protein_graphs_dir, fname)
    try:
        data = torch.load(fpath)
    except Exception as e:
        print(f"Could not load {protein_name} because: {e}")
        could_not_load.append((protein_name, ""))
        continue

    edge_index = data["aa", "close_to", "aa"]
    # If empty dictionary, skip
    if not edge_index or edge_index.edge_index.size(1) == 0:
        no_structure.append((protein_name, data["protein"].sequence))
        print(f"No structure in grpah for {protein_name}")

write_fasta(no_structure, no_structure_fasta)
write_fasta(could_not_load, could_not_load_fasta)
