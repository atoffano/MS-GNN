# Bridging Scales: A Multi-Level Graph Neural Network for Protein Function Prediction

**MS-GNN** is a deep learning framework for protein function prediction in the form of Gene Ontology terms. It utilizes a graph neural network architecture based on attention mechanisms to learn simultaneously form both residue-level 3D structures and global protein–protein association networks.

This repository contains the source code, training configurations, and evaluation logic for modeling protein functions across all Gene Ontology (GO) sub-ontologies (MFO, BPO, CCO). It accompanies the paper: *"[MS-GNN: A Multi-Scale Graph Neural Network for Protein Function Prediction](https://hal.science/hal-05580207)."*

Predicted annotations for all SwissProt proteins can be found in this repository under `predictions`.

## Architecture Highlights

![MS-GNN Architecture](gnn_model.png)
*Overview of the multi-scale graph architecture, detailing the flow of features between individual amino acid nodes and the global protein--protein interaction network. **a.** Global information flow through the model. **b**. Residue-scale information propagation during GAT layers along edges between residues determined by contact maps (yellow edges) and between residues and global protein node (dotted red edges). **c.** Protein network scale information propagation during GAT layers between neighbors based on homology (red edges) and STRING associations (blue edges).*

MS-GNN integrates heterogeneous data at two scales:
1. **Protein-Level Graph (Atomic Scale):** Each protein is modeled as a spatial graph where nodes correspond to amino acids initialized with pre-trained protein language model (pLM ESM-1b) embeddings. Edges connect residues within 10Å of each other based on AlphaFold/ESMFold predictions.
2. **Network-Level Graph (Systemic Scale):** Proteins are embedded into a broader systemic network, with edges weighted by sequence similarity (DIAMOND) and functional associations (STRING database).

A specialized central node for each protein aggregates functional signatures (InterPro, GO) and connects both levels, which allows information to propagate between local structural motifs and broader protein-protein associations via Graph Attention Networks (GAT).

## Installation

### Prerequisites
- Python 3.12+
- CUDA 12.4
- Conda or Miniconda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/atoffano/MS-GNN.git
cd MS-GNN
```

2. Create the conda environment:
```bash
conda env create -f msgnn.yml
conda activate msgnn
```

3. Install PyTorch Geometric dependencies:
```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
```

## Data Preparation

### 1. Download Datasets & AlphaFold Structures
A preprocessing script automatically downloads AlphaFold structures corresponding to SwissProt targets and fetches functional annotations (e.g., InterPro), plus network interactions from STRING database:
```bash
python -m src.data.gather_data
```

*(Note: While MS-GNN supports ESMFold predictions to fill gaps for missing sequences, you can currently skip the ESMFold structure generation for a faster initial setup.)*

### 2. Generate ESM-1b Residue Embeddings
Generate the pre-trained amino acid language embeddings using the ESM-1b model:
```bash
python -m src.data.embed_residues --fasta data/swissprot/2024_01/swissprot_2024_01.fasta
```

### 3. Compute Alignments
To enable homology context, compute DIAMOND protein alignments (requires `diamond` executable):
```bash
diamond makedb --in data/swissprot/2024_01/swissprot_2024_01.fasta -d data/swissprot/2024_01/swissprot_2024_01
diamond blastp -q data/swissprot/2024_01/swissprot_2024_01.fasta -d data/swissprot/2024_01/swissprot_2024_01 -o data/swissprot/2024_01/swissprot_2024_01_alignment.tsv --more-sensitive -e 1e-3
```

### 4. Construct Multi-level Graphs
Compile data into PyTorch Geometric hetero-graph structures:
```bash
python -m src.data.create_graphs --fasta data/swissprot/2024_01/swissprot_2024_01.fasta --num-workers 16
```

## Usage

### Training
Train the model by pointing to a specific configuration file:

```bash
# Standard training run
python main.py --config src/configs/cfg.yaml
```

Modify `src/configs/cfg.yaml` to configure target `dataset` (e.g., `D1`, `ATGO`), specify the `subontology` (`MFO`, `BPO`, or `CCO`), and toggle between experimental logic (`exp_only: true`) vs. curated labels paradigms.

### Evaluation and Inference
To generate predictions from a trained model run and automatically evaluate against the testing set:

```bash
# Run prediction and save outputs
python -m src.utils.predict \
    --input_dir results/D1/your_training_run_directory_here \
    --subontology MFO \
    --device cuda

# Complete CAFA/BeProf benchmark metrics are computed during train cycles or manually via evaluation scripts
python -m src.utils.evaluation \
    --input_dir results/D1/your_training_run_directory_here \
    --dataset D1 \
    --subontology MFO
```

## Data availability
Benchmark datasets and evaluation scripts rely on original published sources from `cafa-eval` and original dataset providers.

## Acknowledgements
We thank the UniProt consortium, the CAFA community, and developers of open-source bioinformatics tools used throughout our analysis. Computing HPC resources were provided by the Grand équipement national de calcul intensif at IDRIS [grant 2024-AD011012511R3]. Funding support from the French National Agency for Research through grant DIG-AI [ANR-22-CE23-0012] is gratefully acknowledged.
