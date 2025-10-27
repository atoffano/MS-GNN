# PFP_layer - Protein Function Prediction with Graph Neural Networks

A deep learning framework for protein function prediction using heterogeneous Graph Neural Networks (GNNs). This project leverages protein structure, sequence embeddings, and protein-protein interactions to predict Gene Ontology (GO) terms across multiple ontologies.

## Features

- **Heterogeneous Graph Neural Networks**: Uses GAT (Graph Attention Networks) to model relationships between proteins and amino acids
- **Multi-level Protein Representation**: Combines:
  - ESM (Evolutionary Scale Modeling) embeddings for residue-level features
  - Protein structure information from ESMFold
  - InterPro domain annotations
  - Protein-protein alignments
- **GO Term Prediction**: Supports prediction across three Gene Ontology subontologies:
  - Molecular Function Ontology (MFO)
  - Biological Process Ontology (BPO)
  - Cellular Component Ontology (CCO)
- **Flexible Training**: Train on SwissProt or custom datasets with experimental or computational annotations
- **Interpretability**: Includes attention mechanism visualization and Captum-based attribution analysis

## Installation

### Prerequisites

- Python 3.12+
- CUDA 12.4+ (for GPU support)
- Conda or Miniconda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/atoffano/PFP_layer.git
cd PFP_layer
```

2. Create the conda environment:
```bash
conda env create -f pyg.yml
conda activate pyg2
```

3. Install PyTorch Geometric dependencies:
```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
```

## Data Preparation

### 1. Download SwissProt InterPro Annotations

Download InterPro annotations from:
```
https://ftp.ebi.ac.uk/pub/databases/interpro/releases/106.0/protein2ipr.dat.gz
```

### 2. Get Protein Structures from ESMFold

Generate protein structures for your dataset:
```bash
python -m src.data.get_structures --fasta data/swissprot/2024_01/swissprot_2024_01.fasta --local
```

### 3. Preprocess Data

The preprocessing pipeline includes:
- Protein alignment computation (DIAMOND)
- Graph construction with protein and amino acid nodes
- ESM embedding generation
- InterPro annotation integration

See `src/data/` directory for specific preprocessing scripts.

## Usage

### Training

Train the model using a configuration file:

```bash
python main.py --config src/configs/cfg.yaml
```

For a quick test with toy data:
```bash
python main.py --config src/configs/toy_cfg.yaml
```

### Configuration

Edit `src/configs/cfg.yaml` to customize:

```yaml
model:
  hidden_channels: 256
  batch_size: 32

trainer:
  epochs: 8
  device: cuda
  compile: true  # Use torch.compile for optimization

data:
  dataset: 'D1'
  subontology: ['MFO']  # Choose from MFO, BPO, CCO
  train_on_swissprot: False
  exp_only: true  # Use only experimental annotations

run:
  save_predictions:
    val: true
    test: true
  save_model: true
```

### Evaluation

The evaluation pipeline automatically computes:
- Area Under Precision-Recall Curve (AUPR)
- F-max score
- Precision-Recall curves

Results are saved to `./results/<dataset>/<run_id>/`

## Model Architecture

The model uses a heterogeneous graph structure with:

**Node Types:**
- `protein`: Protein-level nodes with aggregated features
- `aa`: Amino acid residue nodes with ESM embeddings

**Edge Types:**
- `aa -> belongs_to -> protein`: Links amino acids to their parent protein
- `protein -> aligned_with -> protein`: Protein-protein sequence alignments
- (Optional) `aa -> close_to -> aa`: Spatial proximity in protein structure
- (Optional) `protein -> stringdb -> protein`: STRING database interactions

**Architecture:**
1. Input projection layer (HeteroDictLinear)
2. Two GAT convolutional layers with attention mechanism
3. Skip connections with concatenation
4. Output layer for GO term prediction

## Project Structure

```
PFP_layer/
├── main.py                 # Main training script
├── pyg.yml                 # Conda environment specification
├── src/
│   ├── configs/           # Configuration files
│   ├── data/              # Data preprocessing scripts
│   ├── models/            # GNN model definitions
│   └── utils/             # Utility functions (evaluation, visualization, etc.)
├── notebooks/             # Jupyter notebooks for analysis
└── data/                  # Data directory (not tracked in git)
```

## Monitoring

Training progress is logged to [Weights & Biases](https://wandb.ai/):
- Training/validation loss
- AUPR and F-max metrics
- Precision-Recall curves
- Model checkpoints

## Citation

If you use this code in your research, please cite:

```bibtex
@software{pfp_layer,
  title = {PFP_layer: Protein Function Prediction with Graph Neural Networks},
  author = {Toffano, A.},
  year = {2024},
  url = {https://github.com/atoffano/PFP_layer}
}
```

## License

This project is available for academic and research purposes.

## Acknowledgments

- ESM embeddings from [fair-esm](https://github.com/facebookresearch/esm)
- PyTorch Geometric for graph neural network utilities
- InterPro database for protein domain annotations
- SwissProt/UniProt for protein annotations