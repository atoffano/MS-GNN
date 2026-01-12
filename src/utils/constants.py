"""Constants and default paths for the inference preprocessing pipeline.

This module centralizes all configuration constants, file paths, and default
parameters used throughout the inference pipeline for protein function prediction.
"""

from pathlib import Path

# ============================================================================
# Root Directories
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
SRC_ROOT = PROJECT_ROOT / "src"

# ============================================================================
# Data Constants
# ============================================================================
USES_ENTRYID = ["H30", "D1"]

# SwissProt
SWISSPROT_ROOT = DATA_ROOT / "swissprot" / "2024_01"
SWISSPROT_FASTA = SWISSPROT_ROOT / "swissprot_2024_01.fasta"
PID_MAPPING = SWISSPROT_ROOT / "swissprot_2024_01_annotations.tsv"
DIAMOND_ALIGNMENT = SWISSPROT_ROOT / "diamond_swissprot_2024_01_alignment.tsv"

# InterPro
INTERPRO_TSV = SWISSPROT_ROOT / "swissprot_interpro_106_0.tsv"
INTERPRO_VOCAB = SWISSPROT_ROOT / "protein_graphs" / "interpro_vocab.pkl"

# Gene Ontology
GO_OBO_PATH = DATA_ROOT / "go.obo"
GO_ANNOTATION_TEMPLATE = SWISSPROT_ROOT / "swissprot_2024_01_{onto}_annotations.tsv"
GO_EXP_ANNOTATION_TEMPLATE = (
    SWISSPROT_ROOT / "swissprot_2024_01_{onto}_exp_annotations.tsv"
)
GO_VOCAB = SWISSPROT_ROOT / "protein_graphs" / "go_vocab.pkl"

# Embeddings
EMBED_H5_PATH = SWISSPROT_ROOT / "swissprot_esm1b_per_aa.h5"

# Structures
ALPHAFOLD_PDB_DIR = SWISSPROT_ROOT / "alphafold_pdb"
ESMFOLD_PDB_DIR = SWISSPROT_ROOT / "esmfold_pdb"

# Graphs
PROTEIN_GRAPHS_DIR = SWISSPROT_ROOT / "protein_graphs"
STRUCTURE_MISSING_PATH = PROTEIN_GRAPHS_DIR / "structure_missing_rev.fasta"
STRUCTURE_MISSING_FASTA = SWISSPROT_ROOT / "structure_missing.fasta"

# StringDB
STRINGDB_PATH = SWISSPROT_ROOT / "swissprot_stringdb.tsv"
STRINGDB_SWISSPROT_MAPPING = SWISSPROT_ROOT / "idmapping_swissprot_stringdb.tsv"

# ============================================================================
# Utilities Paths
# ============================================================================

DIAMOND_DB_DIR = DATA_ROOT / "diamond"
DIAMOND_SWISSPROT_DB = SWISSPROT_ROOT / "swissprot_2024_01.dmnd"
MUSCLE_EXECUTABLE = DATA_ROOT / "muscle-linux-x86.v5.3"

# ============================================================================
# Model Checkpoints and Configs
# ============================================================================

CONFIGS_DIR = SRC_ROOT / "configs"
DEFAULT_CONFIG = CONFIGS_DIR / "cfg.yaml"
TOY_CONFIG = CONFIGS_DIR / "toy_cfg.yaml"

MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# ESM Model
ESM_CHECKPOINT_DIR = DATA_ROOT / "esm_checkpoints"
ESM1B_CHECKPOINT = ESM_CHECKPOINT_DIR / "esm1b_t33_650M_UR50S.pt"
ESMFOLD_CHECKPOINT = ESM_CHECKPOINT_DIR / "esmfold_v1.pt"

# ============================================================================
# Output Directories
# ============================================================================

RESULTS_DIR = PROJECT_ROOT / "results"
INFERENCE_OUTPUT_DIR = DATA_ROOT / "inference_outputs"
TMP_DIR = DATA_ROOT / "tmp"

# ============================================================================
# ESM Embedding Parameters
# ============================================================================

ESM_MODEL_NAME = "esm1b_t33_650M_UR50S"
ESM_EMBEDDING_DIM = 1280
ESM_NUM_LAYERS = 33
ESM_DEFAULT_REPR_LAYER = 33

# Batching parameters
ESM_MAX_AA_PER_BATCH = 4000
ESM_DEFAULT_BATCH_SIZE = 8
ESM_MAX_SEQUENCE_LENGTH = 1024

# ============================================================================
# ESMFold Parameters
# ============================================================================

ESMFOLD_MODEL_NAME = "facebook/esmfold_v1"
ESMFOLD_DEFAULT_CHUNK_SIZE = 128
ESMFOLD_MAX_SEQUENCE_LENGTH = 1024
ESMFOLD_CONDA_ENV = "esmfold"
ESMFOLD_TRUNC_LEN = 1024

# ============================================================================
# InterProScan API Parameters
# ============================================================================

INTERPROSCAN_BASE_URL = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"
INTERPROSCAN_DEFAULT_EMAIL = "antoine.toffano@lirmm.fr"
INTERPROSCAN_DEFAULT_TIMEOUT = 300  # seconds
INTERPROSCAN_DEFAULT_POLL_INTERVAL = 20  # seconds
INTERPROSCAN_MAX_RETRIES = 3

# ============================================================================
# DIAMOND Alignment Parameters
# ============================================================================

DIAMOND_EXECUTABLE = "diamond"
DIAMOND_DEFAULT_EVALUE = 1e-3
DIAMOND_OUTPUT_FORMAT = "6"  # Tabular format
DIAMOND_DEFAULT_THREADS = 4

# ============================================================================
# Graph Construction Parameters
# ============================================================================

# Contact edges
CONTACT_CUTOFF = 10.0  # Angstroms
CONTACT_CHUNK_SIZE = 512

# Node types
NODE_TYPE_AA = "aa"
NODE_TYPE_PROTEIN = "protein"

# Edge types
EDGE_TYPE_BELONGS_TO = ("aa", "belongs_to", "protein")
EDGE_TYPE_HAS = ("protein", "has", "aa")
EDGE_TYPE_CLOSE_TO = ("aa", "close_to", "aa")
EDGE_TYPE_STRINGDB = ("protein", "stringdb", "protein")

# ============================================================================
# Gene Ontology Parameters
# ============================================================================

GO_ONTOLOGIES = ["BPO", "CCO", "MFO"]
GO_ANNOTATION_TYPES = ["experimental", "curated"]

# Root GO terms for each ontology
GO_ROOT_TERMS = {"GO:0003674", "GO:0008150", "GO:0005575"}
GO_BIOLOGICAL_PROCESS = "GO:0008150"
GO_MOLECULAR_FUNCTION = "GO:0003674"
GO_CELLULAR_COMPONENT = "GO:0005575"

GO_FUNC_DICT = {
    "cc": GO_CELLULAR_COMPONENT,
    "mf": GO_MOLECULAR_FUNCTION,
    "bp": GO_BIOLOGICAL_PROCESS,
}

GO_NAMESPACE_MAP = {
    "biological_process": "BPO",
    "cellular_component": "CCO",
    "molecular_function": "MFO",
}

GO_NAMESPACES = {
    "cc": "cellular_component",
    "mf": "molecular_function",
    "bp": "biological_process",
}

GO_NAMESPACES_REVERT = {
    "cellular_component": "cc",
    "molecular_function": "mf",
    "biological_process": "bp",
}

# ============================================================================
# Training/Evaluation Parameters
# ============================================================================

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

# ============================================================================
# Model Interpretability Parameters
# ============================================================================

SUPPORTED_CAPTUM_METHODS = [
    "IntegratedGradients",
    "Saliency",
    "InputXGradient",
    "Deconvolution",
    "ShapleyValueSampling",
    "GuidedBackprop",
]

# ============================================================================
# External API URLs
# ============================================================================

UNIPROT_JSON_URL = "https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
PDB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
ALPHAFOLD_STRUCTURE_URL = (
    "https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.pdb"
)

# ============================================================================
# Logging Configuration
# ============================================================================

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL_DEFAULT = "INFO"

# ============================================================================
# Conda Environments
# ============================================================================

CONDA_ENV_MAIN = "pyg2"
CONDA_ENV_ESMFOLD = "esmfold"
