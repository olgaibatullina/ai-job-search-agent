from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_ROOT = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"

# Vector Store
FAISS_INDEX = DATA_DIR / "faiss_index"

# Datasets
RAW_JOBS = DATA_DIR / "raw_jobs/raw_jobs.json"
RETRIEVAL_EVAL = DATA_DIR / "golden/retrieval_eval.yaml"
RETRIEVAL_NEGATIVE_EVAL = DATA_DIR / "golden/retrieval_negative_eval.yaml"
RETRIEVAL_TYPO_EVAL = DATA_DIR / "golden/retrieval_typo_eval.yaml"

#LLM Configurations
LLM_CONFIG = CONFIG_ROOT / "llm_config.yaml"