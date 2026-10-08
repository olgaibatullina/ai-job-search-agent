from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

RETRIEVAL_EVAL = DATA_DIR / "golden/retrieval_eval.yaml"
RAW_JOBS = DATA_DIR / "raw_jobs/raw_jobs.json"
FAISS_INDEX = DATA_DIR / "faiss_index"