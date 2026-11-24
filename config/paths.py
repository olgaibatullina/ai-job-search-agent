from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_ROOT = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"

# Vector Store
FAISS_INDEX = DATA_DIR / "faiss_index"

# Datasets
RAW_JOBS = DATA_DIR / "raw_jobs/raw_jobs.json"
PROCESSED_JOBS = DATA_DIR / "raw_jobs/processed_jobs.json"
RETRIEVAL_EVAL = DATA_DIR / "golden/retrieval_eval.yaml"
RETRIEVAL_NEGATIVE_EVAL = DATA_DIR / "golden/retrieval_negative_eval.yaml"
RETRIEVAL_TYPO_EVAL = DATA_DIR / "golden/retrieval_typo_eval.yaml"

# LLM Configurations
LLM_CONFIG = CONFIG_ROOT / "llm_config.yaml"

# Evaluation results
EVALUATION_RESULTS = PROJECT_ROOT / "evaluation_results"
E1_RESULTS = EVALUATION_RESULTS / "experiments/E1_judge_context_size"
E2_RESULTS = EVALUATION_RESULTS / "experiments/E2_rag_evaluation_nan_investigation"
E3_RESULTS = EVALUATION_RESULTS / "experiments/E3_retrieval_metrics_validation"

E2_H9_RESULTS = E2_RESULTS / "E2_H9_ollama_structured_output"
E2_H10_RESULTS = E2_RESULTS / "E2_H10_ollama_native_vs_langchain"
E2_H11_RESULTS = E2_RESULTS / "E2_H11_ollama_openai_compatibility"
E2_H12_RESULTS = E2_RESULTS / "E2_H12_ollama_instructor_integration"


# Prompts
E2_H9_PROMPT = PROJECT_ROOT / "evaluations/prompts/e2_h9_ollama.yaml"