"""Evaluate retrieval F1 against the golden dataset."""

import pytest
import evaluate

from itertools import chain

from config import paths
from evaluations import helper
from src.rag import retriever


retrieval_cases = helper.load_retrieval_dataset(paths.RETRIEVAL_EVAL)["queries"]
retrieval_negative_cases = helper.load_retrieval_dataset(paths.RETRIEVAL_NEGATIVE_EVAL)["queries"]
retrieval_typo_cases = helper.load_retrieval_dataset(paths.RETRIEVAL_TYPO_EVAL)["queries"]

retrieval_eval_cases = list(
    chain(
        retrieval_cases,
        retrieval_typo_cases,
        retrieval_negative_cases,
    )
)

f1_metric = evaluate.load("f1")

@pytest.mark.parametrize("case", retrieval_eval_cases)
def test_f1_retrieval(job_index, all_jobs, case):
    """Evaluate retrieval F1 score against the golden dataset.

    Args:
        job_index: FAISS index used by the retriever to find relevant jobs.
        all_jobs: All jobs from the raw jobs dataset, used to build the complete
            set of job IDs for binary retrieval labels.
        case: Evaluation case containing the query and expected relevant job IDs.

    Steps:
        1. Retrieve the top 5 jobs for the test query.
        2. Extract unique job IDs from the retrieved results.
        3. Extract all job IDs from the raw jobs dataset.
        4. Convert retrieved and relevant job IDs into binary labels.
        5. Calculate retrieval F1 score using the golden labels.
    """  
    results = retriever.retrieve_relevant_jobs(job_index, case["query"], k=5)
    retrieved_job_ids = list({result.metadata["job_id"] for result in results})

    all_job_ids = [job["job_id"] for job in all_jobs]

    predictions, references = helper.build_retrieval_labels(
        all_job_ids, 
        retrieved_job_ids, 
        case["relevant_job_ids"]
        )
    f1_result = f1_metric.compute(predictions=predictions, references=references)

    print(f"Query: {case["query_id"]} . {case["query"]}")   
    print(f"F1: {f1_result["f1"]}\n")