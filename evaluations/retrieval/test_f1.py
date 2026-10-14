"""Evaluate retrieval F1 against the golden dataset."""

import pytest
import evaluate

from config.paths import RETRIEVAL_EVAL
from evaluations import helper
from src.rag.retriever import retrieve_relevant_jobs


retrieval_cases = helper.load_retrieval_dataset(RETRIEVAL_EVAL)
f1_metric = evaluate.load("f1")

@pytest.mark.parametrize("case", retrieval_cases)
def test_f1_retrieval(job_index, all_job_ids, case):
    results = retrieve_relevant_jobs(job_index, case["query"], k=5)
    retrieved_job_ids = list({result.metadata["job_id"] for result in results})

    predictions, references = helper.build_retrieval_labels(
        all_job_ids, 
        retrieved_job_ids, 
        case["relevant_job_ids"]
        )
    f1_result = f1_metric.compute(predictions=predictions, references=references)

    print(f"Query: {case["query_id"]} . {case["query"]}")   
    print(f"F1: {f1_result["f1"]}\n")