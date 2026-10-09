"""Evaluate retrieval Precision against the golden dataset."""

import pytest
import evaluate

from config.paths import RETRIEVAL_EVAL
from evaluations import helper
from src.rag.retriever import retrieve_from_vector_store


retrival_cases = helper.load_retrival_dataset(RETRIEVAL_EVAL)
precision_metric = evaluate.load("precision")

@pytest.mark.parametrize("case", retrival_cases)
def test_precision_retrieval(job_index, all_job_ids, case):
    results = retrieve_from_vector_store(job_index, case["query"], k=5)
    retrieved_job_ids = list({result.metadata["job_id"] for result in results})

    predictions, references = helper.build_retrieval_labels(
        all_job_ids, 
        retrieved_job_ids, 
        case["relevant_job_ids"]
        )
    precision_result = precision_metric.compute(predictions=predictions, references=references)

    print(f"Query: {case["query_id"]} . {case["query"]}")   
    print(f"Precision: {precision_result["precision"]}\n")
