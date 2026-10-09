"""Evaluate retrieval Recall against the golden dataset."""

import pytest
import evaluate

from config.paths import RETRIEVAL_EVAL
from evaluations import helper
from src.rag.retriever import retrieve_from_vector_store


retrival_cases = helper.load_retrival_dataset(RETRIEVAL_EVAL)
recall_metric = evaluate.load("recall")

@pytest.mark.parametrize("case", retrival_cases)
def test_recall_retrieval(job_index, all_job_ids, case):
    results = retrieve_from_vector_store(job_index, case["query"], k=5) 
    retrieved_job_ids = list({result.metadata["job_id"] for result in results}) 

    predictions, references = helper.build_retrieval_labels(
        all_job_ids, 
        retrieved_job_ids, 
        case["relevant_job_ids"]
        )
    recall_result = recall_metric.compute(predictions=predictions, references=references)

    print(f"Query: {case["query_id"]} . {case["query"]}")   
    print(f"Recall: {recall_result["recall"]}\n")