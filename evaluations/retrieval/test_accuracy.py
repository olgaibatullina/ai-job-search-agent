"""Evaluate retrieval Accuracy against the golden dataset."""

import pytest
import evaluate

from config.paths import RETRIEVAL_EVAL
from evaluations import helper
from src.rag.retriever import retrieve_relevant_jobs


retrieval_cases = helper.load_retrieval_dataset(RETRIEVAL_EVAL)
accuracy_metric = evaluate.load("accuracy")

@pytest.mark.parametrize("case", retrieval_cases)
def test_accuracy_retrieval(job_index, all_job_ids, case):
    results = retrieve_relevant_jobs(job_index, case["query"], k=5)
    retrieved_job_ids = list({result.metadata["job_id"] for result in results})

    predictions, references = helper.build_retrieval_labels(
        all_job_ids, 
        retrieved_job_ids, 
        case["relevant_job_ids"])
    accuracy_result = accuracy_metric.compute(predictions=predictions, references=references)

    print(f"Query: {case["query_id"]} . {case["query"]}")   
    print(f"Accuracy: {accuracy_result["accuracy"]}\n")
