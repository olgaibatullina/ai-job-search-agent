"""Test RAG retrieval latency."""

import pytest
import time

from config.paths import RETRIEVAL_EVAL
from evaluations import helper
from src.rag import retriever


retrieval_cases = helper.load_retrieval_dataset(RETRIEVAL_EVAL)["queries"]

@pytest.mark.parametrize("case", retrieval_cases)
def test_rag_retrieval_latency(job_index, case):
    """Compare latency and unique job count for simple and deduplicated retrieval.

    Runs each retrieval method multiple times to observe warm-up effects
    and collect repeated latency measurements for later analysis.
    """
    
    for i in range(10):
        # Measure simple retrieval latency.
        simple_retrieval_start = time.perf_counter()   
        simple_retrieval = retriever.retrieve_from_vector_store(job_index, case["query"], k=5)
        simple_retrieval_latency = time.perf_counter() - simple_retrieval_start

        # Measure deduplicated retrieval latency.
        dedup_retrieval_start = time.perf_counter()    
        dedup_retrieval = retriever.retrieve_relevant_jobs(job_index, case["query"], k=5)
        dedup_retrieval_latency = time.perf_counter() - dedup_retrieval_start

        # Compare latency and the number of unique jobs returned.
        print(f"{i+1}: {case['query_id']}, {case['query']}: \n"
            f"Simple retrieval : {simple_retrieval_latency * 1000:.2f} ms, "
            f"unique_jobs= {len({document.metadata["job_id"] for document in simple_retrieval})} \n"
            f"Dedup retrieval : {dedup_retrieval_latency * 1000:.2f} ms, "
            f"unique_jobs= {len({document.metadata["job_id"] for document in dedup_retrieval})}")

    print(f"--------------------\n")