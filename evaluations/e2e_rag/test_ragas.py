"""Evaluate RSGAS against the golden dataset."""

import pytest

from ragas.metrics import NonLLMContextPrecisionWithReference
from ragas.metrics.collections import ContextPrecision
from ragas.llms import llm_factory

from config.paths import RETRIEVAL_EVAL
from config.paths import LLM_CONFIG
from evaluations import helper
from src.jobs import loader
from src.rag.retriever import retrieve_relevant_jobs


retrieval_cases = helper.load_retrieval_dataset(RETRIEVAL_EVAL)

@pytest.mark.parametrize("case", retrieval_cases)
def test_context_precision(job_index, all_jobs, case):
    """Evaluate context precision of retrieved job descriptions using RAGAS.

    Args:
        job_index: FAISS vector store used to retrieve relevant job chunks.
        all_jobs: All jobs loaded from the raw jobs dataset.
        case: Retrieval evaluation case containing the query and relevant job IDs.
    
    Steps:
        1. Create the RAGAS context precision metric.
        2. Retrieve job chunks for the evaluation query.
        3. Convert retrieved chunks into full job records using their job IDs.
        4. Extract full job descriptions from the retrieved jobs as contexts.
        5. Extract full job descriptions for the manually labeled relevant jobs
            from the golden dataset as reference contexts.
        6. Calculate Context Precision using the legacy RAGAS evaluation API.
    """
    
    context_precision = NonLLMContextPrecisionWithReference()
    
    retrieved_chunks = retrieve_relevant_jobs(job_index, case["query"], k=5)
    retrieved_jobs = loader.get_jobs_from_chunks(retrieved_chunks)

    retrieved_contexts = [
        job["posting"]["description"]
        for job in retrieved_jobs
    ]

    reference_contexts = [
        job["posting"]["description"]
        for job in all_jobs
        if job["job_id"] in case["relevant_job_ids"]
    ]

    context_precision_metric = helper.calculate_ragas_legacy_metric(
        context_precision, 
        case["query"], 
        retrieved_contexts, 
        reference_contexts
        )

    print(f"Query: {case["query_id"]} . {case["query"]}")   
    print(f"Context Precision: {context_precision_metric}\n")


@pytest.mark.asyncio
@pytest.mark.parametrize("case", retrieval_cases)
async def test_context_precision_llm_judge(job_index, all_jobs, case):
    """Evaluate context precision of retrieved job descriptions using RAGAS.

    Args:
        job_index: FAISS vector store used to retrieve relevant job chunks.
        all_jobs: All jobs loaded from the raw jobs dataset.
        case: Retrieval evaluation case containing the query, reference, and relevant job IDs.
    
    Steps:
        1. Create the RAGAS context precision metric with the configured LLM judge.
        2. Retrieve job chunks for the evaluation query.
        3. Convert retrieved chunks into full job records using their job IDs.
        4. Extract full job descriptions from the retrieved jobs as contexts.
        5. Calculate the LLM-as-a-Judge Context Precision score using the query, reference, and retrieved contexts.
    """

    retrieved_chunks = retrieve_relevant_jobs(job_index, case["query"], k=5)
    retrieved_jobs = loader.get_jobs_from_chunks(retrieved_chunks)

    retrieved_contexts = [
        job["posting"]["description"]
        for job in retrieved_jobs
    ]

    context_precision_metric = await helper.calculate_ragas_metric_llm_judge(
        metric=ContextPrecision, 
        query=case["query"], 
        retrieved_contexts=retrieved_contexts, 
        reference=case["reference"],
        llm_config_path=LLM_CONFIG
        )
    
    print(f"Query: {case["query_id"]} . {case["query"]}")   
    print(f"Context Precision LLM-AS-JUDGE: {context_precision_metric.value}\n")