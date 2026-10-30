"""Evaluate RAG quality using RAGAS metrics."""

import pytest

from itertools import chain

from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall
from ragas.metrics.collections import ContextPrecision, ContextRecall
from ragas.metrics.collections import AnswerAccuracy, AnswerRelevancy, Faithfulness, ResponseGroundedness

from config import paths
from evaluations import helper
from src.jobs import loader
from src.rag import retriever, generator


retrieval_cases = helper.load_retrieval_dataset(paths.RETRIEVAL_EVAL)["queries"]
retrieval_negative_cases = helper.load_retrieval_dataset(paths.RETRIEVAL_NEGATIVE_EVAL)["queries"]
retrieval_typo_cases = helper.load_retrieval_dataset(paths.RETRIEVAL_TYPO_EVAL)["queries"]

eval_cases = list(
    chain(
        retrieval_cases,
        retrieval_typo_cases,
        retrieval_negative_cases,
    )
)

retrieval_metrics_non_llm = [NonLLMContextPrecisionWithReference, NonLLMContextRecall]
retrieval_metrics_llm = [ContextPrecision, ContextRecall]
generation_metrics_llm = [AnswerAccuracy, AnswerRelevancy, Faithfulness, ResponseGroundedness]

@pytest.mark.parametrize("case", eval_cases)
@pytest.mark.parametrize("metric", retrieval_metrics_non_llm)
def test_ragas_retrieval_metric_non_llm(job_index, all_jobs, case, metric):
    """Evaluate RAG retrieval quality using non-LLM RAGAS metrics.

    Args:
        job_index: FAISS vector store used to retrieve relevant job chunks.
        all_jobs: All jobs loaded from the raw jobs dataset.
        case: Retrieval evaluation case containing the query and relevant job IDs.
        metric: RAGAS non-LLM retrieval metric to evaluate.

    Steps:
        1. Retrieve relevant job results for the evaluation query.
        2. Convert retrieved chunks into full job records using their job IDs.
        3. Extract full job descriptions from the retrieved jobs as contexts.
        4. Extract full job descriptions for the manually labeled relevant jobs from the 
            golden dataset as reference contexts.
        5. Calculate the selected RAGAS retrieval metric using retrieved and reference contexts 
            with legacy RAGAS evaluation API.
    """
    retrieved_chunks = retriever.retrieve_relevant_jobs(job_index, case["query"], k=5)
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

    retrieval_metric_non_llm = helper.calculate_retrieval_metric_non_llm(
        metric, 
        case["query"], 
        retrieved_contexts, 
        reference_contexts
        )

    # Console output
    metric_name = (
        type(metric).__name__
        if not isinstance(metric, type)
        else metric.__name__
    )

    print(f"Query: {case["query_id"]} . {case["query"]}")   
    print(f"Metric retrieval/non-llm: {metric_name}: {retrieval_metric_non_llm}\n")


# LLM evaluation is parameterized across 75 evaluation cases × 2 metrics
# (ContextPrecision, ContextRecall), resulting in 150 evaluations.
# A full run takes approximately 4 hours locally.
# To run a single dataset/slice independently, comment out other
# parametrized datasets in eval_cases.
@pytest.mark.asyncio
@pytest.mark.parametrize("case", eval_cases)
@pytest.mark.parametrize("metric", retrieval_metrics_llm)
async def test_ragas_retrieval_metric_llm(job_index, case, metric):
    """Evaluate RAG retrieval quality using RAGAS LLM-as-a-Judge metrics.

    Args:
        job_index: FAISS vector store used to retrieve relevant job results.
        case: Retrieval evaluation case containing the query and reference.
        metric: RAGAS LLM-based retrieval metric to evaluate.

    Steps:
        1. Create the selected RAGAS metric with the configured LLM judge.
        2. Retrieve relevant job results for the evaluation query.
        3. Convert retrieved results into full job records using their job IDs.
        4. Extract full job descriptions from the retrieved jobs as retrieved contexts.
        5. Calculate the selected RAGAS retrieval metric using the query, reference,
            and retrieved contexts.
    """
    retrieved_chunks = retriever.retrieve_relevant_jobs(job_index, case["query"], k=5)
    retrieved_jobs = loader.get_jobs_from_chunks(retrieved_chunks)

    retrieved_contexts = [
        job["posting"]["description"]
        for job in retrieved_jobs
    ]

    # Debug: uncomment to inspect retrieved contexts for a single evaluation case.
    # for i, job in enumerate(retrieved_contexts):
    #     print(f"---------------------\n{i+1}\n---------------------\n{job[:500]}\n\n")

    retrieval_metric_llm = await helper.calculate_retrieval_metric_llm(
        metric=metric, 
        query=case["query"], 
        retrieved_contexts=retrieved_contexts, 
        reference=case["reference"],
        llm_config_path=paths.LLM_CONFIG
        )

    # Console output
    metric_name = (
        type(metric).__name__
        if not isinstance(metric, type)
        else metric.__name__
    )
    
    print(f"Query: {case["query_id"]} . {case["query"]}")   
    print(f"Metric retrieval/llm: {metric_name}: {retrieval_metric_llm.value}\n")


# LLM evaluation is parameterized across 75 evaluation cases × 4 metrics
# (AnswerAccuracy, AnswerRelevancy, Faithfulness, ResponseGroundedness), 
# resulting in 300 evaluations.
# A full run takes approximately 8 hours locally.
# To run a single dataset/slice independently, comment out the other
# parametrized datasets in eval_cases.
@pytest.mark.asyncio
@pytest.mark.parametrize("case", eval_cases)
@pytest.mark.parametrize("metric", generation_metrics_llm)
async def test_ragas_generation_metric_llm(job_index, case, metric):
    """Evaluate RAG generation quality using RAGAS LLM-as-a-Judge metrics.

    Args:
        job_index: FAISS vector store used to retrieve relevant job results.
        case: Generation evaluation case containing the query.
        metric: RAGAS LLM-based generation metric to evaluate.

    Steps:
        1. Retrieve relevant job results for the evaluation query.
        2. Convert retrieved results into full job records using their job IDs.
        3. Extract full job descriptions from the retrieved jobs as retrieved contexts.
        4. Prepare the generator LLM and prompt using the configured generator settings.
        5. Generate an answer using the evaluation query and retrieved context.
        6. Calculate the selected RAGAS generation metric using the generated response
           and retrieved contexts.
        7. Pass the user input to metrics that require it.
    """
    retrieved_chunks = retriever.retrieve_relevant_jobs(job_index, case["query"], k=1)
    retrieved_jobs = loader.get_jobs_from_chunks(retrieved_chunks)

    retrieved_contexts = [
        job["posting"]["description"]
        for job in retrieved_jobs
    ]

    # Prepare generator LLM and prompt using configured generator settings.
    # Build the context from retrieved results and generate an LLM response
    # for the evaluation query.
    llm, prompt = helper.prepare_ragas_generator(paths.LLM_CONFIG)
    context = generator.prepare_context(retrieved_chunks)

    llm_response = generator.generate_answer(
        query=case["query"],
        context=context,
        prompt=prompt,
        llm=llm,
    )

    # Prepare metric arguments dynamically as different RAGAS generation
    # metrics require additional inputs: Faithfulness requires user_input,
    # AnswerAccuracy requires user_input and reference, and AnswerRelevancy
    # requires embeddings.
    kwargs = {
        "metric": metric,
        "response": llm_response,
        "llm_config_path": paths.LLM_CONFIG,
    }

    if metric is AnswerAccuracy:
        kwargs["user_input"] = case["query"]
        kwargs["reference"] = case["reference"]

    if metric is AnswerRelevancy:
        kwargs["user_input"] = case["query"]
        kwargs["embedding"] = True

    if metric is Faithfulness:
        kwargs["user_input"] = case["query"]
        kwargs["retrieved_contexts"] = retrieved_contexts

    if metric is ResponseGroundedness:
        kwargs["retrieved_contexts"] = retrieved_contexts

    retrieval_metric_llm = await helper.calculate_generation_metric_llm(**kwargs)
    
    # Console output
    metric_name = (
        type(metric).__name__
        if not isinstance(metric, type)
        else metric.__name__
    )
    
    print(f"Query: {case["query_id"]} . {case["query"]}")   
    print(f"Metric generation/llm: {metric_name}: {retrieval_metric_llm.value}\n")