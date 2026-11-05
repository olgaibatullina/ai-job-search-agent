"""Evaluate RAG Faithfulness using RAGAS with LangSmith tracing and CSV reporting."""

import pytest
import time

from itertools import chain

from langsmith import trace

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

run_name = "relevancy_baseline"

faithfulness_failed_cases = ["q003", "q028", "neg006", "neg019", "neg020", "neg024", "neg028"]
relevancy_failed_cases = ["neg009", "neg029"]

eval_cases = [
    case
    for case in chain(
        retrieval_cases,
        retrieval_typo_cases,
        retrieval_negative_cases,
    )
    if case["query_id"] in relevancy_failed_cases
] * 1

all_cases = list(
    chain(
        retrieval_cases,
        retrieval_typo_cases,
        retrieval_negative_cases,
    )
)

retrieval_metrics_non_llm = [NonLLMContextPrecisionWithReference, NonLLMContextRecall]
retrieval_metrics_llm = [ContextPrecision, ContextRecall]
generation_metrics_llm = [AnswerAccuracy, AnswerRelevancy, Faithfulness, ResponseGroundedness]

@pytest.mark.asyncio
@pytest.mark.parametrize("case", all_cases)
async def test_ragas_metric_eval(job_index, case):
    """Evaluate RAG generation faithfulness using RAGAS LLM-as-a-Judge."""
    k = 1
    metric = AnswerRelevancy

    # Retrieve jobs and prepare evaluation contexts.
    retrieved_chunks = retriever.retrieve_relevant_jobs(
        job_index, case["query"], k
    )
    retrieved_jobs = loader.get_jobs_from_chunks(retrieved_chunks)

    retrieved_contexts = [
        job["posting"]["description"]
        for job in retrieved_jobs
    ]

    # Generate RAG response.
    llm, prompt = helper.prepare_ragas_generator(paths.LLM_CONFIG)
    context = generator.prepare_context(retrieved_chunks)

    
    llm_response = generator.generate_answer(
        query=case["query"],
        context=context,
        prompt=prompt,
        llm=llm,
    )
    
    start = time.perf_counter()
    score = None
    status = "SUCCESS"
    trace_id = None
    error_type = ""

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

    try:
        # Group all LLM calls for one Faithfulness evaluation under a parent trace.
        with trace(
            name=f"{metric.__name__} Evaluation",
            run_type="chain",
            metadata={
                "query_id": case["query_id"],
                "k": k,
            },
        ) as parent_run:

            # Store the trace ID to link CSV results with LangSmith.
            trace_id = str(parent_run.id)

            # Calculate Faithfulness using the configured LLM judge.
            result = await helper.calculate_generation_metric_llm(**kwargs)
        score = result.value

    except Exception as e:
        # Record evaluation failures without losing the test case results.
        status = "ERROR"
        error_type = type(e).__name__
        print(f"Evaluation error: {e}")

    finally:
        # Record execution time and persist results for subsequent analysis.
        latency = time.perf_counter() - start

        helper.save_evaluation_result({
            "query_id": case["query_id"],
            "query": case["query"],
            "k": k,
            "latency_s": round(latency, 3),
            "metric": metric,
            "score": score,
            "status": status,
            "error_type": error_type,
            "judge_max_tokens": helper.load_retrieval_dataset(paths.LLM_CONFIG)["judge"]["max_tokens"],

            # Token usage is populated later from LangSmith traces.
            "judge_input_tokens": None,
            "judge_output_tokens": None,
            "judge_total_tokens": None,
            "judge_call_count": None,
            "judge_avg_output_tokens_per_call": None,
            "judge_max_output_tokens_per_call": None,
            "trace_id": trace_id
        },
        run_name)
    
    # Console output.
    metric_name = (
        type(metric).__name__
        if not isinstance(metric, type)
        else metric.__name__
    )

    print(f"\nQuery: {case['query_id']} - {case['query']}")
    print(f"K: {k}")
    print(f"{metric_name}: {score}")
    print(f"Status: {status}")

    if status == "ERROR":
        print(f"Error: {error_type}")

    print(f"Latency: {latency:.3f}s\n")
