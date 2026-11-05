"""Shared helper functions for evaluation and metric calculation."""

import yaml
import csv
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI

from ragas import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics.base import Metric
from ragas.metrics.collections import BaseMetric
from ragas.metrics.result import MetricResult
from ragas.embeddings.base import embedding_factory

from config.paths import E1_RESULTS


def load_retrieval_dataset(dataset: Path) -> list[dict]:
    """Load retrieval evaluation queries from a YAML dataset.

    Args:
        dataset: Path to the YAML dataset.

    Returns:
        A list of query dictionaries.
    """
    with dataset.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_retrieval_labels(
    all_job_ids: list[str], 
    retrieved_jobs: list[str], 
    relevant_jobs: list[str]
    ) -> tuple[list[int], list[int]]:
    """Build prediction and reference labels for retrieval metrics.

    Args:
        all_job_ids: All job IDs from data/raw_jobs.json.
        retrieved_jobs: Job IDs returned by the retriever.
        relevant_jobs: Job IDs marked as relevant in the golden dataset.

    Returns:
        A tuple containing prediction and reference labels.
    """
    predictions = []
    references =[]

    # Convert jobs to sets for faster lookups.    
    relevant_jobs = set(relevant_jobs)
    retrieved_jobs = set(retrieved_jobs)

    for job_id in all_job_ids:
        references.append(1 if job_id in relevant_jobs else 0)
        predictions.append(1 if job_id in retrieved_jobs else 0)    
    
    return predictions, references


def prepare_ragas_generator(llm_config_path: Path) -> tuple[ChatOllama, ChatPromptTemplate]:
    """Prepare the generator LLM and prompt using the configured settings.

    Args:
        llm_config_path (Path): Path to the YAML file containing generator configuration.

    Returns:
        tuple[ChatOllama, ChatPromptTemplate]: Configured generator LLM and chat prompt template.
    """
    llm_config = load_retrieval_dataset(llm_config_path)

    llm = ChatOllama(
        model=llm_config["generator"]["model"],
        temperature=llm_config["generator"]["temperature"],
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", llm_config["generator"]["system_prompt_v1"]),
        ("user", llm_config["generator"]["user_prompt"]),
    ])
    
    return llm, prompt


def prepare_ragas_llm_scorer(
        metric: type[BaseMetric], 
        llm_config_path: Path,
        embedding: bool = False
    ) -> BaseMetric:
    """Prepare a RAGAS metric with the configured LLM judge.

    Args:
        metric (type[BaseMetric]): RAGAS metric class to instantiate with the LLM judge.
        llm_config_path (Path): Path to the YAML file containing judge LLM configuration.
        embedding (bool): Whether to initialize embeddings for metrics, 
            required by AnswerRelevancy metrics.

    Returns:
        BaseMetric: Instantiated RAGAS metric configured with the LLM judge.
    """
    load_dotenv()

    llm_config = load_retrieval_dataset(llm_config_path)
    
    # Ollama exposes an OpenAI-compatible API.
    # Create an async OpenAI client to communicate with the Ollama model server.
    # wrap_openai() enables LangSmith tracing for LLM calls, using settings from environment variables:
    # LANGSMITH_TRACING=tru/false in .env.
    client = wrap_openai(
        AsyncOpenAI(
            api_key="ollama",
            base_url=llm_config["judge"]["base_url"],
        )
    )
    
    # Create a RAGAS-compatible LLM wrapper using the configured judge model.
    # RAGAS uses this LLM to evaluate the selected metric.
    llm = llm_factory(
        model = llm_config["judge"]["model"], 
        provider = "openai", 
        client = client,
        max_tokens = llm_config["judge"]["max_tokens"],
        # reasoning_effort="none", # Disable thinking mode
    )

    kwargs = {
        "llm": llm,
    }

    if embedding:
        # Create a RAGAS-compatible embeddings wrapper using the configured embedding model.
        # RAGAS uses these embeddings for metrics that require semantic similarity evaluation.
        embedding = embedding_factory(
            provider="huggingface",
            model="sentence-transformers/all-mpnet-base-v2",
        )
        kwargs["embeddings"] = embedding

    # Instantiate selected RAGAS metric with the configured LLM judge.
    return metric(**kwargs)


def calculate_retrieval_metric_non_llm(
        metric: type[Metric], 
        query: str, 
        retrieved_contexts: list[str], 
        reference_contexts: list[str]
    ) -> float:
    """Calculate a RAGAS non-LLM retrieval metric using the legacy API.

    This helper is used to compare legacy RAGAS metrics with their
    modern Collections API counterparts.

    Args:
        metric (type[Metric]): RAGAS non-LLM retrieval metric class to evaluate.
        query (str): User query used for retrieval evaluation.
        retrieved_contexts (list[str]): Job descriptions retrieved by the RAG system.
        reference_contexts (list[str]): Reference job descriptions expected for the query.

    Returns:
        float: Calculated RAGAS metric score.
    """
    
    # Context Precision/Recall require reference contexts to compare
    # against retrieved contextsю
    # Negative cases have no reference contexts, meaning the reference-based 
    # metric is undefined: relevant_job_ids: [].
    if not reference_contexts:
        return None
    
    metric_instance = metric()
    sample = SingleTurnSample(
        user_input=query,
        retrieved_contexts=retrieved_contexts,
        reference_contexts=reference_contexts,
    )

    # Calculate metric score using the prepared evaluation sample.
    return metric_instance.single_turn_score(sample)


async def calculate_retrieval_metric_llm(
    metric: type[BaseMetric], 
    query: str, 
    retrieved_contexts: list[str], 
    reference: str, 
    llm_config_path: Path
    ) -> MetricResult:
    """Calculate a RAGAS LLM-based retrieval metric.

    Args:
        metric (BaseMetric): RAGAS retrieval metric to evaluate.
        query (str): User query used for retrieval evaluation.
        retrieved_contexts (list[str]): Job descriptions retrieved by the RAG system.
        reference (str): Reference answer describing the expected retrieval result.
        llm_config_path (Path): Path to the YAML file containing judge LLM configuration.

    Returns:
        MetricResult: RAGAS metric evaluation result.
    """
    scorer = prepare_ragas_llm_scorer(metric=metric, llm_config_path=llm_config_path)

    # Evaluate selected retrieval metric using the query, reference,
    # and contexts retrieved by the RAG system.
    return await scorer.ascore(       
        user_input = query, 
        reference = reference,
        retrieved_contexts = retrieved_contexts
    )


async def calculate_generation_metric_llm(
    metric: type[BaseMetric], 
    response: str, 
    llm_config_path: Path,
    retrieved_contexts: list[str] | None = None,
    user_input: str | None = None,
    reference: str | None = None,
    embedding: bool = False
    ) -> MetricResult:
    """Calculate a RAGAS LLM-based generation metric.

    Args:
        metric (type[BaseMetric]): RAGAS generation metric class to evaluate.
        response (str): Response generated by the RAG system.
        llm_config_path (Path): Path to the YAML file containing judge LLM configuration.
        retrieved_contexts (list[str] | None): Contexts retrieved by the RAG system,
            required by Faithfulness, ResponseGroundedness metrics.
        user_input (str | None): Original user query, 
            required by AnswerAccuracy, AnswerRelevancy, Faithfulness metrics.
        reference (str | None): Expected reference answer, 
            required by AnswerAccuracy metrics.
        embedding (bool): Whether to initialize embeddings for metrics, 
            required by AnswerRelevancy metrics.

    Returns:
        MetricResult: RAGAS metric evaluation result.
    """
    scorer = prepare_ragas_llm_scorer(metric=metric, llm_config_path=llm_config_path, embedding=embedding)

    # Different generation metrics require different inputs.
    # Start with the common arguments and add user_input only when required.    
    kwargs = {
        "response": response
    }
 
    if user_input:
        kwargs["user_input"] = user_input

    if reference:
        kwargs["reference"] = reference

    if retrieved_contexts:
        kwargs["retrieved_contexts"] = retrieved_contexts

    # Evaluate metric using the prepared arguments.
    return await scorer.ascore(**kwargs)


def save_evaluation_result(row: dict, run_name: str):
    """Append an evaluation result to CSV, creating the file if needed."""
    csv_path = (E1_RESULTS / f"{run_name}_{datetime.now():%Y-%m-%d}.csv")

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "query_id",
        "query",
        "k",
        "latency_s",
        "metric",
        "score",
        "status",
        "error_type",
        "judge_max_tokens",
        "judge_input_tokens",
        "judge_output_tokens",
        "judge_total_tokens",
        "judge_call_count",
        "judge_avg_output_tokens_per_call",
        "judge_max_output_tokens_per_call",
        "trace_id"
    ]

    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)