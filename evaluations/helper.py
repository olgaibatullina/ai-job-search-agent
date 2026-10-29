"""Shared helper functions for evaluation and metric calculation."""

import yaml

from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from openai import AsyncOpenAI

from ragas import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics.base import Metric
from ragas.metrics.collections import BaseMetric
from ragas.metrics.result import MetricResult


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
        ("system", llm_config["generator"]["system_prompt"]),
        ("user", llm_config["generator"]["user_prompt"]),
    ])
    
    return llm, prompt


def prepare_ragas_llm_scorer(metric: type[BaseMetric], llm_config_path: Path) -> BaseMetric:
    """Prepare a RAGAS metric with the configured LLM judge.

    Args:
        metric (type[BaseMetric]): RAGAS metric class to instantiate with the LLM judge.
        llm_config_path (Path): Path to the YAML file containing judge LLM configuration.

    Returns:
        BaseMetric: Instantiated RAGAS metric configured with the LLM judge.
    """
    llm_config = load_retrieval_dataset(llm_config_path)
    
    # Ollama exposes an OpenAI-compatible API.
    # Create an async OpenAI client to communicate with the Ollama model server.
    client = AsyncOpenAI(
        api_key = "ollama",
        base_url = llm_config["judge"]["base_url"]
    )
    
    # Create a RAGAS-compatible LLM wrapper using the configured judge model.
    # RAGAS uses this LLM to evaluate the selected metric.
    llm = llm_factory(
        llm_config["judge"]["model"], 
        provider = "openai", 
        client = client,
        max_tokens = llm_config["judge"]["max_tokens"]
    )

    # Instantiate selected RAGAS metric with the configured LLM judge.
    return metric(llm=llm)


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
    
    # No reference contexts means the reference-based metric is undefined for this case.
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
    retrieved_contexts: list[str], 
    llm_config_path: Path, 
    user_input: str | None = None
    ) -> MetricResult:
    """Calculate a RAGAS LLM-based generation metric.

    Args:
        metric (type[BaseMetric]): RAGAS generation metric class to evaluate.
        response (str): Response generated by the RAG system.
        retrieved_contexts (list[str]): Contexts retrieved by the RAG system.
        llm_config_path (Path): Path to the YAML file containing judge LLM configuration.
        user_input (str | None): Original user query, required only by metrics
            that use it, such as Faithfulness.

    Returns:
        MetricResult: RAGAS metric evaluation result.
    """
    scorer = prepare_ragas_llm_scorer(metric=metric, llm_config_path=llm_config_path)

    # Different generation metrics require different inputs.
    # Start with the common arguments and add user_input only when required.    
    kwargs = {
        "response": response,
        "retrieved_contexts": retrieved_contexts,
    }
 
    if user_input is not None:
        kwargs["user_input"] = user_input

    # Evaluate metric using the prepared arguments.
    return await scorer.ascore(**kwargs)