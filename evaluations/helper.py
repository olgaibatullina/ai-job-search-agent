"""Shared helper functions for evaluation and metric calculation."""

import yaml

from pathlib import Path
from datasets import Dataset

from openai import AsyncOpenAI

from ragas import SingleTurnSample
from ragas.llms import llm_factory


def load_retrieval_dataset(dataset: Path) -> list[dict]:
    """Load retrieval evaluation queries from a YAML dataset.

    Args:
        dataset: Path to the YAML dataset.

    Returns:
        A list of query dictionaries.
    """
    with dataset.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return data["queries"]


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


def calculate_ragas_legacy_metric(
        metric, 
        query: str, 
        retrieved_contexts: list[str], 
        reference_contexts: list[str]
    ):
    """Calculate a RAGAS metric using the legacy evaluate API.

    This helper is used to compare legacy RAGAS metrics with their
    modern Collections API counterparts.    

    Args:
        metric: RAGAS metric to evaluate.
        query: User query.
        retrieved_contexts: Contexts retrieved by the system.
        reference_contexts: Reference contexts expected for the query.

    Returns:
        Evaluation result returned by RAGAS.
    """
    sample = SingleTurnSample(
        user_input=query,
        retrieved_contexts=retrieved_contexts,
        reference_contexts=reference_contexts,
    )

    return metric.single_turn_score(sample)


async def calculate_ragas_metric_llm_judge(metric, query: str, retrieved_contexts: list[str], reference: str, llm_config_path: Path):
    with llm_config_path.open("r", encoding="utf-8") as file:
        llm_config = yaml.safe_load(file)

    # Ollama exposes OpenAI-compatible API
    client = AsyncOpenAI(
        api_key = "ollama",
        base_url = llm_config["judge"]["base_url"]
    )
    llm = llm_factory(
        llm_config["judge"]["model"], 
        provider = "openai", 
        client = client            
    )

    # Create metric
    scorer = metric(llm=llm)

    # Evaluate
    return await scorer.ascore(       
        user_input = query, 
        reference = reference,
        retrieved_contexts = retrieved_contexts
    )