"""Helper functions for retrieval evaluation and label generation."""

import yaml
from pathlib import Path


def load_retrival_dataset(dataset: Path) -> list[dict]:
    """Load retrieval evaluation queries from a YAML dataset.

    Args:
        dataset: Path to the YAML dataset.

    Returns:
        A list of query dictionaries.
    """
    with open(dataset, "r") as file:
        data = yaml.safe_load(file)

    return data["queries"]


def precision_labels(retrieved_jobs: list[str], relevant_jobs: list[str]) -> tuple[list[int], list[int]]:
    """Build prediction and reference labels for retrieval Precision.

    Args:
        retrieved_jobs: Job IDs returned by the retriever.
        relevant_jobs: Job IDs marked as relevant in the golden dataset.

    Returns:
        A tuple containing prediction and reference labels.
    """
    relevant_jobs = set(relevant_jobs)  # Convert to set for faster lookups
    retrieved_jobs = set(retrieved_jobs)  # Convert to set to remove duplicates. Ref.: project_documentation/engineering_log.md EL-001
    
    predictions = [1] * len(retrieved_jobs)
    references = []
    
    for job in retrieved_jobs:
        references.append(1) if job in relevant_jobs else references.append(0)

    return predictions, references


def recall_labels(retrieved_jobs: list[str], relevant_jobs: list[str]) -> tuple[list[int], list[int]]:
    """Build prediction and reference labels for retrieval Recall.

    Args:
        retrieved_jobs: Job IDs returned by the retriever.
        relevant_jobs: Job IDs marked as relevant in the golden dataset.

    Returns:
        A tuple containing prediction and reference labels.
    """
    relevant_jobs = set(relevant_jobs)  # Convert to set for faster lookups
    retrieved_jobs = set(retrieved_jobs)  # Convert to set to remove duplicates. Ref.: project_documentation/engineering_log.md EL-001

    predictions = []
    references = [1] * len(relevant_jobs)
    
    for job in relevant_jobs:
        predictions.append(1) if job in retrieved_jobs else predictions.append(0)

    return predictions, references