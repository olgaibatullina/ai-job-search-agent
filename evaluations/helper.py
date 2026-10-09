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

    relevant_jobs = set(relevant_jobs)  # Convert to set for faster lookups
    retrieved_jobs = set(retrieved_jobs)  # Convert to set to remove duplicates. Ref.: project_documentation/engineering_log.md EL-001

    for job_id in all_job_ids:
        references.append(1) if job_id in relevant_jobs else references.append(0)
        predictions.append(1) if job_id in retrieved_jobs else predictions.append(0)    
    
    return predictions, references