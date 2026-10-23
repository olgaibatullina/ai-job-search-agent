"""Utilities for loading full job records from the raw jobs dataset."""

import json

from langchain_core.documents import Document
from config.paths import RAW_JOBS


def get_jobs_from_chunks(retrieved_chunks: list[Document]) -> list[dict]:
    """Get full job records corresponding to retrieved document chunks.

    Args:
        retrieved_chunks: Retrieved document chunks containing job IDs in
            their metadata.

    Returns:
        Full job records whose IDs match the retrieved chunks.
    """
    retrieved_job_ids = {
        doc.metadata["job_id"]
        for doc in retrieved_chunks
    }

    with RAW_JOBS.open("r", encoding="utf-8") as file:
        jobs = json.load(file)

    return [
        job for job in jobs
        if job["job_id"] in retrieved_job_ids
    ]