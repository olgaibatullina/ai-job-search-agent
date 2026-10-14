"""
Job data loader for the RAG pipeline.

This module is part of the AI Job Search Agent RAG pipeline.
It prepares raw job data for further processing and retrieval.

V1: Reads job postings from raw_jobs.json and converts each
posting into a LangChain Document with job metadata.
"""

import json
from pathlib import Path
from langchain_core.documents import Document


def load_raw_jobs(path: Path) -> list[dict]:
    """Load and parse job postings from a JSON file.

    Args:
        path (Path): Path to the JSON file.
    Returns:
        list[dict]: A list of job postings.
    """
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def job_to_document(job: dict) -> Document:
    """Convert a job posting dictionary into a LangChain Document.

    Args:
        job: Job posting data.

    Returns:
        A LangChain Document containing the job posting and metadata.
    """   
    return Document(
        page_content=job["posting"]["description"],
        metadata={
            "job_id": job["job_id"],
            "title": job["job"]["title"],
            "company": job["job"]["company"],
        },
    )


def load_documents(path: Path) -> list[Document]:
    """Load job postings from a JSON file and convert them to Documents.

    Args:
        path: Path to the JSON file.

    Returns:
        A list of LangChain Documents.
    """
    jobs = load_raw_jobs(path)
    
    return [job_to_document(job) for job in jobs]