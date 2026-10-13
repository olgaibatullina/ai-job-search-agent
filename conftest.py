"""Shared fixtures for evaluation tests."""

import pytest
import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config.paths import RAW_JOBS
from config.paths import FAISS_INDEX
from src.rag.loader import load_documents
from src.rag.chunker import split_text
from src.rag.vector_store import create_vector_store


@pytest.fixture(scope="session")
def job_index():
    """
    Prepare a FAISS index for the retrieval evaluation test session.

    Loads an existing local FAISS index if available. Otherwise, loads the
    raw job data, splits the documents, creates embeddings, builds the index,
    and saves it locally for reuse across future test runs.

    Note:
        The current implementation only checks whether the FAISS index
        exists. It does not yet validate whether the index is up to date
        with the current raw job data, chunking configuration, or embedding
        model. A fingerprint/version mechanism should be added later to
        detect stale indexes and trigger a rebuild when these inputs change.

    Yields:
        FAISS: A FAISS index containing the job documents for retrieval tests.
    """

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Load the existing FAISS index if available; else create and save a new index.     
    if FAISS_INDEX.exists():       
        job_index = FAISS.load_local(FAISS_INDEX, embedding_model, allow_dangerous_deserialization=True)
        
    else:       
        documents = load_documents(RAW_JOBS)      
        job_index = create_vector_store(split_text(documents), embedding_model)
        job_index.save_local(FAISS_INDEX)

    yield job_index


@pytest.fixture(scope="session")
def all_job_ids():
    """Load all job IDs from the raw jobs dataset.

    Yields:
        list[str]: All job IDs from the raw jobs dataset.
    """
    all_job_ids = []
    with RAW_JOBS.open("r", encoding="utf-8") as file:
        jobs = json.load(file)
    
        for job in jobs:
            all_job_ids.append(job["job_id"])

    yield all_job_ids