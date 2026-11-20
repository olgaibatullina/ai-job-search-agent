"""Shared fixtures for evaluation tests."""

import pytest
import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import config, paths
from src.rag import loader, chunker, vector_index


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

    embedding_model = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODELS["mpnet"]["model"]
    )

    # Load the existing FAISS index if available; else create and save a new index.     
    if paths.FAISS_INDEX.exists():       
        job_index = FAISS.load_local(paths.FAISS_INDEX, embedding_model, allow_dangerous_deserialization=True)
        
    else:       
        documents = loader.load_documents(paths.RAW_JOBS)      
        job_index = vector_index.create_vector_store(chunker.split_text(documents), embedding_model)
        job_index.save_local(paths.FAISS_INDEX)

    yield job_index


@pytest.fixture(scope="session")
def all_jobs():
    """Load all jobs from the raw jobs dataset.

    Yields:
        list[str]: All jobs from the raw jobs dataset.
    """

    with paths.RAW_JOBS.open("r", encoding="utf-8") as file:
        yield json.load(file)