"""
Document retriever for the RAG pipeline.

This module is part of the AI Job Search Agent RAG pipeline.
It performs similarity search over the FAISS vector store and returns
the most relevant document chunks for further processing and retrieval.
"""

from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.rag.loader import load_documents
from src.rag.chunker import split_text
from src.rag.vector_store import create_vector_store


def retrieve_from_vector_store(
        vector_store: FAISS, 
        query: str, 
        k: int
        ) -> list[Document]:
    """Retrieve the most relevant documents for a query.

    Args:
        vector_store: The FAISS vector store used for similarity search.
        query: The search query.
        k: The number of documents to retrieve.

    Returns:
        A list of the most relevant Documents.
    """
    return vector_store.similarity_search(query=query, k=k)

def main():
    documents = load_documents(
            Path("./data/raw_jobs/raw_jobs.json")
        )
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = create_vector_store(split_text(documents), embedding_model)

    query = "Evaluation engineer"

    results = retrieve_from_vector_store(vector_store, query, k=2)

    for i, result in enumerate(results):
        print(f"{i}:\n{result.page_content},\n\n")


if (__name__ == "__main__"):
    main()