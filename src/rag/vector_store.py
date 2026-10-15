"""
Vector store builder for RAG pipeline.

This module is part of the AI Job Search Agent RAG pipeline.
It creates a FAISS vector store from document chunks using
a Hugging Face Sentence Transformers embedding model.

V1: Uses LangChain's FAISS integration to embed document chunks
and store their vector representations for similarity search.
"""

from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.rag.loader import load_documents
from src.rag.chunker import split_text


def create_vector_store(
    chunks: list[Document],
    embedding_model: HuggingFaceEmbeddings,
) -> FAISS:
    """Create a FAISS vector store from document chunks.

    Args:
        chunks: A list of document chunks to be embedded and stored.
        embedding_model: The embedding model used to generate vectors.

    Returns:
        A FAISS vector store containing the document chunks and their embeddings.
    """
    
    return FAISS.from_documents(documents=chunks, embedding=embedding_model)


def main():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    documents = load_documents(
        Path("./data/raw_jobs/raw_jobs.json")
    )

    vector_store = create_vector_store(split_text(documents), embedding_model)

    print(vector_store.index.ntotal)   # number of vectors
    print(vector_store.index.d)        # vector dimension


if __name__ == "__main__":
    main()