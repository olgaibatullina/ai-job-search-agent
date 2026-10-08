"""
Vector store builder for RAG pipeline.

This module is part of the AI Job Search Agent RAG pipeline.
It creates a FAISS vector store from document chunks and their
embeddings for similarity search and retrieval.

V1: Uses a Hugging Face Sentence Transformers model through LangChain
to convert text chunks into numerical vector representations.
"""

from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.rag.loader import load_documents
from src.rag.chunker import split_text


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def embed_documents(
        documents: list[Document], 
        embedding_model: HuggingFaceEmbeddings
        ) -> list[list[float]]:
    """Generate embeddings for document chunks.

    Args:
        documents: A list of document chunks.
        embedding_model: The LangChain embedding model used to generate vectors,
currently backed by Hugging Face Sentence Transformers.
    Returns:
        A list of embedding vectors.
    """

    return embedding_model.embed_documents([chunk.page_content for chunk in split_text(documents)])

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


if __name__ == "__main__":

    documents = load_documents(
        Path("./data/raw_jobs/raw_jobs.json")
    )
    vectors =  embed_documents(documents, embedding_model)

    vector_store = create_vector_store(split_text(documents), embedding_model)

    print(vector_store.index.ntotal)   # number of vectors
    print(vector_store.index.d)        # vector dimension
