"""Vector store builder for RAG pipeline.

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

from src.rag import loader, chunker


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

    # FAISS.from_documents() embeds each chunk using the provided
    # embedding model and stores the resulting vectors with the documents.
    return FAISS.from_documents(documents=chunks, embedding=embedding_model)


def main():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    documents = loader.load_documents(
        Path("./data/raw_jobs/raw_jobs.json")
    )

    # Split full job descriptions into chunks before embedding and indexing.
    vector_store = create_vector_store(chunker.split_text(documents), embedding_model)

    # Number of vectors stored in the FAISS index and dimension of each embedding vector.
    print(vector_store.index.ntotal)
    print(vector_store.index.d)


if __name__ == "__main__":
    main()