"""
Job posting chunker for RAG pipeline.

This module is part of the AI Job Search Agent RAG pipeline.
It splits job posting Documents into smaller chunks for embedding and subsequent vector similarity search.
"""

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.rag import loader


def split_text(documents: list[Document]) -> list[Document]:
    """Split Documents into smaller text chunks using recursive separators.

    Args:
        documents: A list of job posting Documents.

    Returns:
        A list of Documents containing the resulting text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        separators=["\n\n", "\n", " "]
        )

    return text_splitter.split_documents(documents)


def main():
    documents = loader.load_documents(
        Path("./data/raw_jobs/raw_jobs.json")
    )
    
    chunks = split_text(documents)

    print(len(chunks))
    for i, chunk in enumerate(chunks[:21]):
        print(f"{i}:\n{chunk.page_content}\n")


if __name__ == "__main__":
    main()