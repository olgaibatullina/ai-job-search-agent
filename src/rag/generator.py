"""
Generator for the RAG pipeline.

This module prepares the retrieved job context and generates answers
using an LLM via Ollama.

Prerequisites:
- Install Ollama (https://ollama.com/download)
- Pull the required model in the terminal:
  ollama pull qwen3:8b
"""

import yaml

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from src.rag.loader import load_documents
from src.rag.chunker import split_text
from src.rag.vector_store import create_vector_store
from src.rag.retriever import retrieve_relevant_jobs
from src.jobs import loader
from config.paths import RAW_JOBS
from config.paths import LLM_CONFIG


def prepare_context(retrieved_chunks: list[Document]) -> list[Document]:
    """Build full job documents from retrieved chunks.

    Args:
        retrieved_chunks: Retrieved document chunks containing job IDs.

    Returns:
        Full job documents corresponding to the retrieved job IDs.
    """
    jobs = loader.get_jobs_from_chunks(retrieved_chunks=retrieved_chunks)

    return [Document(
        page_content=job["posting"]["description"],
        metadata={
            "job_id": job["job_id"],
            "title": job["job"]["title"],
            "company": job["job"]["company"],
        },
    ) 
    for job in jobs ]
   

def generate_answer(
    query: str,
    context: list[Document],
    prompt: ChatPromptTemplate,
    llm: ChatOllama
) -> str:
    """Generate an answer using the query and retrieved job context.

    Args:
        query: User's job search query.
        context: Full job descriptions retrieved for the query.
        prompt: Chat prompt template used to construct the LLM input.
        llm: ChatOllama model used to generate the answer.

    Returns:
        Generated answer from the LLM.
    """
    prompt_value = prompt.invoke({
        "query": query,
        "context": context
    })

    return llm.invoke(prompt_value).content


def main():
    # Prepare vector store from RAW_JOBS.
    documents = load_documents(RAW_JOBS)
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = create_vector_store(split_text(documents), embedding_model)

    # Load LLM configuration and prepare LLM and prompts.
    with LLM_CONFIG.open("r", encoding="utf-8") as file:
        llm_config = yaml.safe_load(file)
    
    llm = ChatOllama(
        model=llm_config["generator"]["model"],
        temperature=llm_config["generator"]["temperature"],
    )

    query = "Find AI eval jobs"
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            llm_config["generator"]["system_prompt"]
        ),
        (
            "user",
            llm_config["generator"]["user_prompt"]
        )
    ])

    # Retrieve relevant context.
    context = prepare_context(retrieve_relevant_jobs(vector_store, query, k=3))

    # Augument prompt and generate LLM asnwer.
    llm_response = generate_answer(
        query=query, 
        context=context, 
        prompt=prompt, 
        llm=llm)

    print(llm_response)


if (__name__ == "__main__"):
    main()