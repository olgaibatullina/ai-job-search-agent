"""RAG pipeline step 4: Document retriever.

This module is part of the AI Job Search Agent RAG pipeline.
It performs similarity search over the FAISS vector store and returns
the most relevant document chunks for further processing and retrieval.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.rag import loader, chunker, vector_index
from config import config, paths


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


def retrieve_relevant_jobs(
    vector_store: FAISS, 
    query: str, 
    k: int
) -> list[Document]:
    """Retrieve the top-k unique jobs using semantic similarity search.

    The function retrieves more vectors than the requested number of jobs
    because multiple vectors may belong to the same job. Retrieved jobs are
    deduplicated by job_id while preserving their semantic ranking order.

    The search range is gradually expanded until k unique jobs are found
    or all vectors in the index have been searched.

    Args:
        vector_store: FAISS vector store containing job embeddings.
        query: Search query used for semantic similarity retrieval.
        k: Number of unique jobs to return.

    Returns:
        A list of up to k unique jobs ranked by semantic similarity.
    """

    unique_jobs = []            
    unique_job_ids = []

    multiplier = 3
    total_vectors = vector_store.index.ntotal

    while (len(unique_jobs) < k ):
        # Retrieve more vectors than the requested number of jobs.
        # This compensates for multiple vectors belonging to the same job.
        search_k = min(k * multiplier, total_vectors)
        
        retrieved_jobs = retrieve_from_vector_store(vector_store=vector_store, query=query, k=search_k)

        # Deduplicate jobs while preserving the order returned by FAISS.
        for job in retrieved_jobs:
            job_id = job.metadata.get("job_id")
            if job_id not in unique_job_ids:
                unique_job_ids.append(job_id)
                unique_jobs.append(job)
        
        # Stop when the entire vector index has been searched.
        if search_k == total_vectors:
            break
  
        # Expand the search range if not enough unique jobs were found.
        multiplier *= 2 

    return unique_jobs[:k]


def calculate_keyword_score(
    job: Document,
    keywords: list[list[str]],
) -> int:
    """Calculate how many keyword groups are matched by a job.

    Keywords are organized into groups representing alternative terms.
    A job matches a group when at least one keyword from that group
    appears in the job description.

    Therefore:
        - OR logic is applied within each keyword group.
        - Groups contribute independently to the final score.

    For example, three keyword groups can produce a score from 0 to 3.

    Args:
        job: Job document to evaluate.
        keywords: Keyword groups used for reranking.

    Returns:
        Number of keyword groups matched by the job.
    """
    
    text = job.page_content.lower()
    matched_groups = 0

    for group in keywords:
        group_matched = False

        # A single matching keyword is sufficient to match the group.
        for keyword in group:
            if keyword.lower() in text:
                group_matched = True
                break

        if group_matched:
            matched_groups += 1

    return matched_groups


def rerank_by_keywords(
    jobs: list[Document],
    keywords: list[list[str]],
) -> list[Document]:
    """Rerank retrieved jobs according to keyword-group matches.

    Jobs with more matched keyword groups are placed higher in the ranking.
    The function is applied only after semantic retrieval has produced an
    expanded candidate pool.

    Args:
        jobs: Candidate jobs retrieved by semantic similarity.
        keywords: Keyword groups used to calculate the reranking score.

    Returns:
        Jobs sorted by descending keyword-match score.
    """

    return sorted(
        jobs,
        key=lambda job: calculate_keyword_score(job, keywords),
        reverse=True,
    )


def retrieve_relevant_jobs_hybrid(
    vector_store: FAISS,
    query: str,
    k: int,
    keywords: list[list[str]],
) -> list[Document]:
    """Retrieve jobs using semantic search followed by keyword reranking.

    An expanded semantic candidate pool is retrieved first. Keyword-aware
    reranking is then applied to this larger pool before the final k jobs
    are selected.

    This allows a job that is not present in the semantic top-k results
    but appears in the larger candidate pool to be promoted when it
    matches query-specific technical keywords.

    Args:
        vector_store: FAISS vector store containing job embeddings.
        query: Search query used for semantic retrieval.
        k: Number of final jobs to return.
        keywords: Keyword groups used for reranking.

    Returns:
        A list of up to k unique jobs after semantic retrieval and
        keyword-aware reranking.
    """

    unique_jobs = []
    unique_job_ids = set()

    multiplier = 3
    
    # Retrieve a larger candidate pool than the final K.
    # This gives keyword reranking enough candidates to change the
    # composition of the final result set, not just its ordering.
    candidate_k = k * 5
    total_vectors = vector_store.index.ntotal

    while len(unique_jobs) < candidate_k:
        # Search a progressively larger semantic candidate pool.
        search_k = min(candidate_k * multiplier, total_vectors)

        retrieved_jobs = retrieve_from_vector_store(
            vector_store=vector_store,
            query=query,
            k=search_k,
        )

        # Deduplicate jobs while preserving semantic retrieval order.
        for job in retrieved_jobs:
            job_id = job.metadata.get("job_id")

            if job_id not in unique_job_ids:
                unique_job_ids.add(job_id)
                unique_jobs.append(job)

        # Stop if the entire vector index has been searched.
        if search_k == total_vectors:
            break

        # Increase the search range until enough unique candidates
        # have been collected.
        multiplier *= 2

    # Apply keyword reranking before selecting the final K results.
    reranked_jobs = rerank_by_keywords(
        jobs=unique_jobs,
        keywords=keywords,
    )

    return reranked_jobs[:k]


def main():
    documents = loader.load_documents(paths.PROCESSED_JOBS)
    
    embedding_model = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODELS["mpnet"]["model"])
    vector_store = vector_index.create_vector_store(chunker.split_text(documents), embedding_model)

    query = "Evaluation engineer"

    results = retrieve_from_vector_store(vector_store, query, k=2)

    for i, result in enumerate(results):
        print(f"{i}:\n{result.page_content},\n\n")


if (__name__ == "__main__"):
    main()