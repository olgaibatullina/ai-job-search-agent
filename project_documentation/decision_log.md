# Decision Log

## D-001 --- Select Job Search Agent as the Primary Portfolio Project

**Date:** 2026-09-17\
**Status:** Accepted

### Context

The portfolio needs a production-like AI application that is manageable
to build and provides enough scope for LLM, RAG, agent, safety, API, and
regression testing.

Alternatives considered:

-   AI Job Search Agent
-   AI Palm Reading Application
-   AI Financial Analyzer

### Decision

Build the AI Job Search Agent as the product under test and a separate
AI validation framework to evaluate it. Keep the initial agent
relatively simple and do not require model training unless there is a
clear need later.

### Why

Palm Reading would shift the project toward computer vision and provide
less coverage for RAG, tool calling, and grounded factual evaluation.
Financial Analyzer would add substantial financial-domain, numerical,
and safety complexity.

The Job Search Agent provides broad AI validation scenarios without
making the product itself the main source of complexity.

## D-002 --- Candidate Profile and Matching Configuration

**Date:** 2026-09-18\
**Status:** Accepted

### Context

The agent needs candidate data and separate rules for evaluating job
fit.

### Decision

Keep them separate:

-   `candidate_profile.md` --- candidate experience, skills, target
    roles, and preferences.
-   `job_matching_config.yaml` --- matching rules, weights, and scoring
    configuration.

For the first version, the candidate profile is created manually with
GPT from the resume and reviewed by the user. Later, the agent will
generate it automatically from an uploaded resume.

### Why

This keeps the initial implementation focused on job matching and allows
resume extraction to be added and evaluated as a separate AI component
later.

## D-003 --- Local-First Architecture

**Date:** 2026-09-29\
**Status:** Accepted

### Context

The first version does not require production-scale infrastructure, and
keeping the stack local reduces setup cost and complexity.

### Decision

Build and run the initial system locally on Mac rather than AWS.
Backend, LLM, embeddings, vector search, and validation components
should run locally where possible. AWS deployment can be added later as
a separate production-like stage.

### Why

Local development keeps the initial implementation simpler and avoids
cloud costs while leaving AWS deployment as a later architecture step.

## D-004 --- Local Vector Search with FAISS

**Date:** 2026-09-29\
**Status:** Accepted

### Context

The initial RAG workload requires local vector similarity search but
does not yet need a full vector database.

### Decision

Use **FAISS** for the first version of vector similarity search. It is
free, open-source, lightweight, and suitable for local RAG. If needed
later, migrate to **PostgreSQL + pgvector** for a more production-like
architecture.

### Why

FAISS provides the similarity-search capability needed for the first
local implementation without adding database infrastructure.

## D-005 --- Embedding Model Benchmark

**Date:** 2026-09-30\
**Status:** Accepted

### Context

Embedding model selection affects retrieval quality and latency, so the
model should be selected based on measurements rather than subjective
preference.

### Decision

Benchmark three embedding models on the same retrieval dataset:

-   `sentence-transformers/all-MiniLM-L6-v2` --- lightweight baseline.
-   `BAAI/bge-small-en-v1.5` --- English retrieval-oriented model.
-   `intfloat/e5-small-v2` --- retrieval-oriented model using `query:`
    and `passage:` prefixes.

Compare them using **Recall@K, Precision@K, and latency** before
selecting the embedding model for RAG.

For local execution: - Run `all-MiniLM-L6-v2` through Ollama
(`all-minilm:l6-v2`). - Run `bge-small-en-v1.5` through the Ollama build
`qllama/bge-small-en-v1.5`. - Run `e5-small-v2` directly with
`sentence-transformers` if an equivalent Ollama build is not available.

Do not replace E5 with another model only to standardize the runtime.
Keep the benchmark focused on the selected models and use the most
direct available execution method for each.

### Why

Using the same dataset and metrics provides an evidence-based model
choice and creates a reproducible retrieval benchmark for future
regression testing. Runtime consistency is less important than comparing
the intended embedding models.

## D-006 — Select Embedding Model for V1

**Date:** 2026-10-05\
**Status:** Accepted\
**Supersedes:** D-005 and previous D-006 embedding model decisions

### Context

The project needs a practical local embedding model for the initial RAG implementation. Embedding model benchmarking is not a primary focus of the project; the main validation focus is RAG quality and retrieval behavior.

### Decision

Use `sentence-transformers/all-mpnet-base-v2` as the embedding model for V1.
The model will be used locally through LangChain's embedding integration. It does not require special query/document prefixes.
Embedding model benchmarking may be added later as a separate validation experiment if needed.

### Why

The model is suitable for English text retrieval, can run locally, and keeps the initial implementation simple while allowing the project to focus on RAG and AI validation rather than embedding model selection.


## D-007 — Separate Raw and Public Evaluation Datasets

**Date:** 2026-10-08  
**Status:** Accepted

### Context

The project needs original job API responses for local development and retrieval evaluation, while the public GitHub repository needs a reproducible evaluation dataset that does not publish full original job descriptions.

### Decision

Maintain two distinct datasets:

- **Raw dataset:** Preserve `raw_jobs.json` with original API response text, unchanged, for local development and retrieval evaluation. Exclude it from the public Git repository.
- **Public evaluation dataset:** Prepare a smaller, separately identified dataset suitable for publication on GitHub and for reproducible public CI checks, without full original job descriptions.

Keep evaluation results, dataset identifiers, and ground-truth annotations separate. Never silently substitute the public dataset for the raw dataset or present metrics from one as results for the other.

Before publishing the repository, verify that the raw dataset is excluded from Git and remove any previously committed copies from repository history.

### Why

This preserves original test evidence while enabling public reproducibility without exposing full third-party job descriptions. Explicit dataset provenance prevents misleading evaluation comparisons and CI results.