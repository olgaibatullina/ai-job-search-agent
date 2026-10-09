# Engineering Log

## EL-001 — Retrieval returns duplicate jobs

### Date

2026-10-08 11:40

### Observation

FAISS retrieves document chunks, so multiple chunks from the same job can
appear in the top-k results.

Example:

    job_1, job_1, job_2, job_2, job_3

### Problem

The golden dataset defines relevance at the job level (`job_id`), while the
current retrieval operates at the chunk level.

Using chunk-level relevance would couple the evaluation dataset to the current
chunking strategy. Changing chunk size, overlap, or the embedding model could
invalidate the annotations.

### Current approach

Keep golden relevance annotations at the job level.

Deduplicate retrieved results by `job_id` in the evaluation layer.

Binary `0/1` labels are derived at evaluation time from the job-level golden
dataset rather than stored as a separate dataset.

### Planned improvement

Move job-level deduplication into the retriever so that `k` represents the
number of unique jobs returned.

### Validation takeaway

Q: Explain why relevance was annotated at the stable business-entity level
rather than at the implementation-dependent chunk level.

A: The ground truth is maintained at the job level because job-level relevance is stable across changes to chunking and embedding implementation. This allows retrieval configurations to be changed and compared without re-annotating the dataset.


## EL-002 — FAISS import and LangChain deprecation path

### Date

2026-10-08 12:30

### Observation

`FAISS` is available through `langchain_community.vectorstores`, but
LangChain emits a deprecation warning because `langchain-community` is
being sunset.

A migration attempt using `langchain_faiss` failed because the installed
package does not export `FAISS`.

`langchain_classic.vectorstores.FAISS` is also deprecated and recommends
using `langchain_community.vectorstores.FAISS`.

### Problem

FAISS does not currently have a supported standalone integration package
within the new LangChain package structure.

A native `faiss` library is available and could be used directly, but this
would require bypassing LangChain's vector store abstraction and rewriting
the current vector-store and retrieval implementation.

Ref: https://github.com/langchain-ai/langchain-community/issues/674

### Current approach

For the current project stage, keep using:

    from langchain_community.vectorstores import FAISS

The deprecation warning is accepted temporarily. Revisit this migration as
a future dependency/architecture improvement rather than changing the
working retrieval pipeline now.

### Planned improvement

Revisit the FAISS import when a supported standalone integration becomes
available or when the project's dependency versions are intentionally
upgraded.

### Validation takeaway

Explain how dependency migration was investigated instead of blindly
following a suggested replacement: the proposed standalone package did not
expose the required class, while the classic compatibility path was also
deprecated.

## EL-003 — Retrieval Evaluation Label Space

### Date

2026-10-08 17:14

### Observation

The initial retrieval evaluation built binary labels using different sets of jobs for different metrics:

- Precision — based on retrieved jobs.
- Recall — based on relevant jobs.

This produced mathematically valid values, but the metrics were based on different evaluation spaces.

### Problem

The current approach does not provide a single, consistent label space for Precision, Recall, F1, and Accuracy. Jobs that were not retrieved are not represented in the Precision evaluation, while irrelevant jobs that were not retrieved are also excluded.

### Current approach

Keep `relevant_job_ids` as the golden annotation.

The current implementation is retained temporarily as an initial retrieval evaluation and will be replaced after the evaluation design is corrected.

### Planned improvement

Use the complete evaluation corpus (`all_jobs`) as the common evaluation space.

For every job:

- `reference = 1` if the job is relevant according to the golden dataset, otherwise `0`.
- `prediction = 1` if the retriever returned the job, otherwise `0`.

This creates a consistent `TP / FP / FN / TN` representation and allows Precision, Recall, F1, and Accuracy to use the same labels.

### Impact

The initial implementation exposed a limitation in the evaluation design rather than a retrieval-quality problem. The current Precision and Recall results should therefore not be treated as the final retrieval baseline.

### Validation takeaway

Metric results depend on how the evaluation label space is defined. Using different subsets of the corpus for different metrics can produce numbers that are mathematically valid but misleading for the intended retrieval evaluation.
