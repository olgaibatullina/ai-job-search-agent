# Engineering Log

## EL-001 — Retrieval returns duplicate jobs

### Date

2026-10-08 11:40

### Context / Findings

FAISS retrieves chunks, so multiple chunks from the same job can appear in
top-k results. However, relevance in the golden dataset is defined at the
job level (`job_id`).

### Decision / Approach

Keep relevance annotations at the job level and deduplicate retrieved results
by `job_id`.

### Impact / Takeaway

Job-level annotations remain stable across changes to chunking and embedding,
so the dataset does not need to be re-annotated.

---

## EL-002 — FAISS import and LangChain deprecation path

### Date

2026-10-08 12:30

### Context / Findings

`langchain_community.vectorstores.FAISS` works but is deprecated. The
suggested `langchain_faiss` replacement does not provide the required `FAISS`
class, while `langchain_classic.vectorstores.FAISS` is also deprecated.

Ref: https://github.com/langchain-ai/langchain-community/issues/674

### Decision / Approach

Keep using `langchain_community.vectorstores.FAISS` for now.

### Impact / Takeaway

Accept the warning temporarily and revisit the migration during a future
dependency upgrade.


## EL-003 — Retrieval evaluation label space

### Date

2026-10-08 17:14

### Context / Findings

The initial evaluation used different job sets for Precision and Recall, so the
metrics were calculated over different evaluation spaces.

### Decision / Approach

Use the complete evaluation corpus (`all_jobs`) as the common label space.
For each job, set `reference = 1` if it is relevant and `prediction = 1` if
it was retrieved.

### Impact / Takeaway

Precision, Recall, F1, and Accuracy now use the same `TP / FP / FN / TN`
basis, making the metrics directly comparable.

---

## EL-004 — Job-level deduplication in the retriever

### Date

2026-10-13 12:11

### Context / Findings

FAISS retrieves chunks, so the same job can appear multiple times in top-k.
The retriever should return `k` unique jobs rather than `k` chunks.

### Decision / Approach

Move deduplication into the retriever using an **Expand and Retry** approach:
retrieve extra chunks, keep the first chunk for each `job_id`, and expand the
search if fewer than `k` unique jobs are found.

### Validation / Results

Measure retrieval latency before and after introducing job-level
deduplication.

Compare:

- retrieval latency;
- number of unique jobs returned;
- retrieval evaluation metrics.

The goal is to verify that the new retriever returns `k` unique jobs without
introducing unacceptable retrieval latency.

### Impact / Takeaway

`k` now represents the requested number of unique jobs, not chunks.

## EL-005 — Retrieval quality after job-level deduplication

### Date

2026-10-14 12:14

### Context / Findings

Compared with the baseline, deduplication slightly decreased Precision@5 but
improved Recall@5, F1@5, and Accuracy@5 across 22 evaluation queries.

### Validation / Results

| Metric | Baseline | Dedup | Δ |
|---|---:|---:|---:|
| Precision@5 | 0.7311 | 0.7182 | -0.0129 |
| Recall@5 | 0.1913 | 0.2318 | +0.0406 |
| F1@5 | 0.2919 | 0.3400 | +0.0480 |
| Accuracy@5 | 0.6386 | 0.6489 | +0.0102 |

### Impact / Takeaway

The Precision decrease is an expected trade-off: deduplication retrieves
additional unique jobs, some of which may be irrelevant. The trade-off is
acceptable because job-level retrieval is the intended behavior, and the
overall Recall and F1 improved.

## EL-006 — Retrieval returns results for irrelevant queries

### Date

2026-10-21 13:53

### Context / Findings

Exploratory testing revealed that the retriever returns job results for queries that are not related to job search, including queries such as `"Who is the President of the US?"`.

This indicates that the retriever can return semantically unrelated jobs instead of recognizing that the query has no relevant results.

### Decision / Approach

Add RAGAS-based evaluation tests to systematically detect this retrieval behavior.

Use the existing evaluation dataset, including negative queries, to evaluate retrieval quality and identify cases where irrelevant queries produce retrieved contexts.


### Validation / Results

RAGAS tests are being added to provide automated validation of retrieval behavior.

Retriever improvements will be addressed as the next step after establishing the RAGAS evaluation baseline.

### Impact / Takeaway

The issue is now captured as an explicit retrieval quality problem and will be tracked through automated evaluation rather than exploratory testing alone.