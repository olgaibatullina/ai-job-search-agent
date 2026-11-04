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


## EL-007 — Faithfulness judge fails on large retrieved context

### Date

2026-10-23 14:18

### Context / Findings

While adding RAGAS LLM-based generation evaluation, `Faithfulness` failed with:

```
IncompleteOutputException:
The output is incomplete due to a max_tokens length limit.
```

The failure initially occurred during _create_statements(), where the judge LLM generates structured statements from the model response. After increasing the judge max_tokens, this stage completed, but the evaluation subsequently failed during _create_verdicts() when generating NLIStatementOutput.

The judge configuration was:
```
model: qwen3:8b
temperature: 0.01
max_tokens: 1024 → 2048 → 4096
```

The generated application response was approximately 3,407 characters. The Faithfulness judge receives the generated response together with the retrieved contexts. With k=5, the resulting evaluation prompt was large (approximately 7,885 cached prompt tokens were reported by the API).

### Validation / Results

Reducing the retrieved context from k=5 to k=1 allowed the Faithfulness evaluation to complete successfully.

### Impact / Takeaway

The failure is caused by the combination of a large retrieved context and the local judge model's structured-output generation constraints, rather than by the Faithfulness metric itself.

The retrieval k should not automatically be reduced in the RAG pipeline solely to make the evaluator pass, because that would change the system behavior being evaluated. The evaluation strategy should instead define how much retrieved context can be reliably processed by the selected judge model.

This raises an evaluation-design question: how much retrieved context can the judge reliably process without degradation or evaluation errors?

## EL-008 — LangSmith integration for LLM judge tracing

### Date

2026-10-30 11:30

### Context / Findings

RAGAS evaluation failures required deeper visibility into judge LLM calls, token usage, and execution behavior.

### Decision / Approach

Integrated LangSmith tracing using wrap_openai() with the existing Ollama/OpenAI-compatible client.

Added parent traces for individual evaluation cases and linked them to CSV results using trace_id.

### Validation / Results

Successfully captured nested LLM calls, token usage, latency, and execution status.

Implemented CSV enrichment to retrieve judge token usage from LangSmith.

### Impact / Takeaway

LLM evaluation failures can now be investigated at the individual model-call level.


## EL-009 — Judge output token limit confirmed through tracing

### Date

2026-10-30 16:30

### Context / Findings

LangSmith tracing revealed that a failed Faithfulness evaluation contained an LLM call with:

- Input tokens: 3,117
- Output tokens: 4,096
- Finish reason: length

The call reached the configured output token limit of 4,096, producing incomplete output and causing IncompleteOutputException.

### Validation / Results

Confirmed that max_tokens limits generated output per LLM call, not total tokens across the evaluation trace.

### Impact / Takeaway

The observed failure is directly associated with judge output truncation. Reducing input context may help, but its effect on output token consumption still requires validation.

## EL-010 — Planned context optimization experiment

### Date

2026-10-30 17:50

### Context / Findings

Current RAGAS evaluation passes full job descriptions as retrieved_contexts, potentially increasing judge input size and structured-output generation complexity.

### Decision / Approach

Investigate replacing full job descriptions with shorter relevant retrieved chunks.

Compare the same evaluation cases before and after context reduction, controlling for changes in generated responses.

### Validation / Results

Status: Planned — optimization results not yet available.

Measure:

- Judge input, output, and total tokens
- Evaluation failure rate
- Faithfulness scores
- Evaluation latency

### Impact / Takeaway

Determine whether reducing evaluation context improves judge reliability and efficiency without compromising evaluation quality.


## EL-011 — Judge reasoning loop causes output token exhaustion

### Date
2026-10-30 19:50

### Context / Findings

During E1 Faithfulness evaluation, query `neg020` failed with
`IncompleteOutputException` despite receiving a very short
generator response: "No relevant jobs found."

LangSmith tracing revealed:

- Evaluation stage: Statement Generation
- Input tokens: 504
- Output tokens: 4096
- Finish reason: `length`
- Generated content: empty
- Reasoning: repetitive analysis of the same simple statement

The judge repeatedly reconsidered how to rephrase
"No relevant jobs found" without producing the required JSON.

### Root Cause Analysis

The trace confirmed a reasoning loop in Qwen3 8B.

The model exhausted its output token budget on repetitive
reasoning rather than producing the expected structured output.

This demonstrates that Faithfulness failures are not necessarily
caused by large retrieved contexts or long generator responses.

### Impact / Takeaway

Identified a distinct LLM-as-a-Judge failure mode:

**Reasoning loop → Output token exhaustion → IncompleteOutputException**

Reducing retrieved context alone cannot address this failure mode.

### Next Steps

- Test disabling Qwen3 thinking mode for the judge.
- Compare token consumption, structured-output validity,
  failure rates, and evaluation scores.
- Verify that disabling reasoning does not degrade judge accuracy.


## EL-012 — Repetition Loop with Thinking Disabled

### Date
2026-11-03 17:39

### Context / Findings
- During H6 evaluation, `q010` failed with Thinking OFF.
- All 10 repeated executions failed (10/10).
- Failure occurred during Faithfulness Verdict Generation with 13 statements and a long retrieved context.
- Statement Generation completed successfully.
- The failed LLM call returned no output or token usage in LangSmith.

### Root Cause Analysis
- Ollama returned HTTP 500:
  `prediction aborted, token repeat limit reached`.
- Instructor exhausted its retry attempts and raised `InstructorRetryException`.
- The underlying trigger of repetitive generation remains unknown.

### Impact
- Disabling thinking does not eliminate all repetition loops.
- Repetition failures can occur independently of explicit reasoning.
- Such failures prevent Faithfulness evaluation from completing.

### Next Steps
- Include `q010` in H7 regression cases.
- Investigate sampling parameters and repetition penalties.
- Verify mitigation across the complete evaluation dataset.