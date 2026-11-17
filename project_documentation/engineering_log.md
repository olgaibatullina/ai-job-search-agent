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

## EL-013 — H7: Judge Reliability Improvement Through Sampling Parameters and Repetition Penalty

### Date
2026-11-04 17:20

### Context / Findings
- Completed H7 evaluation of Qwen3 judge sampling parameters with Thinking ON.
- Final configuration: `temperature=0.6`, `top_p=0.95`, `repeat_penalty=1.1`.
- Full regression run completed on all 75 evaluation cases.
- Comparison against the original baseline:
  - Successful evaluations: **68/75 → 75/75**.
  - Failure rate: **9.3% → 0%**.
  - Average output tokens: **2,693 → 2,492** (−7.5%).
  - Average latency: **56.34s → 54.01s**.
- On the 68 commonly successful cases, mean Faithfulness changed from **0.8696 to 0.8645**.
- Manual inspection revealed inconsistent verdict behavior:
  - `typo005`: Faithfulness decreased from 1.0 to 0.4. H7 rejected unsupported relationships accepted by baseline.
  - `q013`: Faithfulness increased from 0.625 to 1.0. H7 accepted several claims not fully supported by the retrieved context.

### Root Cause Analysis
- Sampling parameter adjustments and repetition penalty may reduce repetitive generation and token exhaustion.
- The final configuration eliminated observed evaluation failures in the full regression run.
- Statement Generation remained broadly similar in the inspected cases, while Verdict Generation produced different judgments.
- Sampling parameters may influence how strictly the judge evaluates contextual support.
- The individual contribution of each parameter was not isolated.
- Judge accuracy has not been systematically validated against human-labeled ground truth.

### Impact
- Execution reliability improved significantly, and average token consumption decreased.
- No systematic improvement in Faithfulness scores was demonstrated.
- Higher execution reliability does not guarantee higher judge accuracy.
- Inconsistent verdicts and composite statements remain potential sources of evaluation instability.

### Next Steps
- Evaluate Answer Relevancy using baseline and final H7 configurations.
- Compare failure rates, token consumption, latency, and evaluation scores.
- Document overall E1 findings and remaining limitations.
- Defer judge accuracy validation to a separate investigation using human-labeled ground truth.


## EL-014 — E2: RAGAS ResponseGroundedness NaN Investigation (H1, H2, H4)

### Date
2026-11-04 to 2026-11-06 

### Context / Findings
- Started E2 investigation into NaN results from RAGAS `ResponseGroundedness`.
- **H1 — Partially Confirmed:** LangSmith traces revealed empty or malformed LLM judge outputs (`{}` or empty content).
- **H2 — Partially Confirmed:** Missing or invalid `rating` fields (0, 1, 2) triggered validation errors and retries.
- **H4 — SUCCESS:** Confirmed that `_average_scores()` returns NaN when both judges exhaust their retries without valid scores.

### Root Cause Analysis
- Invalid structured judge outputs trigger repeated LLM calls.
- RAGAS uses two judges, each allowing up to five retry attempts.
- Observed call counts:
  - `q001`: 21 calls, score 1.0, latency 152.85s.
  - `q002`: 40 calls, score NaN, latency 307.25s.
- Excessive retries increase latency without guaranteeing valid verdicts.
- Exact distribution between metric-level and Instructor-level retries remains unverified.

### Impact
- Identified the immediate causes of NaN scores and excessive evaluation latency.
- Confirmed how failed judge evaluations propagate into NaN results.

### Next Steps
- Investigate why the judge produces invalid structured outputs.
- Isolate model behavior from framework and integration effects.
- Continue with H5 and subsequent reliability experiments.


## EL-015 — E2-H5: Structured Output Reliability and Judge Accuracy

### Date
2026-11-09 16:25

### Context / Findings
- Investigated whether disabling reasoning in Qwen3:8b eliminates invalid structured outputs.
- Compared reasoning ON (baseline) and OFF across 75 evaluation cases.
- Results:
  - NaN rate: **29.3% → 0%**.
  - All 75 cases returned valid groundedness scores with reasoning OFF.
  - Mean latency decreased approximately 10x.
- LangSmith trace analysis identified several scoring inconsistencies:
  - `q001`: Potential score inflation (0.5 → 1.0).
  - `q021`: Potential accuracy degradation (0.75 → 0.5).
  - `neg011`: Judge disagreement (1.0 → 0.5).
  - `neg020`: Unsupported geographic claim accepted with reasoning OFF (0.5 → 1.0).
  - `q016`: Baseline run-to-run variability (0.5 → NaN).

### Root Cause Analysis
- Disabling reasoning improved structured output reliability and eliminated observed NaN failures.
- However, some reasoning OFF verdicts contradicted evidence in the retrieved context.
- Higher structured output reliability did not guarantee groundedness scoring accuracy.
- The underlying cause of scoring inconsistencies remains unverified.

### Impact
- **H5 partially supported:** Reliability improved, but the requirement to preserve scoring accuracy was not demonstrated.
- Reasoning OFF was rejected as the preferred solution due to potential score inflation and accuracy degradation.
- Reasoning OFF remains a possible fallback with documented limitations.

### Next Steps
- Investigate alternative structured output fixes in H9.
- Defer systematic judge accuracy validation to E4 using human-labeled ground truth.



## EL-016 — E2-H6: Repetition Penalty Impact on Structured Output Reliability

### Date
2026-11-10 17:30

### Context / Findings
- Investigated whether increasing `repeat_penalty` from 1.0 to 1.1 improves structured output reliability.
- Tested `q001` and `q002` using `qwen3-h7` with `repeat_penalty=1.1`.
- Initial `q002` results:
  - Groundedness score: NaN → 1.0.
  - LLM calls: 40 → 21 (−47.5%).
  - Latency: 358.87s → 177.3s (−50.6%).
- Full regression across 75 cases revealed increased NaN rates:
  - `max_retries=5`: 29.3% → 41.3%.
  - `max_retries=1`: 26.7% → 41.3%.
- Only 11 failed cases overlapped between the two model configurations at `max_retries=5`.

### Root Cause Analysis
- The initial improvement on `q002` did not generalize across the evaluation dataset.
- Repetition penalty did not consistently prevent invalid structured outputs.
- Failure distribution changed between configurations.

### Impact
- **H6 — FAIL:** Increasing `repeat_penalty` did not improve overall reliability.
- The tested configuration increased the NaN rate despite improving the initial failing case.

### Next Steps
- Continue investigating structured output failures at the model and integration layers.


## EL-017 — E2-H8: Retry Reduction Impact on Evaluation Reliability

### Date
2026-11-12 10:30

### Context / Findings
- Investigated whether reducing RAGAS `max_retries` from 5 to 1 improves efficiency without increasing NaN failures.
- Maximum LLM calls decreased from 40 to 8.
- Full regression across 75 cases showed:
  - NaN rate: **26.7% → 29.3%**.
  - With `qwen3-h7`, the NaN rate remained **41.3%** for both configurations.

### Root Cause Analysis
- Fewer retries reduced unnecessary LLM calls but did not resolve invalid structured outputs.
- Some evaluations required additional retries to obtain valid scores.

### Impact
- **H8 — FAIL:** Retry reduction improved efficiency but did not consistently preserve evaluation reliability.
- Reducing retries alone is insufficient to resolve NaN failures.

### Next Steps
- Investigate model-level structured output behavior independently of RAGAS.


## EL-018 — E2-H9: Reasoning and Structured Output Interaction

### Date
2026-11-12 12:30

### Context / Findings
- Investigated whether Qwen3:8b thinking mode interferes with Ollama JSON schema enforcement.
- Tested native Ollama `/api/chat` directly using identical prompts.
- Four configurations: Thinking ON/OFF × Schema ON/OFF.
- Five repetitions per configuration.
- Results:
  - **20/20 valid JSON responses**.
  - No empty `{}` responses.
  - Thinking ON latency: approximately 7.7s.
  - Thinking OFF latency: approximately 0.13s.

### Root Cause Analysis
- Neither thinking mode nor JSON schema enforcement caused invalid outputs in direct Ollama API testing.
- The original RAGAS NaN issue remains unresolved.

### Impact
- **H9 — FAIL / Not Supported:** Native Ollama API consistently returned valid structured outputs.
- Findings suggest further investigation is needed at higher integration layers.

### Next Steps
- Compare native Ollama API with LangChain `ChatOllama` in H10.



## EL-019 — E2-H10: Native Ollama API vs LangChain

### Date
2026-11-12 14:00

### Context / Findings
- Investigated whether LangChain introduces invalid JSON responses when communicating with Ollama.
- Compared native Ollama API and `ChatOllama` using identical prompts, model settings, and JSON schemas.
- Four configurations: Thinking ON/OFF × Schema ON/OFF.
- Five repetitions per configuration.
- Results:
  - Native Ollama: **20/20 valid JSON responses**.
  - LangChain: **20/20 valid JSON responses**.
  - Total: **40/40 valid responses**.
  - All responses returned `rating = 2`.
  - No empty `{}` responses.
  - Latency was comparable between interfaces.

### Root Cause Analysis
- LangChain did not introduce invalid structured outputs under the tested conditions.
- The failure could not be reproduced outside the RAGAS evaluation pipeline.
- The exact cause of the original NaN failures remains unidentified.

### Impact
- **H10 — FAIL / Not Supported:** No evidence that LangChain causes malformed JSON responses.
- Native Ollama and LangChain behaved consistently in the experiment.

### Next Steps
- Investigate RAGAS structured output handling, Instructor integration, and response parsing.
- Reproduce the original failure within the RAGAS evaluation pipeline.


## EL-021 — E2-H11: OpenAI-Compatible API Investigation

### Date
2026-11-13 16:33

### Context / Findings
- Investigated whether Ollama's OpenAI-compatible API causes malformed structured outputs.
- Tested direct API calls outside the RAGAS evaluation pipeline.
- The original empty `{}` response and resulting `NaN` could not be reproduced.

### Root Cause Analysis
- No evidence that the OpenAI-compatible API independently caused the observed failures.
- Further investigation of Instructor integration was required.

### Impact
- **H11 — FAIL / Not Supported:** The original hypothesis was not confirmed.

### Next Steps
- Investigate Instructor's structured output handling and retry behavior.

---

## EL-022 — E2-H12: Instructor Integration Investigation

### Date
2026-11-13 18:15
 
### Context / Findings
- Investigated whether Instructor integration caused malformed structured outputs.
- Compared JSON Object and JSON Schema response formats.
- Identified that `Mode.JSON` uses `response_format={"type":"json_object"}`, while `Mode.JSON_SCHEMA` uses schema-constrained output.
- The original failure was not reproduced during the initial H12 tests.

### Root Cause Analysis
- JSON Object mode does not enforce the required output fields.
- This allows responses such as `{}` that cannot be parsed into the expected rating schema.
- JSON Schema mode provides stronger output constraints.

### Impact
- **H12 — FAIL / Not Supported:** The original hypothesis was not confirmed by the initial tests.
- The investigation identified a promising configuration change for H13.

### Next Steps
- Reproduce an original failing request and validate the JSON Schema fix.

---

## EL-023 — E2-H13: JSON Schema Fix Validation

### Date
2026-11-16 18:50

### Context / Findings
- Reproduced the original malformed structured output using a previously failing request.
- Identified `Mode.JSON` as the configuration associated with empty `{}` responses.
- Upgraded to RAGAS 0.4.4, which supports `instructor.Mode.JSON_SCHEMA` through `llm_factory()`.
- Re-ran the full 75-query `ResponseGroundedness` evaluation.
- Results:
  - Baseline: **20/75 `NaN`**.
  - Final: **0/75 `NaN`**.
  - All 75 evaluations returned numerical scores.

### Root Cause Analysis
- `Mode.JSON` permitted empty JSON objects that did not satisfy the expected structured output schema.
- Instructor retries could not consistently recover from these failures.
- Switching to `Mode.JSON_SCHEMA` resolved the observed malformed-output failures.

### Impact
- **H13 — PASS / Confirmed:** JSON Schema eliminated the observed `ResponseGroundedness` `NaN` failures in the full regression run.
- Evaluation completion improved from **55/75 to 75/75**.
- The fix required no custom RAGAS monkeypatching.

### Decision
- Adopt RAGAS 0.4.4 with `instructor.Mode.JSON_SCHEMA` for subsequent experiments.

### Next Steps
- Use the validated structured output configuration in E3 and E4.
- Investigate remaining metric reliability issues separately from structured output failures.

---

## EL-024 — E2-H14: AnswerAccuracy Error Suppression and Reasoning Loop Mitigation

### Date
2026-11-17 14:50

### Context / Findings
- Investigated why `AnswerAccuracy` returned `SUCCESS` with score `0.0` despite repeated LLM failures.
- LangSmith trace for `neg024` showed:
  - Judge 1 returned `rating=0`.
  - Judge 2 entered a reasoning loop and exhausted `max_tokens=4096` across five attempts.
  - Final result: `score=0.0`, `status=SUCCESS`.
- Inspected RAGAS 0.4.4 implementations of `AnswerAccuracy` and `Faithfulness`.
- Tested selected failure cases with `JSON_SCHEMA` and `repetition_penalty=1.1`.
- All six selected `AnswerAccuracy` cases returned numerical scores without `NaN`.
- For `neg024`, LLM calls decreased from 6 to 2, and latency from approximately 389s to 10s.
- Overall metric scores showed improvement, but exact score changes were not documented as part of this experiment. Metric quality and score reliability will be evaluated in E4.

### Root Cause Analysis
- `AnswerAccuracy` catches LLM exceptions and returns `NaN` after exhausting retries.
- If one judge fails, the metric silently uses the successful judge's score.
- If both judges fail, the metric returns `NaN` instead of propagating the original exception.
- This can produce misleading numerical scores and hide actual evaluation failures.
- `Faithfulness` does not catch these exceptions within its evaluation pipeline, allowing errors such as `IncompleteOutputException` to propagate.

### Impact
- **H14 — PASS / Confirmed:** `AnswerAccuracy` suppresses LLM errors and can report incomplete evaluations as successful.
- `repetition_penalty=1.1` eliminated observed reasoning loops in targeted `Faithfulness` and `AnswerAccuracy` tests.
- Selected regression cases demonstrated improved generation stability and substantial latency reductions in some cases.
- The underlying RAGAS error-handling limitation remains unresolved.

### Decision
- Adopt RAGAS 0.4.4 with `JSON_SCHEMA` and `repetition_penalty=1.1` as the baseline configuration for E3 and E4.
- No full-dataset regression was performed for the penalty change; conclusions are limited to tested cases.

### Next Steps
- Document the H14 findings and baseline configuration in the evaluation strategy and decision log.
- Investigate metric score reliability and evaluation quality in E4.
- Address `AnswerAccuracy` reference-answer validity separately in E5.