# Dataset Specification

## 1. Purpose

This document defines the dataset used to develop and validate the AI Job Search Agent.

The dataset is designed to support evaluation of:

- job retrieval and ranking;
- candidate-to-job matching;
- LLM-generated match analysis;
- groundedness and hallucination behavior;
- regression testing as the system evolves.

The dataset specification is separate from the job data itself and from the matching rules. `raw_jobs.json` contains source job postings, while `job_matching_config.yaml` defines matching behavior and weighting.

## 2. V1 Dataset Scope

The V1 raw corpus contains **40 real-world job postings** collected manually from job listings relevant to the target search space.

The corpus intentionally contains both suitable and unsuitable jobs. A dataset containing only good matches would not adequately test retrieval or matching behavior.

### V1 composition

| Category | Jobs | Share |
| --- | ---: | ---: |
| Relevant | 13 | 32.5% |
| Hard negative | 12 | 30.0% |
| Clear negative | 15 | 37.5% |
| **Total** | **40** | **100%** |

Forty jobs are considered sufficient for the first portfolio-scale version. The corpus may be expanded later if evaluation results show that additional examples are needed.

## 3. Dataset Categories

### Relevant

A job whose primary professional function is aligned with the candidate's target direction, such as AI/ML quality engineering, AI/ML validation, LLM evaluation, AI testing, GenAI quality, or a closely related role.

A Relevant job does not need to be a perfect candidate match. Individual skill or experience gaps may still exist.

### Hard negative

A job whose primary function is sufficiently close to the target role that it may initially appear suitable, but whose detailed requirements reveal substantial mismatch.

Examples can include roles with relevant QA, validation, evaluation, or AI terminology that require significant experience in another core area such as backend/distributed systems, cybersecurity, robotics, or another specialized domain.

Hard negatives are particularly useful for testing whether the system evaluates the complete job description instead of relying primarily on titles and keywords.

### Clear negative

A job whose primary professional function is outside the target role family even when the posting contains overlapping terms such as AI, ML, evaluation, quality, validation, benchmarks, or testing.

Examples include ML engineering, AI research, research engineering, data science, MLOps, AI infrastructure, hardware/ASIC validation, and similar roles where the core occupation is different from AI/ML QA or validation.

Clear negatives test whether semantic or keyword overlap causes false-positive matches.

## 4. Raw Job Data

Raw job postings are stored separately from evaluation annotations.

The raw dataset follows the canonical schema defined by `raw_job_template.json` and contains source-visible information such as:

- source and source job ID;
- job title and company;
- location and workplace type;
- employment type;
- complete job description;
- posting-date information when available;
- compensation when explicitly provided;
- collection metadata.

Raw records must remain close to the source. Missing source fields are stored as `null`; skills, seniority, domains, match labels, or other attributes must not be inferred and inserted into the raw record.

The substantive job description should be preserved as collected, without rewriting its meaning or silently correcting source content.

## 5. Ground Truth and Golden Dataset

Ground truth is created separately from `raw_jobs.json`.

Each annotated example references its raw posting by `job_id`. The raw job therefore does not need to be duplicated inside the golden dataset.

Ground-truth annotation is human reviewed and uses:

- the candidate profile as the source of candidate experience and skills;
- `job_matching_config.yaml` as the source of matching rules;
- the complete job description as the source of job requirements.

The annotation process must not invent candidate experience or treat target-direction skills as confirmed professional experience.

The golden dataset is the reviewed collection of examples with predetermined expected behavior used for evaluation and regression testing.

## 6. Matching Ground Truth

Matching annotations should capture the expected match outcome and the evidence needed to evaluate the system's reasoning.

Relevant evidence categories may include:

- confirmed matches;
- transferable matches;
- target-direction matches;
- minor gaps;
- major gaps.

Exact scoring expectations should only be added when the deterministic scoring behavior is sufficiently defined. Human annotation should not introduce arbitrary numeric scores merely to create a target value.

## 7. Retrieval Ground Truth

The primary retrieval evaluation is performed at the **job level**, even if job descriptions are split into multiple chunks for embedding and FAISS retrieval.

Conceptually:

`raw jobs → chunks → embeddings → FAISS Top-K chunks → aggregate/deduplicate by job_id → Top-K jobs → retrieval evaluation`

Evaluation queries should identify which jobs are relevant to each query. Retrieval metrics can then evaluate whether those jobs are returned at the appropriate ranks.

Chunk-level annotation is not required for the entire corpus. A small diagnostic subset may be added later to evaluate chunking behavior, boilerplate retrieval, chunk size, or overlap problems.

## 8. Dataset Quality Principles

The V1 dataset should preserve the following properties:

- **Realism:** use real job postings from the target search space rather than synthetic ideal matches.
- **Diversity:** include different companies, role families, seniority expectations, domains, and technical requirements.
- **Negative coverage:** include both obvious and difficult negative examples.
- **Source fidelity:** preserve raw source content and keep inferred information out of raw records.
- **Grounded annotation:** derive expected outcomes only from the job description, candidate profile, and documented matching rules.
- **Reproducibility:** use stable `job_id` links between raw records and annotations and apply the same category definitions consistently.
- **Separation of concerns:** keep raw source data, matching configuration, and evaluation ground truth separate.

## 9. V1 Completion Criterion

Raw-data collection for V1 is complete at **40 job postings**.

Further collection is not required merely to reach a round-number target. Additional jobs should be added only when they improve evaluation coverage, address an identified dataset gap, or support a new test scenario.

The next dataset phase is human annotation and construction of the golden evaluation set.
