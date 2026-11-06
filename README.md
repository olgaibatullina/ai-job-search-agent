# AI Job Search & Validation Framework

## Overview

A local-first AI application built as a portfolio project for AI/ML validation and LLM evaluation.

The project combines a RAG-based job search application using a fixed local job corpu with a separate validation framework for testing:

- **RAG evaluation** — retrieval/generation quality, Answer Accuracy, Answer Relevancy, Faithfulness and Response Groundedness
- **LLM evaluation** — LLM-as-a-Judge reliability and evaluation failures
- **Evaluation frameworks** — RAGAS, DeepEval and Promptfoo
- **Adversarial testing** — negative cases, hallucinations, prompt injection and unsupported claims
- **Agent validation** — tool calling, tool selection and task completion
- **Backend validation** — FastAPI, API and SQL testing

## Experimentation & Documentation

The project uses controlled evaluation experiments to investigate model and system behavior. Models, prompts, evaluation configurations, and other relevant parameters can be changed between experiments while keeping the evaluation setup as consistent as possible.

Each experiment is documented with its hypothesis, configuration, measurements, results, limitations, and conclusions.

Evaluation results are stored in:
- Machine-readable CSV results: [evaluation_results/experiments/](https://github.com/olgaibatullina/ai-job-search-agent/tree/main/evaluation_results/experiments)
- HTML experiment reports: [reports/](https://github.com/olgaibatullina/ai-job-search-agent/tree/main/reports)

Engineering decisions and implementation changes are tracked separately in:
- Implementation changes, findings, validation results, and technical issues: [engineering log](https://github.com/olgaibatullina/ai-job-search-agent/tree/main/project_documentation/engineering_log.md)
- Architectural and evaluation decisions, rationale, and their status: [decision log](https://github.com/olgaibatullina/ai-job-search-agent/tree/main/project_documentation/decision_log.md)

This provides a traceable record of how the validation framework evolves and why particular approaches were chosen.

## How to Run
### 1. Install dependencies

Create and activate a Python virtual environment, then install the project dependencies:
```
pip install -r requirements.txt
```

### 2. Install Ollama and download the model

The project uses Ollama to run Qwen3 8B locally for RAG generation and LLM-as-a-Judge evaluation.

Install Ollama, then download the model:
- [Install Ollama](https://ollama.com/download)
- Pull the required model in the terminal:

  ```
  ollama pull qwen3:8b
  ```

Make sure Ollama is running before starting the evaluation tests.

### 3. Run the evaluation suite

The evaluation framework is built with pytest.

Run the full test suite:
```
python -m pytest -s
```
Or run individual evaluation areas, for example:
```
python -m pytest evaluations/e2e_rag/test_ragas_generation_metrics_llm.py::test_ragas_metric_eval -s
```

### 4. LangSmith tracing

LLM evaluation calls can be traced with LangSmith.

Configure the required LangSmith environment variables in your local environment, including the API key, then run the evaluation tests normally.

The RAGAS LLM-as-a-Judge calls are instrumented with LangSmith tracing, allowing individual evaluation cases to be investigated for:
- Token usage
- Latency
- Failures
- Incomplete outputs
- LLM execution details

### 5. Evaluation results

Evaluation runs produce machine-readable CSV results and experiment reports.

The results can be used together with LangSmith traces to investigate individual evaluation cases and analyze metric behavior.


## Current Status
The project is developed through focused evaluation experiments rather than optimizing a single overall score.

- [x] The core RAG and evaluation pipeline is implemented
- [x] E1 — LLM Judge failure reduction
- [ ] E2 — RAG evaluation NaN investigation
- [ ] E3 — Retrieval metrics disagreement
- [ ] Independent evaluation frameworks (DeepEval and Promptfoo)
- [ ] Adversarial & safety validation
- [ ] Agent / tool-call validation
- [ ] Backend/API validation
- [ ] E4 — Judge accuracy & consistency

