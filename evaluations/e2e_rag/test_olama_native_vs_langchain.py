"""E2-H10: Compare native Ollama API and LangChain responses.

Investigate whether LangChain introduces invalid JSON responses
that could explain RAGAS ResponseGroundedness NaN results.

Compare both interfaces using identical prompts, model settings,
thinking modes, and JSON schema configurations.

Test four configurations (thinking ON/OFF × schema ON/OFF),
with five repeated requests per configuration.

Record JSON validity, rating, latency, and raw responses.
This is a diagnostic experiment, not a pass/fail validation test.
"""

import json
import time
from urllib.request import Request, urlopen

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import paths
from evaluations import helper


@pytest.mark.integration
@pytest.mark.parametrize("think", ["none", "high"])
@pytest.mark.parametrize("use_schema", [True, False])
def test_e2_h10_langchain(think, use_schema):
    """Compare structured JSON responses from native Ollama and LangChain.

    Evaluate response validity, rating extraction, and latency with
    reasoning and JSON Schema enabled or disabled. Save results for
    each interface and configuration to support H10 investigation."""
    
    # Load shared prompt configuration and prepare the results directory.
    config = helper.load_yaml(paths.E2_H9_PROMPT)
    results_dir = paths.E2_H10_RESULTS
    results_dir.mkdir(parents=True, exist_ok=True)

    # Prepare equivalent messages for the native Ollama API.
    messages = [
        {"role": "system", "content": config["system"]},
        {"role": "user", "content": config["user"]},
    ]

    # Convert the same prompts into LangChain message objects.
    lc_messages = [
        SystemMessage(content=config["system"]),
        HumanMessage(content=config["user"]),
    ]

    # Configure LangChain with matching model and generation parameters.
    llm = ChatOllama(
        model=config["model"],
        temperature=0,
        num_predict=4096,
        reasoning=think,
        format=config["schema"] if use_schema else None,
    )

    # Run both interfaces under the same experimental configuration.
    for attempt in range(1, 6):

        # Build the native Ollama API request.
        payload = {
            "model": config["model"],
            "messages": messages,
            "reasoning_effort": think,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 4096,
            },
        }

        # Apply JSON Schema only when enabled for this test case.
        if use_schema:
            payload["format"] = config["schema"]

        request = Request(
            config["url"],
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        # Execute the native Ollama request and measure latency.
        start = time.perf_counter()
        with urlopen(request, timeout=240) as response:
            native_data = json.load(response)
        native_latency = round(time.perf_counter() - start, 2)

        # Extract generated content from the native response.
        native_content = native_data.get("message", {}).get("content", "")

        # Execute the equivalent LangChain request and measure latency.
        start = time.perf_counter()
        lc_response = llm.invoke(lc_messages)
        lc_latency = round(time.perf_counter() - start, 2)

         # Extract generated content from the LangChain response.
        lc_content = lc_response.content

        # Collect validation results from both interfaces.
        results = []

        for interface, content, latency, raw_response in [
            (
                "native",
                native_content,
                native_latency,
                native_data,
            ),
            (
                "langchain",
                lc_content,
                lc_latency,
                {
                    "content": lc_response.content,
                    "additional_kwargs": lc_response.additional_kwargs,
                    "response_metadata": lc_response.response_metadata,
                },
            ),
        ]:
            # Parse JSON and validate the expected rating field.
            try:
                rating = json.loads(content).get("rating")
                valid = type(rating) is int and rating in (0, 1, 2)
            except (ValueError, TypeError, AttributeError):
                rating = None
                valid = False

            # Record configuration, validation outcome, and raw response.
            result = {
                "interface": interface,
                "think": think,
                "schema": use_schema,
                "attempt": attempt,
                "valid": valid,
                "rating": rating,
                "latency_s": latency,
                "raw_response": raw_response,
            }

            # Generate a unique filename for each interface and configuration.
            filename = (
                f"{interface}_think_{think}_"
                f"schema_{use_schema}_run_{attempt}.json"
            )

            # Persist results for reproducibility and comparison.
            (results_dir / filename).write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # Store the result for potential comparison within this run.
            results.append(result)

            print(
                f"{filename}: "
                f"valid={valid}, rating={rating}, latency={latency}s"
            )

        # Console output.
        native_result, lc_result = results

        print(
            f"Comparison run {attempt}: "
            f"valid_match={native_result['valid'] == lc_result['valid']}, "
            f"rating_match={native_result['rating'] == lc_result['rating']}"
        )