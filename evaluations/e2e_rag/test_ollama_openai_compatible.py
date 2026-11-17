"""E2-H11: Test Ollama OpenAI-compatible structured output.

Compare structured-output reliability across thinking modes and schema
enforcement using the Ollama OpenAI-compatible API.

Test four configurations (thinking none/high, schema ON/OFF), with five
repeated requests per configuration.

Record whether the response is valid, the parsed rating, latency, errors,
and the raw API response for each run.
"""

import json
import time
from urllib.request import Request, urlopen

import pytest

from config import paths
from evaluations import helper


@pytest.mark.integration
@pytest.mark.parametrize("think", ["none", "high"])
@pytest.mark.parametrize("use_schema", [True, False])
def test_e2_h11_openai_compatible(think, use_schema):
    """Record structured-output validity, rating, latency, and errors.

    Invalid responses and API errors are recorded rather than asserted
    because this is a diagnostic experiment, not a pass/fail validation test.
    """
    # Load the shared E2-H9 prompt and model configuration.
    config = helper.load_yaml(paths.E2_H9_PROMPT)

    # Create the output directory for E2-H11 results.
    results_dir = paths.E2_H11_RESULTS
    results_dir.mkdir(parents=True, exist_ok=True)

    # Convert the native Ollama /api/chat URL to the OpenAI-compatible endpoint.
    base_url = config["url"].split("/api/chat")[0]
    endpoint = f"{base_url}/v1/chat/completions"

    # Repeat each configuration five times to check consistency.
    for attempt in range(1, 6):
        # Build the common request payload for the OpenAI-compatible API.
        payload = {
            "model": config["model"],
            "messages": [
                {"role": "system", "content": config["system"]},
                {"role": "user", "content": config["user"]},
            ],
            "temperature": 0,
            "max_tokens": 4096,
            "stream": False,
            "reasoning_effort": think,
        }

        # Add JSON schema enforcement for schema=True configurations.
        if use_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "groundedness_rating",
                    "strict": True,
                    "schema": config["schema"],
                },
            }

        # Build the HTTP POST request without sending it yet.
        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        # Initialize diagnostic fields before making the API request.
        start = time.perf_counter()
        data = None
        error = None
        rating = None
        valid = False

        # Send the request, parse the response, and validate the rating.
        try:
            with urlopen(request, timeout=240) as response:
                data = json.load(response)

            # Extract the model-generated content from the completion response.
            content = data["choices"][0]["message"].get("content") or ""

            # Parse the structured JSON returned by the model.
            parsed = json.loads(content)

            # Extract and validate the groundedness rating.
            rating = parsed.get("rating")
            valid = type(rating) is int and rating in (0, 1, 2)

        # Record HTTP, JSON parsing, or response-structure errors.
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        # Calculate total request latency regardless of success or failure.
        latency = round(time.perf_counter() - start, 2)

        # Store configuration, validation result, latency, error, and response.
        result = {
            "interface": "openai_compatible",
            "think": think,
            "schema": use_schema,
            "attempt": attempt,
            "valid": valid,
            "rating": rating,
            "latency_s": latency,
            "error": error,
            "raw_response": data,
        }

        # Build a descriptive filename containing the experiment configuration.
        filename = (
            f"openai_think_{think}_"
            f"schema_{use_schema}_run_{attempt}.json"
        )

        # Save the complete diagnostic result as a JSON file.
        (results_dir / filename).write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Print a compact result summary for the pytest console output.
        print(f"{filename}: valid={valid}, rating={rating}, error={error}")