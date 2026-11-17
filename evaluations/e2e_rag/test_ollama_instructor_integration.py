"""E2-H12: Test Instructor structured output via Ollama OpenAI-compatible API.

Compare structured-output reliability across thinking modes and schema
enforcement using the Instructor integration.

Test four configurations (thinking none/high, schema ON/OFF), with five
repeated requests per configuration.

Record whether the response is valid, the parsed rating, latency, errors,
and the raw completion response for each run.
"""

import json
import time

import instructor
import pytest
from openai import OpenAI
from pydantic import Field, create_model

from config import paths
from evaluations import helper


@pytest.mark.integration
@pytest.mark.parametrize("think", ["none", "high"])
@pytest.mark.parametrize("use_schema", [True, False])
def test_e2_h12_instructor(think, use_schema):
    """Record structured-output validity, rating, latency, and errors.

    Invalid responses and API errors are recorded rather than asserted
    because this is a diagnostic experiment, not a pass/fail validation test.
    """
    # Load the shared E2-H9 prompt and model configuration.
    config = helper.load_yaml(paths.E2_H9_PROMPT)

    # Create the output directory for E2-H12 results.
    results_dir = paths.E2_H12_RESULTS
    results_dir.mkdir(parents=True, exist_ok=True)

    # Extract the Ollama base URL from the native /api/chat endpoint.
    base_url = config["url"].split("/api/chat")[0]

    # Create an Instructor client using Ollama's OpenAI-compatible API.
    client = instructor.from_openai(
        OpenAI(
            base_url=f"{base_url}/v1",
            api_key="ollama",
            timeout=240,
            max_retries=0,
        ),
        mode=instructor.Mode.JSON,
    )

    # Define the expected structured response and validate the rating range.
    GroundednessRating = create_model(
        "GroundednessRating",
        rating=(int, Field(ge=0, le=2)),
    )

    # Repeat each configuration five times to check consistency.
    for attempt in range(1, 6):
        # Build the common request parameters for the Instructor call.
        kwargs = {
            "model": config["model"],
            "messages": [
                {"role": "system", "content": config["system"]},
                {"role": "user", "content": config["user"]},
            ],
            "response_model": GroundednessRating,
            "temperature": 0,
            "max_tokens": 4096,
            "extra_body": {"reasoning_effort": think},
            "max_retries": 0,
        }

        # Add explicit JSON schema enforcement for schema=True configurations.
        if use_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "groundedness_rating",
                    "strict": True,
                    "schema": config["schema"],
                },
            }

        # Initialize diagnostic fields before making the API request.
        start = time.perf_counter()
        error = None
        rating = None
        valid = False
        data = None

        # Run model inference and capture both the parsed response and completion.
        try:
            response, completion = (
                client.chat.completions.create_with_completion(**kwargs)
            )

            # Extract and validate the structured rating.
            rating = response.rating
            valid = type(rating) is int and rating in (0, 1, 2)

            # Store the raw completion returned by the OpenAI-compatible API.
            data = completion.model_dump(mode="json")

        # Record any Instructor, API, parsing, or validation error.
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        # Calculate total request latency regardless of success or failure.
        latency = round(time.perf_counter() - start, 2)

        # Store configuration, validation result, latency, error, and response.
        result = {
            "interface": "instructor_openai_compatible",
            "think": think,
            "schema": use_schema,
            "attempt": attempt,
            "valid": valid,
            "rating": rating,
            "latency_s": latency,
            "error": error,
            "parsed_response": {"rating": rating} if valid else None,
            "raw_response": data,
        }

        # Build a descriptive filename containing the experiment configuration.
        filename = (
            f"instructor_think_{think}_"
            f"schema_{use_schema}_run_{attempt}.json"
        )

        # Save the complete diagnostic result as a JSON file.
        (results_dir / filename).write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Сonsole output.
        print(f"{filename}: valid={valid}, rating={rating}, error={error}")