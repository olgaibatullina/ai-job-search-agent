"""E2-H9: Investigate Ollama structured output reliability.

Check whether thinking mode and JSON schema enforcement affect
the ability of qwen3:8b to return a valid groundedness rating.

Test four configurations (thinking ON/OFF schema ON/OFF),
with five repeated requests per configuration.

Call the native Ollama API directly to isolate model behavior
from RAGAS and OpenAI-compatible API integration.
"""

import json
import time
from urllib.request import Request, urlopen

import pytest

from evaluations import helper
from config import paths


@pytest.mark.integration
@pytest.mark.parametrize('think', [True, False])
@pytest.mark.parametrize('use_schema', [True, False])
def test_e2_h9_ollama(think, use_schema):
    """Record JSON validity, rating, latency, and raw response for each run.

    Invalid responses are recorded rather than asserted because
    this is a diagnostic experiment, not a pass/fail validation test.
    """
    config = helper.load_yaml(paths.E2_H9_PROMPT)

    paths.E2_H9_RESULTS.mkdir(parents=True, exist_ok=True)

    # Repeat each configuration five times to check consistency.
    for attempt in range(1, 6):
        # Prepare the request body for the Ollama API.
        payload = {
            'model': config['model'],
            'messages': [
                {'role': 'system', 'content': config['system']},
                {'role': 'user', 'content': config['user']},
            ],
            'think': think,
            'stream': False,
            'options': {'temperature': 0, 'num_predict': 4096},
        }

        # Enable JSON schema enforcement only for schema=True configurations.
        if use_schema:
            payload['format'] = config['schema']

        # Build an HTTP POST request to the native Ollama API.
        # Request() prepares the request but does not send it yet.
        request = Request(
            config['url'],
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        
        # Send the request, run model inference, and measure response time.
        start = time.perf_counter()
        with urlopen(request, timeout=240) as response:
            data = json.load(response)

        latency_s = round(time.perf_counter() - start, 2)

        # Extract the model-generated text from the API response.
        content = data.get('message', {}).get('content', '')
        
        # Check whether the model returned valid JSON with rating 0, 1, or 2.
        try:
            rating = json.loads(content).get('rating')
            valid = type(rating) is int and rating in (0, 1, 2)
        except (ValueError, TypeError, AttributeError):
            rating = None
            valid = False

        # Store test configuration, validation result, latency, and full API response.
        result = {
            'think': think,
            'schema': use_schema,
            'attempt': attempt,
            'valid': valid,
            'rating': rating,
            'latency_s': latency_s,
            'raw_response': data,
        }

        # Console output.        
        name = f'think_{think}_schema_{use_schema}_run_{attempt}.json'
        (paths.E2_H9_RESULTS / name).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        
        print(f'{name}: valid={valid}, rating={rating}')
