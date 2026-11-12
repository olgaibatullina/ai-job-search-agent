"""E2-H9: Investigate Ollama structured output reliability.

Check whether thinking mode and JSON schema enforcement affect
the ability of qwen3:8b to return a valid groundedness rating.

Test four configurations (thinking ON/OFF × schema ON/OFF),
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

    for attempt in range(1, 6):
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
        if use_schema:
            payload['format'] = config['schema']

        request = Request(
            config['url'],
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        start = time.perf_counter()
        with urlopen(request, timeout=240) as response:
            data = json.load(response)

        latency_s = round(time.perf_counter() - start, 2)

        content = data.get('message', {}).get('content', '')
        try:
            rating = json.loads(content).get('rating')
            valid = type(rating) is int and rating in (0, 1, 2)
        except (ValueError, TypeError, AttributeError):
            rating = None
            valid = False

        result = {
            'think': think,
            'schema': use_schema,
            'attempt': attempt,
            'valid': valid,
            'rating': rating,
            'latency_s': latency_s,
            'raw_response': data,
        }
        
        name = f'think_{think}_schema_{use_schema}_run_{attempt}.json'
        (paths.E2_H9_RESULTS / name).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        
        print(f'{name}: valid={valid}, rating={rating}')
