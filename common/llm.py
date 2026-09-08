"""Local LLM transport. No hidden retries, JSON repair, or result fabrication."""
import json
import os
import time
from datetime import datetime, timezone
import requests

MODEL = os.getenv('AOT_MODEL', 'mistralai/ministral-3-14b-reasoning')
BASE_URL = os.getenv('AOT_LLM_URL', 'http://localhost:1234/v1').rstrip('/')

def parse_json(content):
    return json.loads(content)

def complete(messages, schema=None, max_tokens=1024):
    payload = dict(model=MODEL, messages=messages, temperature=0, seed=42,
                   max_tokens=max_tokens, stream=False)
    if schema is not None:
        if hasattr(schema, 'model_json_schema'):
            schema = schema.model_json_schema()
        payload['response_format'] = {'type': 'json_schema', 'json_schema': {
            'name': 'result', 'strict': True, 'schema': schema}}
    started = datetime.now(timezone.utc).isoformat()
    t = time.perf_counter()
    response = requests.post(BASE_URL+'/chat/completions', json=payload, timeout=600)
    response.raise_for_status()
    raw = response.json()
    elapsed = time.perf_counter()-t
    message = raw['choices'][0]['message']
    # Retain observable output and usage; private reasoning text is not needed.
    return {'content': message.get('content') or '', 'usage': raw.get('usage', {}),
            'elapsed_seconds': elapsed, 'model': raw.get('model', MODEL),
            'finish_reason': raw['choices'][0].get('finish_reason'),
            'started_at': started, 'request_id': raw.get('id'),
            'parameters': {'temperature': 0, 'seed': 42, 'max_tokens': max_tokens,
                           'structured_output': schema is not None}}
