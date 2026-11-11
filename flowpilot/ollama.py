"""Loopback-only Ollama transport and model readiness checks."""
import json
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

DEFAULT_MODEL = 'llama3.2:3b'
BASE_URL = 'http://127.0.0.1:11434'


class OllamaError(Exception):
    """A provider failure with a safe message for the dashboard."""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _json_request(path, timeout, payload=None):
    body = json.dumps(payload).encode('utf-8') if payload is not None else None
    request = Request(BASE_URL + path, data=body, headers={'Content-Type': 'application/json'})
    try:
        with build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        raise OllamaError(f'Ollama returned HTTP {exc.code}.') from exc
    except (URLError, TimeoutError, ConnectionError) as exc:
        raise OllamaError('Ollama was unreachable or timed out.') from exc


def installed_model(model):
    """Return installed model metadata and reject cloud-backed entries."""
    name = model if ':' in model.rsplit('/', 1)[-1] else f'{model}:latest'
    catalog = _json_request('/api/tags', timeout=2)
    if not isinstance(catalog, dict) or not isinstance(catalog.get('models'), list):
        raise OllamaError('Ollama returned an invalid model list.')
    for entry in catalog['models']:
        if isinstance(entry, dict) and entry.get('name') == name:
            if entry.get('remote_host') or entry.get('remote_model') or name.endswith('-cloud') or name.endswith(':cloud'):
                raise OllamaError('Select an installed local model.')
            return entry
    raise OllamaError(f'Model {model} is missing. Pull it in Ollama first.')


def chat(model, schema, instructions, inputs):
    installed_model(model)
    response = _json_request('/api/chat', timeout=90, payload={
        'model': model,
        'stream': False,
        'format': schema,
        'options': {'temperature': 0, 'seed': 42, 'num_ctx': 8192, 'num_predict': 1024},
        'messages': [
            {'role': 'system', 'content': instructions},
            {'role': 'user', 'content': json.dumps(inputs)},
        ],
    })
    if response.get('done') is not True or response.get('done_reason') == 'length':
        raise OllamaError('Ollama did not complete its response.')
    return json.loads(response['message']['content'])
