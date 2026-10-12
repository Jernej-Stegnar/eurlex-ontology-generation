"""Tests for the pluggable LLM backends against a local mock HTTP server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from pydantic import BaseModel

from eurlex_ontology_generation.utils import llm_client
from eurlex_ontology_generation.utils.llm_client import embed_texts, inline_refs, llm_chat


class Answer(BaseModel):
    label: str
    score: int


class _Handler(BaseHTTPRequestHandler):
    calls: list = []
    fail_next: list = []          # status codes to return before succeeding
    reject_json_schema = False

    def log_message(self, *args):  # silence
        pass

    def _send(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.calls.append({"path": self.path, "body": body, "headers": dict(self.headers)})
        if self.fail_next:
            return self._send(self.fail_next.pop(0), {"error": "try again"})
        if self.path == "/v1/chat/completions":
            rf = body.get("response_format", {})
            if rf.get("type") == "json_schema" and self.reject_json_schema:
                return self._send(400, {"error": "response_format json_schema not supported"})
            return self._send(200, {"choices": [{"message": {"content": '{"label": "ok", "score": 3}'},
                                                 "finish_reason": "stop"}],
                                    "usage": {"prompt_tokens": 11, "completion_tokens": 5}})
        if self.path == "/v1/embeddings":
            data = [{"index": i, "embedding": [float(i), 1.0]} for i, _ in enumerate(body["input"])]
            return self._send(200, {"data": list(reversed(data))})
        if self.path == "/v1/messages":
            return self._send(200, {"content": [{"type": "tool_use", "name": "emit_output",
                                                 "input": {"label": "ok", "score": 4}}],
                                    "stop_reason": "tool_use", "usage": {"input_tokens": 7, "output_tokens": 2}})
        return self._send(404, {})


@pytest.fixture()
def server(monkeypatch):
    _Handler.calls, _Handler.fail_next, _Handler.reject_json_schema = [], [], False
    llm_client._FORMAT_DOWNGRADE.clear()
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_openai_compatible_structured_output(server, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    backend = {"provider": "openai_compatible", "base_url": f"{server}/v1", "api_key_env": "TEST_KEY"}
    r = llm_chat(backend, "some-model", "sys", "user", 0.1, schema_model=Answer, seed=7, max_tokens=50)
    assert Answer.model_validate_json(r.content).score == 3
    assert (r.prompt_tokens, r.output_tokens, r.done_reason) == (11, 5, "stop")
    call = _Handler.calls[0]
    assert {k.lower(): v for k, v in call["headers"].items()}["authorization"] == "Bearer sk-test"
    assert call["body"]["response_format"]["type"] == "json_schema"
    assert "$ref" not in json.dumps(call["body"]["response_format"])
    assert call["body"]["seed"] == 7 and call["body"]["max_tokens"] == 50


def test_openai_compatible_falls_back_when_schema_rejected(server):
    _Handler.reject_json_schema = True
    backend = {"provider": "openai_compatible", "base_url": f"{server}/v1"}
    llm_chat(backend, "m", "s", "u", 0.0, schema_model=Answer)
    assert [c["body"].get("response_format", {}).get("type") for c in _Handler.calls] == ["json_schema", "json_object"]
    llm_chat(backend, "m", "s", "u", 0.0, schema_model=Answer)        # remembered: no second rejection
    assert _Handler.calls[-1]["body"]["response_format"]["type"] == "json_object"


def test_retries_on_rate_limit(server):
    _Handler.fail_next = [429, 503]
    backend = {"provider": "openai_compatible", "base_url": f"{server}/v1", "max_retries": 3}
    r = llm_chat(backend, "m", "s", "u", 0.0, json_mode=True)
    assert r.content and len(_Handler.calls) == 3


def test_no_retry_on_client_error(server):
    _Handler.fail_next = [401]
    backend = {"provider": "openai_compatible", "base_url": f"{server}/v1"}
    with pytest.raises(Exception, match="401"):
        llm_chat(backend, "m", "s", "u", 0.0)
    assert len(_Handler.calls) == 1


def test_anthropic_forced_tool_call(server, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_TEST_KEY", "ak-test")
    backend = {"provider": "anthropic", "base_url": f"{server}/v1/messages", "api_key_env": "ANTHROPIC_TEST_KEY"}
    r = llm_chat(backend, "a-model", "sys", "user", 0.0, schema_model=Answer)
    assert Answer.model_validate_json(r.content).score == 4
    body = _Handler.calls[0]["body"]
    assert body["tool_choice"] == {"type": "tool", "name": "emit_output"}
    assert body["system"] == "sys" and body["max_tokens"] == 4096
    headers = {k.lower(): v for k, v in _Handler.calls[0]["headers"].items()}
    assert headers["x-api-key"] == "ak-test" and headers["anthropic-version"] == "2023-06-01"


def test_missing_key_is_a_clear_error():
    backend = {"provider": "openai_compatible", "base_url": "http://127.0.0.1:9/v1", "api_key_env": "DEFINITELY_UNSET_VAR"}
    with pytest.raises(RuntimeError, match="DEFINITELY_UNSET_VAR"):
        llm_chat(backend, "m", "s", "u", 0.0)


def test_embeddings_openai_compatible_keeps_order(server):
    vecs = embed_texts({"provider": "openai_compatible", "base_url": f"{server}/v1"}, "emb", ["a", "b", "c"])
    assert [v[0] for v in vecs] == [0.0, 1.0, 2.0]


def test_ollama_path_unchanged(monkeypatch):
    import ollama

    seen = {}

    def fake_chat(model, messages, options, format):
        seen.update(model=model, options=options, format=format)
        return {"message": {"content": "{}"}, "done_reason": "stop", "prompt_eval_count": 1, "eval_count": 1}

    monkeypatch.setattr(ollama, "chat", fake_chat)
    r = llm_chat({"provider": "ollama"}, "qwen2.5:3b", "s", "u", 0.1, json_mode=True, num_ctx=4096, seed=1)
    assert r.done_reason == "stop" and seen["format"] == "json"
    assert seen["options"] == {"temperature": 0.1, "seed": 1, "num_ctx": 4096}
    llm_chat({}, "m", "s", "u", 0.0, schema_model=Answer)                 # default provider is ollama
    assert seen["format"]["properties"]["label"]["type"] == "string"


def test_unknown_provider():
    with pytest.raises(ValueError, match="Unknown"):
        llm_chat({"provider": "nope"}, "m", "s", "u", 0.0)


def test_inline_refs_resolves_nested_models():
    from eurlex_ontology_generation.utils.ontology_schema import build_ontology_model

    schema = inline_refs(build_ontology_model().model_json_schema())
    text = json.dumps(schema)
    assert "$ref" not in text and "$defs" not in text and "EUInstitution" in text
