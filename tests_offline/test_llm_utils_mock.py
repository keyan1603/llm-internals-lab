# tests_offline/test_llm_utils_mock.py
# Verifies config construction for generate_with_backoff and
# embed_with_backoff against a fake client that records what it was
# called with, and verifies attempt_logprobs_confidence's real
# behavior: catch a 400 and return a structured "not available" result,
# re-raise anything else. The 400 shape mirrors the real one confirmed
# against the live API (see this module's header comment and this
# post's Pitfalls section), constructed via the real
# google.genai.errors.ClientError constructor, not a plain Exception.

from types import SimpleNamespace
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from common.llm_utils import generate_with_backoff, embed_with_backoff, count_tokens_with_backoff, attempt_logprobs_confidence, attempt_multi_candidate


class _FakeModels:
    def __init__(self, raise_error=None):
        self.last_generate_config = None
        self.last_embed_config = None
        self.count_tokens_calls = 0
        self._raise_error = raise_error

    def generate_content(self, model, contents, config=None):
        if self._raise_error is not None:
            raise self._raise_error
        self.last_generate_config = config
        return SimpleNamespace(text="mock response")

    def embed_content(self, model, contents, config=None):
        self.last_embed_config = config
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2, 0.3])])

    def count_tokens(self, model, contents):
        self.count_tokens_calls += 1
        return SimpleNamespace(total_tokens=42)


def _fake_client(raise_error=None):
    return SimpleNamespace(models=_FakeModels(raise_error))


def test_generate_with_backoff_config():
    client = _fake_client()
    generate_with_backoff(client, "mock-model", "hello", temperature=0.5, candidate_count=3)
    config = client.models.last_generate_config
    assert isinstance(config, genai_types.GenerateContentConfig)
    assert config.temperature == 0.5 and config.candidate_count == 3
    print("  [generate_with_backoff builds a typed config with temperature and candidate_count] OK")

    client2 = _fake_client()
    generate_with_backoff(client2, "mock-model", "hello")
    assert client2.models.last_generate_config is None, "expected no config when no optional args are passed"
    print("  [generate_with_backoff skips config entirely when nothing is passed] OK")


def test_embed_with_backoff_config():
    client = _fake_client()
    embed_with_backoff(client, "mock-embed-model", "some text", task_type="RETRIEVAL_QUERY")
    config = client.models.last_embed_config
    assert isinstance(config, genai_types.EmbedContentConfig)
    assert config.task_type == "RETRIEVAL_QUERY"
    print("  [embed_with_backoff builds a config with task_type] OK")


def test_count_tokens_with_backoff():
    client = _fake_client()
    result = count_tokens_with_backoff(client, "mock-model", "some text")
    assert result.total_tokens == 42
    print("  [count_tokens_with_backoff passes through the real response] OK")


def test_attempt_logprobs_confidence_real_failure_shape():
    real_shaped_error = genai_errors.ClientError(
        400, {"error": {"code": 400, "message": "Logprobs is not enabled for this model", "status": "INVALID_ARGUMENT"}}
    )
    client = _fake_client(raise_error=real_shaped_error)
    result = attempt_logprobs_confidence(client, "mock-model", "some prompt")
    assert result["available"] is False
    assert "Logprobs is not enabled" in result["error"]
    print("  [attempt_logprobs_confidence catches the real 400 shape and returns a structured result] OK")

    other_error = genai_errors.ClientError(500, {"error": {"code": 500, "message": "server error", "status": "INTERNAL"}})
    client2 = _fake_client(raise_error=other_error)
    try:
        attempt_logprobs_confidence(client2, "mock-model", "some prompt")
        assert False, "expected a non-400 ClientError to be re-raised, not swallowed"
    except genai_errors.ClientError:
        print("  [attempt_logprobs_confidence re-raises non-400 errors] OK")


def test_attempt_multi_candidate_real_failure_shape():
    real_shaped_error = genai_errors.ClientError(
        400, {"error": {"code": 400, "message": "Multiple candidates is not enabled for this model", "status": "INVALID_ARGUMENT"}}
    )
    client = _fake_client(raise_error=real_shaped_error)
    result = attempt_multi_candidate(client, "mock-model", "some prompt")
    assert result["available"] is False
    assert "Multiple candidates is not enabled" in result["error"]
    print("  [attempt_multi_candidate catches the real 400 shape and returns a structured result] OK")


def main():
    test_generate_with_backoff_config()
    test_embed_with_backoff_config()
    test_count_tokens_with_backoff()
    test_attempt_logprobs_confidence_real_failure_shape()
    test_attempt_multi_candidate_real_failure_shape()
    print("llm_utils offline validation: all checks passed")


if __name__ == "__main__":
    main()
