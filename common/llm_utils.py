# common/llm_utils.py
# Same retry discipline as every earlier repo: a 429 is a structured
# error the SDK already understands, a dropped connection is a raw
# transport failure that surfaces as an httpx/httpcore exception and
# isn't always absorbed by the SDK's own retry logic.
#
# This repo calls 3 different Gemini surfaces (generate_content,
# embed_content, count_tokens), all 3 get the same throttle and retry
# treatment through separate wrappers, rather than one wrapper trying to
# cover every call shape.
#
# attempt_logprobs_confidence is different on purpose: a 400 "Logprobs
# is not enabled for this model" is not a transport problem to retry
# past, it's the real, confirmed behavior of every currently available
# Gemini Developer-API model (verified directly, not assumed, see this
# post's Pitfalls section). This wrapper treats that specific error as
# an expected, structured result instead of an exception, since the
# whole point of this repo's confidence.py is to show that failure
# honestly rather than crash on it.

import time
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from common.rate_limiter import throttle

MAX_RETRIES = 4


def _retry_call(fn):
    """Shared retry loop: throttle, call fn(), retry on 429 with
    exponential backoff, retry on transport-level exceptions with linear
    backoff, raise anything else immediately."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            throttle()
            return fn()
        except genai_errors.ClientError as e:
            last_error = e
            if e.code == 429:
                wait_seconds = 2 ** attempt * 5
                print(f"  (rate limited, waiting {wait_seconds}s, retry {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait_seconds)
            else:
                raise
        except Exception as e:
            last_error = e
            wait_seconds = 3 * (attempt + 1)
            print(f"  (connection issue: {type(e).__name__}, waiting {wait_seconds}s, retry {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait_seconds)
    raise last_error


def generate_with_backoff(client, model: str, contents, temperature: float = None, candidate_count: int = None):
    """Drop-in replacement for client.models.generate_content(...).
    candidate_count is real but confirmed flaky (googleapis/python-genai#1888):
    the API can return fewer candidates than requested, callers must not
    assume len(response.candidates) == candidate_count."""
    config = None
    if temperature is not None or candidate_count is not None:
        config = genai_types.GenerateContentConfig(temperature=temperature, candidate_count=candidate_count)

    if config is not None:
        return _retry_call(lambda: client.models.generate_content(model=model, contents=contents, config=config))
    return _retry_call(lambda: client.models.generate_content(model=model, contents=contents))


def embed_with_backoff(client, model: str, contents, task_type: str = None, title: str = None):
    """Drop-in replacement for client.models.embed_content(...). task_type
    shapes the embedding differently depending on which side of a
    retrieval pair the text is on (RETRIEVAL_DOCUMENT for the side being
    searched over, RETRIEVAL_QUERY for the side doing the searching),
    confirmed via a real call, see scenario_ticket_triage/categorize.py."""
    config = None
    if task_type is not None or title is not None:
        config = genai_types.EmbedContentConfig(task_type=task_type, title=title)

    if config is not None:
        return _retry_call(lambda: client.models.embed_content(model=model, contents=contents, config=config))
    return _retry_call(lambda: client.models.embed_content(model=model, contents=contents))


def count_tokens_with_backoff(client, model: str, contents):
    """Drop-in replacement for client.models.count_tokens(...). Returns
    only a total count on the Developer API, see this module's header
    comment and scenario_ticket_triage/tokenize_check.py for the real
    boundary this runs into."""
    return _retry_call(lambda: client.models.count_tokens(model=model, contents=contents))


def attempt_logprobs_confidence(client, model: str, contents, logprobs: int = 3) -> dict:
    """Tries to get real per-token log probabilities for a classification
    decision. Returns {"available": True, "response": ...} on success, or
    {"available": False, "error": str} on the specific, confirmed 400
    every current Developer-API model returns for this request, so
    calling code can build a real fallback rather than crash."""
    config = genai_types.GenerateContentConfig(response_logprobs=True, logprobs=logprobs, max_output_tokens=20)
    try:
        response = _retry_call(lambda: client.models.generate_content(model=model, contents=contents, config=config))
        return {"available": True, "response": response}
    except genai_errors.ClientError as e:
        if e.code == 400:
            return {"available": False, "error": str(e)}
        raise


def attempt_multi_candidate(client, model: str, contents, candidate_count: int = 3) -> dict:
    """Tries a single call with candidate_count > 1. Returns
    {"available": True, "response": ...} on success, or
    {"available": False, "error": str} on the confirmed 400
    "Multiple candidates is not enabled for this model" that
    gemini-3.5-flash-lite returns, a second real gap of the same shape
    as the logprobs one, see scenario_ticket_triage/triage_response.py's
    header comment for why self-consistency in this repo degrades to N
    separate calls instead of relying on this."""
    config = genai_types.GenerateContentConfig(candidate_count=candidate_count)
    try:
        response = _retry_call(lambda: client.models.generate_content(model=model, contents=contents, config=config))
        return {"available": True, "response": response}
    except genai_errors.ClientError as e:
        if e.code == 400:
            return {"available": False, "error": str(e)}
        raise
