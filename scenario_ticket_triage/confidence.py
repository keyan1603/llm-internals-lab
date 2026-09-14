# scenario_ticket_triage/confidence.py
# What you'd actually want for a triage system: per-token confidence on
# the category classification decision, straight from the model. What
# you actually get on every current Gemini Developer-API model: a real,
# confirmed 400 (see common/llm_utils.py's attempt_logprobs_confidence
# and this post's Pitfalls section). This module shows both the real
# failure and the 2 real, working alternatives this repo already
# computes elsewhere: the embeddings similarity margin from
# categorize.categorize_ticket, and self-consistency agreement from
# triage_response.classify_with_self_consistency.

from common.llm_utils import attempt_logprobs_confidence
from scenario_ticket_triage.triage_response import CLASSIFY_PROMPT_TEMPLATE

MARGIN_HIGH_THRESHOLD = 0.05
MARGIN_LOW_THRESHOLD = 0.02


def try_real_logprobs_confidence(client, model: str, ticket_text: str) -> dict:
    """Returns the real, structured result of attempting logprobs on a
    classification prompt, {"available": False, "error": ...} on every
    currently available Gemini Developer-API model, confirmed directly,
    not assumed."""
    prompt = CLASSIFY_PROMPT_TEMPLATE.format(ticket_text=ticket_text)
    return attempt_logprobs_confidence(client, model, prompt)


def similarity_margin_confidence(margin: float) -> str:
    """A practical substitute for per-token confidence: how much wider
    the winning category's similarity is than the runner-up's. A small
    margin means the ticket sat ambiguously between 2 categories in the
    embedding space, a large margin means it was unambiguous."""
    if margin >= MARGIN_HIGH_THRESHOLD:
        return "high"
    if margin >= MARGIN_LOW_THRESHOLD:
        return "medium"
    return "low"
