# scenario_ticket_triage/triage_response.py
# Two real, distinct uses of sampling parameters:
#
# generate_triage_ack demonstrates plain temperature variance: the same
# ticket and category, asked for an acknowledgment message twice, at
# temperature=0.0 (should be near-identical) and temperature=1.0
# (should genuinely differ), a real answer to "should a production
# triage system use temperature=0 for determinism."
#
# classify_with_self_consistency is a real alternative to logprobs-based
# confidence (see common/llm_utils.py's attempt_logprobs_confidence and
# this post's Pitfalls section): instead of one call and a token-level
# probability the API won't give you, ask the same classification
# question at a non-zero temperature multiple times and measure how
# often the answers agree. High agreement is a genuine, real proxy for
# confidence.
#
# This makes N separate calls, not one call with candidate_count=N.
# candidate_count is a real GenerateContentConfig field (see
# common/llm_utils.py's generate_with_backoff), but a real call against
# gemini-3.5-flash-lite confirmed it's rejected too: "400 INVALID_ARGUMENT:
# Multiple candidates is not enabled for this model", the same shape of
# gap as the logprobs finding, a second documented SDK capability this
# model doesn't actually grant. N separate single-candidate calls is
# what the technique degrades to once you can't rely on multi-candidate
# responses, and it's honestly the more portable version of the
# technique anyway, since it doesn't depend on any model supporting
# candidate_count at all.

from collections import Counter
from common.llm_utils import generate_with_backoff

CLASSIFY_PROMPT_TEMPLATE = (
    "Classify this support ticket into exactly one category: billing, technical, or account. "
    "Reply with only the single category word, nothing else.\n\nTicket: {ticket_text}"
)


def generate_triage_ack(client, model: str, ticket_text: str, category: str, temperature: float) -> str:
    prompt = (
        f"Write a one-sentence acknowledgment message to a customer whose support ticket "
        f"was just routed to the {category} team. Ticket: {ticket_text}"
    )
    response = generate_with_backoff(client, model, prompt, temperature=temperature)
    return response.text


def classify_with_self_consistency(client, model: str, ticket_text: str, temperature: float = 1.0, num_samples: int = 3) -> dict:
    """Makes num_samples separate single-candidate calls (see this
    module's header comment for why not one candidate_count=N call) and
    measures agreement across them."""
    prompt = CLASSIFY_PROMPT_TEMPLATE.format(ticket_text=ticket_text)

    votes = []
    for _ in range(num_samples):
        response = generate_with_backoff(client, model, prompt, temperature=temperature)
        text = response.text.strip().lower() if response.text else ""
        cleaned = "".join(ch for ch in text if ch.isalpha())
        if cleaned in ("billing", "technical", "account"):
            votes.append(cleaned)

    if not votes:
        return {"votes": {}, "majority_category": None, "agreement_rate": 0.0, "num_samples_parsed": 0}

    vote_counts = Counter(votes)
    majority_category, majority_count = vote_counts.most_common(1)[0]
    return {
        "votes": dict(vote_counts),
        "majority_category": majority_category,
        "agreement_rate": majority_count / len(votes),
        "num_samples_parsed": len(votes)
    }
