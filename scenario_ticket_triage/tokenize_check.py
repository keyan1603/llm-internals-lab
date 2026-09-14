# scenario_ticket_triage/tokenize_check.py
# A real boundary, not a workaround: count_tokens on the Developer API
# returns only a total count (confirmed by reading the installed SDK's
# CountTokensResponse, which has total_tokens and cached_content_token_count,
# nothing else). The SDK also has compute_tokens, which does return real
# token strings and IDs, but its own source raises ValueError unless the
# client is configured for Vertex AI, not usable with the Developer API
# key this whole series runs on. So a ticket-length check before sending
# anything to the model can tell you how many tokens it costs, never how
# the text actually got split.

from common.llm_utils import count_tokens_with_backoff


def check_ticket_length(client, model: str, ticket_text: str, max_tokens: int = 200) -> dict:
    result = count_tokens_with_backoff(client, model, ticket_text)
    total = result.total_tokens
    return {
        "total_tokens": total,
        "within_limit": total <= max_tokens,
        "max_tokens": max_tokens
    }
