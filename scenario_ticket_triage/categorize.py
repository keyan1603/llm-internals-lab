# scenario_ticket_triage/categorize.py
# Real embeddings-based routing, no LLM call in the hot path at all.
# Category descriptions get embedded once with task_type="RETRIEVAL_DOCUMENT"
# (the side being searched over), an incoming ticket gets embedded with
# task_type="RETRIEVAL_QUERY" (the side doing the searching), confirmed
# via a real embed_content call that this asymmetry is a real, accepted
# config option, not a guess. Cosine similarity between the ticket vector
# and each category vector is the actual geometry of the model's
# internal representation space, this is the one genuinely real window
# into "internals" this whole post can point at directly.

import math
from common.llm_utils import embed_with_backoff

EMBEDDING_MODEL_ENV_DEFAULT = "models/gemini-embedding-001"


def cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def embed_categories(client, embedding_model: str, categories: dict) -> dict:
    """Embeds every category description once with task_type=RETRIEVAL_DOCUMENT.
    Returns {category_name: vector}."""
    return {
        name: embed_with_backoff(client, embedding_model, description, task_type="RETRIEVAL_DOCUMENT", title=name).embeddings[0].values
        for name, description in categories.items()
    }


def categorize_ticket(client, embedding_model: str, ticket_text: str, category_embeddings: dict) -> dict:
    """Embeds one ticket with task_type=RETRIEVAL_QUERY and returns the
    best-matching category, every similarity score, and the margin
    between the top match and the runner-up. That margin is what
    scenario_ticket_triage/confidence.py falls back on when real
    per-token logprobs aren't available (see confidence.py)."""
    ticket_vector = embed_with_backoff(client, embedding_model, ticket_text, task_type="RETRIEVAL_QUERY").embeddings[0].values

    similarities = {
        name: cosine_similarity(ticket_vector, category_vector)
        for name, category_vector in category_embeddings.items()
    }
    ranked = sorted(similarities.items(), key=lambda kv: kv[1], reverse=True)
    best_category, best_score = ranked[0]
    margin = best_score - ranked[1][1] if len(ranked) > 1 else best_score

    return {
        "category": best_category,
        "similarities": similarities,
        "margin": margin
    }
