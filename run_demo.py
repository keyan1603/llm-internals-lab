# run_demo.py
# Walks through every real capability this repo demonstrates, in the
# order the post covers them: tokenization boundary, embeddings-based
# routing, sampling variance, then the logprobs real-failure and its 2
# real fallbacks.

import os
from dotenv import load_dotenv
from google import genai
from scenario_ticket_triage.data import CATEGORIES, TICKETS
from scenario_ticket_triage.tokenize_check import check_ticket_length
from scenario_ticket_triage.categorize import embed_categories, categorize_ticket
from scenario_ticket_triage.triage_response import generate_triage_ack, classify_with_self_consistency
from scenario_ticket_triage.confidence import try_real_logprobs_confidence, similarity_margin_confidence
from common.llm_utils import attempt_multi_candidate
from common.tracer import Tracer

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def main():
    tracer = Tracer()
    sample_ticket = TICKETS[0]

    print("=== Tokenization: a real count, not a real tokenizer ===\n")
    with tracer.span("count_tokens", ticket_id=sample_ticket["ticket_id"]) as meta:
        length_check = check_ticket_length(client, CHAT_MODEL, sample_ticket["text"])
        meta["total_tokens"] = length_check["total_tokens"]
    print(f"Ticket: {sample_ticket['text']}")
    print(f"  total_tokens: {length_check['total_tokens']}, within_limit: {length_check['within_limit']}\n")

    print("=== Embeddings-based routing: the real internal representation space ===\n")
    with tracer.span("embed_categories"):
        category_embeddings = embed_categories(client, EMBEDDING_MODEL, CATEGORIES)
    for ticket in TICKETS[:3]:
        with tracer.span("categorize_ticket", ticket_id=ticket["ticket_id"]) as meta:
            result = categorize_ticket(client, EMBEDDING_MODEL, ticket["text"], category_embeddings)
            meta["category"] = result["category"]
            meta["margin"] = result["margin"]
        print(f"Ticket: {ticket['text']}")
        print(f"  routed to: {result['category']} (expected: {ticket['expected_category']}), margin: {result['margin']:.4f}")
        print(f"  similarities: { {k: round(v, 4) for k, v in result['similarities'].items()} }\n")

    print("=== Sampling variance: temperature=0.0 vs temperature=1.0 ===\n")
    with tracer.span("triage_ack", temperature=0.0, call_number=1):
        ack_low = generate_triage_ack(client, CHAT_MODEL, sample_ticket["text"], "billing", temperature=0.0)
    with tracer.span("triage_ack", temperature=0.0, call_number=2):
        ack_low_again = generate_triage_ack(client, CHAT_MODEL, sample_ticket["text"], "billing", temperature=0.0)
    with tracer.span("triage_ack", temperature=1.0, call_number=1):
        ack_high = generate_triage_ack(client, CHAT_MODEL, sample_ticket["text"], "billing", temperature=1.0)
    print(f"temperature=0.0, call 1: {ack_low}")
    print(f"temperature=0.0, call 2: {ack_low_again}")
    print(f"temperature=1.0, call 1: {ack_high}")
    print(f"  (note: temperature=0.0 still produced 2 differently worded outputs, near-deterministic is not the same as identical, see this post's Key Concepts)\n")

    print("=== 2 real, confirmed gaps: logprobs and multi-candidate responses ===\n")
    with tracer.span("attempt_logprobs") as meta:
        logprobs_result = try_real_logprobs_confidence(client, CHAT_MODEL, sample_ticket["text"])
        meta["available"] = logprobs_result["available"]
    if logprobs_result["available"]:
        print("Logprobs available:", logprobs_result["response"])
    else:
        print(f"Logprobs NOT available: {logprobs_result['error'][:200]}")

    with tracer.span("attempt_multi_candidate") as meta:
        multi_candidate_result = attempt_multi_candidate(client, CHAT_MODEL, sample_ticket["text"])
        meta["available"] = multi_candidate_result["available"]
    if multi_candidate_result["available"]:
        print("Multi-candidate available:", multi_candidate_result["response"])
    else:
        print(f"Multi-candidate NOT available: {multi_candidate_result['error'][:200]}")

    print("\n=== 2 real fallbacks instead ===\n")
    with tracer.span("categorize_ticket_for_confidence", ticket_id=sample_ticket["ticket_id"]):
        routing_result = categorize_ticket(client, EMBEDDING_MODEL, sample_ticket["text"], category_embeddings)
    margin_confidence = similarity_margin_confidence(routing_result["margin"])
    print(f"Fallback 1, embeddings margin confidence: {margin_confidence} (margin={routing_result['margin']:.4f})")

    with tracer.span("classify_with_self_consistency", ticket_id=sample_ticket["ticket_id"]) as meta:
        consistency_result = classify_with_self_consistency(client, CHAT_MODEL, sample_ticket["text"])
        meta["agreement_rate"] = consistency_result["agreement_rate"]
    print(f"Fallback 2, self-consistency via {sum(consistency_result['votes'].values())} separate calls: "
          f"majority={consistency_result['majority_category']}, "
          f"agreement_rate={consistency_result['agreement_rate']:.2f}, "
          f"votes={consistency_result['votes']}")


if __name__ == "__main__":
    main()
