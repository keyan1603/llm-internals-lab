# common/evaluator.py
# Different shape from every earlier repo's evaluator, deliberately.
# This post is about what a hosted API's internals actually let you
# observe, so the thing worth checking is whether embeddings-based
# routing (the one real, working window into the model's internal
# representation space this post has) actually classifies tickets
# correctly, plus how wide the real similarity margin is between the
# winning category and the runner-up, since that margin is what
# confidence.py falls back on when logprobs aren't available.

import time


def check_routing_correctness(chosen_category: str, expected_category: str) -> dict:
    return {"correct": chosen_category == expected_category}


class RoutingEvaluator:
    def __init__(self, cases: list):
        """cases: list of {"case_name": str, "ticket_text": str, "expected_category": str}."""
        self.cases = cases

    def run(self, pipeline_fn) -> dict:
        """pipeline_fn(ticket_text: str) -> {"category": str, "margin": float,
        "similarities": dict}, the real return shape of
        scenario_ticket_triage.categorize.categorize_ticket."""
        per_case = []
        for case in self.cases:
            start = time.monotonic()
            result = pipeline_fn(case["ticket_text"])
            latency_ms = round((time.monotonic() - start) * 1000, 2)

            correctness = check_routing_correctness(result["category"], case["expected_category"])
            per_case.append({
                "case_name": case["case_name"],
                "correct": correctness["correct"],
                "expected_category": case["expected_category"],
                "chosen_category": result["category"],
                "margin": result["margin"],
                "latency_ms": latency_ms
            })

        correct_count = sum(1 for c in per_case if c["correct"])
        return {
            "routing_accuracy": correct_count / len(per_case) if per_case else None,
            "correct_count": correct_count,
            "total_count": len(per_case),
            "avg_margin": sum(c["margin"] for c in per_case) / len(per_case) if per_case else None,
            "avg_latency_ms": sum(c["latency_ms"] for c in per_case) / len(per_case) if per_case else None,
            "per_case": per_case
        }
