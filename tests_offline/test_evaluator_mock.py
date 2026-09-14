# tests_offline/test_evaluator_mock.py
# check_routing_correctness and RoutingEvaluator tested against a fake
# pipeline_fn, no real embeddings or API calls involved.

from common.evaluator import check_routing_correctness, RoutingEvaluator


def test_check_routing_correctness():
    assert check_routing_correctness("billing", "billing")["correct"] is True
    assert check_routing_correctness("billing", "technical")["correct"] is False
    print("  [check_routing_correctness compares chosen vs expected] OK")


def _fake_pipeline_fn(ticket_text: str):
    if "refund" in ticket_text:
        return {"category": "billing", "margin": 0.08, "similarities": {}}
    return {"category": "technical", "margin": 0.01, "similarities": {}}


def test_routing_evaluator():
    cases = [
        {"case_name": "T1", "ticket_text": "I want a refund", "expected_category": "billing"},
        {"case_name": "T2", "ticket_text": "something is broken", "expected_category": "billing"},
    ]
    results = RoutingEvaluator(cases).run(_fake_pipeline_fn)
    assert results["routing_accuracy"] == 0.5, f"expected 1 of 2 correct, got {results['routing_accuracy']}"
    assert results["per_case"][0]["correct"] is True
    assert results["per_case"][1]["correct"] is False
    assert abs(results["avg_margin"] - 0.045) < 1e-9, f"expected avg of 0.08 and 0.01, got {results['avg_margin']}"
    print("  [RoutingEvaluator aggregates accuracy and margin correctly] OK")


def main():
    test_check_routing_correctness()
    test_routing_evaluator()
    print("evaluator offline validation: all checks passed")


if __name__ == "__main__":
    main()
