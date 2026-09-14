# eval_cases.py
# Every ticket in scenario_ticket_triage/data.py has a known expected
# category, so the eval set is just that data reshaped for
# common.evaluator.RoutingEvaluator, no separate hand-authored case list
# to keep in sync.

from scenario_ticket_triage.data import TICKETS

EVAL_CASES = [
    {"case_name": t["ticket_id"], "ticket_text": t["text"], "expected_category": t["expected_category"]}
    for t in TICKETS
]
