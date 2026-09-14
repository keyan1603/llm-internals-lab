# tests_offline/test_data_mock.py
# Pure sanity checks on the fixture domain, no LLM, no API, no mocking
# needed at all. Exists so a future edit to data.py can't silently
# introduce a ticket whose expected_category doesn't match any real
# category, which would make every downstream eval result meaningless
# without ever raising an error on its own.

from scenario_ticket_triage.data import CATEGORIES, TICKETS


def main():
    assert set(CATEGORIES.keys()) == {"billing", "technical", "account"}, f"got {list(CATEGORIES.keys())}"
    print("  [exactly the 3 expected categories] OK")

    ticket_ids = [t["ticket_id"] for t in TICKETS]
    assert len(ticket_ids) == len(set(ticket_ids)), f"duplicate ticket_id in {ticket_ids}"
    print("  [no duplicate ticket_id] OK")

    for t in TICKETS:
        assert t["expected_category"] in CATEGORIES, f"{t['ticket_id']} has unknown expected_category {t['expected_category']}"
    print("  [every ticket's expected_category is a real category] OK")

    assert len(TICKETS) >= 6, "expected at least 6 tickets for a meaningful eval set"
    print("  [enough tickets for a meaningful eval] OK")

    print("data offline validation: all checks passed")


if __name__ == "__main__":
    main()
