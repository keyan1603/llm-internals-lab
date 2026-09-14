# scenario_ticket_triage/data.py
# A small support-ticket-triage domain: 3 categories, each described in
# enough detail to embed meaningfully, and a set of real-sounding
# tickets with a known correct category for the offline tests and the
# real eval. Deliberately includes a couple of tickets that lean on
# word choice a naive keyword match would get wrong (e.g. "charge" and
# "charges" appear in both a billing ticket and nowhere near a billing
# ticket), so routing has to actually depend on embedding similarity,
# not surface keyword overlap.

CATEGORIES = {
    "billing": (
        "Billing and payment issues, including charges, invoices, refunds, "
        "subscription costs, plan changes, and payment method problems."
    ),
    "technical": (
        "Technical problems with the product itself, including bugs, errors, "
        "crashes, slow performance, and features not working as expected."
    ),
    "account": (
        "Account access and management issues, including login problems, "
        "password resets, account settings, and profile changes."
    ),
}

TICKETS = [
    {"ticket_id": "T1", "text": "I was charged twice for my subscription this month, can I get a refund?", "expected_category": "billing"},
    {"ticket_id": "T2", "text": "The app crashes every time I try to upload a photo.", "expected_category": "technical"},
    {"ticket_id": "T3", "text": "I can't log into my account, it says my password is incorrect even after resetting it.", "expected_category": "account"},
    {"ticket_id": "T4", "text": "My invoice this month shows a charge I don't recognize, what is this for?", "expected_category": "billing"},
    {"ticket_id": "T5", "text": "The search feature returns no results even for terms I know exist.", "expected_category": "technical"},
    {"ticket_id": "T6", "text": "I need to change the email address associated with my account.", "expected_category": "account"},
    {"ticket_id": "T7", "text": "Is it possible to downgrade my plan to a cheaper tier?", "expected_category": "billing"},
    {"ticket_id": "T8", "text": "The dashboard takes over a minute to load every time I open it.", "expected_category": "technical"},
]
