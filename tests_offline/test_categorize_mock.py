# tests_offline/test_categorize_mock.py
# cosine_similarity is tested directly as a pure function. categorize_ticket
# is tested against a fake embed_with_backoff (monkeypatched on the
# categorize module, matching the real return shape,
# result.embeddings[0].values) so the ranking/margin logic can be
# verified without a real API call or a real embedding model.

from types import SimpleNamespace
import scenario_ticket_triage.categorize as categorize
from scenario_ticket_triage.categorize import cosine_similarity, embed_categories, categorize_ticket


def _fake_embed_response(vector):
    return SimpleNamespace(embeddings=[SimpleNamespace(values=vector)])


def test_cosine_similarity():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0, "identical vectors should have similarity 1.0"
    assert abs(cosine_similarity([1, 0], [0, 1])) < 1e-9, "orthogonal vectors should have similarity 0.0"
    assert cosine_similarity([0, 0], [1, 0]) == 0.0, "a zero vector should not divide by zero"
    print("  [cosine_similarity handles identical, orthogonal, and zero vectors] OK")


def test_categorize_ticket(monkeypatch):
    fake_vectors = {
        "billing description": [1.0, 0.0, 0.0],
        "technical description": [0.0, 1.0, 0.0],
        "account description": [0.0, 0.0, 1.0],
        "ticket text": [0.9, 0.1, 0.0],
    }

    def fake_embed_with_backoff(client, model, text, task_type=None, title=None):
        return _fake_embed_response(fake_vectors[text])

    monkeypatch.setattr(categorize, "embed_with_backoff", fake_embed_with_backoff)

    category_embeddings = embed_categories(None, "fake-model", {
        "billing": "billing description", "technical": "technical description", "account": "account description"
    })
    assert set(category_embeddings.keys()) == {"billing", "technical", "account"}
    print("  [embed_categories returns one vector per category] OK")

    result = categorize_ticket(None, "fake-model", "ticket text", category_embeddings)
    assert result["category"] == "billing", f"expected billing to win (closest vector), got {result['category']}"
    assert result["margin"] > 0, f"expected a positive margin between the winner and runner-up, got {result['margin']}"
    print("  [categorize_ticket picks the closest category and computes a positive margin] OK")


def main():
    test_cosine_similarity()

    class _FakeMonkeypatch:
        def setattr(self, obj, name, value):
            setattr(obj, name, value)

    test_categorize_ticket(_FakeMonkeypatch())
    print("categorize offline validation: all checks passed")


if __name__ == "__main__":
    main()
