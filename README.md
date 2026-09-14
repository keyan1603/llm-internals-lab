# llm-internals-lab

Companion repo for the "LLM Internals" post (#15) on
[karthiksolution.wordpress.com](https://karthiksolution.wordpress.com/),
in the AI/agents series.

## What's in here

A small support-ticket-triage system built around the question: what
does a hosted Gemini API actually let you see about a model's
internals, and what does it only look like it lets you see?

- **Tokenization** (`scenario_ticket_triage/tokenize_check.py`): a real
  `count_tokens` call before sending a ticket anywhere. Real count,
  never the actual token split, `compute_tokens` (which does return
  real token strings/IDs) is Vertex-AI-only and unusable with the
  Developer API key this repo runs on.
- **Embeddings-based routing** (`scenario_ticket_triage/categorize.py`):
  a real `embed_content` call routes each ticket to a category (billing,
  technical, account) by cosine similarity, no LLM call in the hot path
  at all. This is the one genuinely real window into the model's
  internal representation space a REST API can offer.
- **Sampling variance** (`scenario_ticket_triage/triage_response.py`):
  real `temperature` comparisons, including the honest finding that
  temperature=0.0 does not guarantee byte-identical repeated outputs on
  the real API.
- **2 real, confirmed gaps, not crashes** (`common/llm_utils.py`,
  `scenario_ticket_triage/confidence.py`): `response_logprobs` and
  `candidate_count > 1` both return a real, confirmed `400
  INVALID_ARGUMENT` on `gemini-3.5-flash-lite`. Both are demonstrated as
  structured, expected failures, with 2 real working fallbacks instead:
  an embeddings similarity margin, and self-consistency via repeated
  single-candidate calls.

`common/evaluator.py`'s `RoutingEvaluator` checks routing accuracy and
the real similarity margin against 8 real tickets with known expected
categories, no LLM judge, this post is about what's actually observable,
not answer quality.

## Setup (Windows / PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY` to your real key.

## Running the demo

```powershell
python run_demo.py
```

## Running the offline validation (no API key required beyond a placeholder)

```powershell
python -m tests_offline.test_data_mock
python -m tests_offline.test_categorize_mock
python -m tests_offline.test_evaluator_mock
python -m tests_offline.test_llm_utils_mock
```

## Running the evaluator

```powershell
python run_eval.py --save-baseline
python run_eval.py
```

## License

MIT, see [LICENSE](LICENSE).
