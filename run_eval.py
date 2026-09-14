# run_eval.py
# Runs every ticket in eval_cases.py through the real embeddings-based
# routing pipeline and reports routing_accuracy: did the ticket land in
# its expected category, plus the real similarity margin, since that
# margin is what confidence.py falls back on in place of logprobs.

import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from common.drift_monitor import save_baseline, check_drift
from common.evaluator import RoutingEvaluator
from scenario_ticket_triage.data import CATEGORIES
from scenario_ticket_triage.categorize import embed_categories, categorize_ticket
from eval_cases import EVAL_CASES

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
BASELINE_PATH = Path("baselines/ticket_routing.json")


def main(save: bool):
    category_embeddings = embed_categories(client, EMBEDDING_MODEL, CATEGORIES)

    def pipeline_fn(ticket_text: str):
        return categorize_ticket(client, EMBEDDING_MODEL, ticket_text, category_embeddings)

    results = RoutingEvaluator(EVAL_CASES).run(pipeline_fn)
    print(f"routing_accuracy: {results['routing_accuracy']} ({results['correct_count']}/{results['total_count']})")
    print(f"avg_margin: {results['avg_margin']:.4f}")
    print(f"avg_latency_ms: {results['avg_latency_ms']}")
    for c in results["per_case"]:
        mark = "OK" if c["correct"] else "WRONG"
        print(f"  [{mark}] {c['case_name']}: expected={c['expected_category']}, chosen={c['chosen_category']}, margin={c['margin']:.4f}")

    if save:
        save_baseline(results, BASELINE_PATH)
        print(f"saved baseline -> {BASELINE_PATH}")
    else:
        drift = check_drift(results, BASELINE_PATH)
        print(f"drift check: {drift['status']}")
        if drift["status"] != "no_baseline":
            print(f"deltas: {drift['deltas']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate embeddings-based ticket routing against eval_cases.py.")
    parser.add_argument("--save-baseline", action="store_true",
                         help="Save this run's results as the new baseline instead of checking drift against the existing one.")
    args = parser.parse_args()
    main(args.save_baseline)
