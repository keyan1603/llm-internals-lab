# common/drift_monitor.py
# Same save-baseline/check-drift discipline as every earlier repo,
# applied here to the embeddings-based routing pipeline. routing_accuracy
# is the metric that matters most, a change to category descriptions or
# embedding config that quietly makes routing less reliable is exactly
# the kind of regression this exists to catch.

import json
import os
from pathlib import Path

DRIFT_ACCURACY_THRESHOLD = float(os.getenv("DRIFT_ACCURACY_THRESHOLD", "0.1"))
DRIFT_LATENCY_THRESHOLD_PCT = float(os.getenv("DRIFT_LATENCY_THRESHOLD_PCT", "0.5"))


def save_baseline(eval_results: dict, baseline_path: Path):
    """Call this once, deliberately, after a run you have reviewed and
    trust. Never call it automatically as part of a regular run, or a
    silent regression could get saved as the new normal instead of
    getting flagged."""
    baseline = {
        "routing_accuracy": eval_results["routing_accuracy"],
        "avg_margin": eval_results["avg_margin"],
        "avg_latency_ms": eval_results["avg_latency_ms"]
    }
    baseline_path = Path(baseline_path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2)


def check_drift(eval_results: dict, baseline_path: Path) -> dict:
    baseline_path = Path(baseline_path)
    if not baseline_path.exists():
        return {"status": "no_baseline", "message": "No baseline saved yet. Run with save_baseline() first."}

    with open(baseline_path) as f:
        baseline = json.load(f)

    current = {
        "routing_accuracy": eval_results["routing_accuracy"],
        "avg_margin": eval_results["avg_margin"],
        "avg_latency_ms": eval_results["avg_latency_ms"]
    }
    deltas = {k: round(current[k] - baseline[k], 4) for k in current}

    accuracy_dropped = deltas["routing_accuracy"] < -DRIFT_ACCURACY_THRESHOLD
    latency_up_pct = deltas["avg_latency_ms"] / baseline["avg_latency_ms"] if baseline["avg_latency_ms"] else 0
    latency_regressed = latency_up_pct > DRIFT_LATENCY_THRESHOLD_PCT

    return {
        "status": "drift_detected" if (accuracy_dropped or latency_regressed) else "stable",
        "accuracy_dropped": accuracy_dropped,
        "latency_regressed": latency_regressed,
        "baseline": baseline,
        "current": current,
        "deltas": deltas
    }
