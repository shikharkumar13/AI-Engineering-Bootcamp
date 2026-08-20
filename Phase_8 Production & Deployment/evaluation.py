"""
evaluation.py — RAGAS Evaluation Suite

A versioned, reusable test set for regression-testing the RAG pipeline.
Run this after ANY change to chunking, retrieval, prompts, or models to
detect quality regressions before they reach production.

Run standalone:
    python evaluation.py

Run as a pytest gate:
    pytest evaluation.py -v
"""

import json
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

EVAL_SUITE_FILE = "eval_suite.json"
BASELINE_FILE   = "eval_baseline.json"
REGRESSION_THRESHOLD = 0.05   # fail if any metric drops more than this vs baseline


# ── Default test suite (swap in your own domain questions) ────────────────────

DEFAULT_TEST_CASES = [
    {
        "question": "What is the refund policy for annual subscriptions?",
        "ground_truth": "Annual subscriptions are eligible for a prorated refund "
                         "within the first 30 days of purchase.",
        "tags": ["billing", "policy"],
    },
    {
        "question": "How do I reset my password?",
        "ground_truth": "Use the 'Forgot Password' link on the login page to "
                         "receive a reset email.",
        "tags": ["account", "how-to"],
    },
    {
        "question": "What happens to my data if I cancel my account?",
        "ground_truth": "Data is retained for 30 days after cancellation, after "
                         "which it is permanently deleted.",
        "tags": ["account", "data", "policy"],
    },
    {
        "question": "Can I downgrade from Pro to Free mid-cycle?",
        "ground_truth": "Yes, downgrades take effect at the end of the current "
                         "billing cycle with no service interruption before then.",
        "tags": ["billing", "how-to"],
    },
]


class EvaluationSuite:
    """
    Maintains a versioned set of (question, ground_truth) test cases and runs
    RAGAS metrics against your live RAG service.
    """

    def __init__(self, test_file: str = EVAL_SUITE_FILE):
        self.test_file = test_file
        self.test_cases = self._load_or_seed()

    def _load_or_seed(self) -> list[dict]:
        path = Path(self.test_file)
        if path.exists():
            return json.loads(path.read_text())
        # Seed with defaults on first run
        path.write_text(json.dumps(DEFAULT_TEST_CASES, indent=2))
        return DEFAULT_TEST_CASES

    def add_case(self, question: str, ground_truth: str, tags: list[str] | None = None):
        self.test_cases.append({"question": question, "ground_truth": ground_truth,
                                  "tags": tags or []})
        Path(self.test_file).write_text(json.dumps(self.test_cases, indent=2))

    def run(self, rag_service, metrics: list | None = None) -> dict:
        """
        Run the full suite against a live RAGService instance.
        Returns aggregate scores per metric.
        """
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        metrics = metrics or [faithfulness, answer_relevancy, context_precision, context_recall]

        eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

        print(f"  Running {len(self.test_cases)} test cases through the RAG pipeline...")
        for i, case in enumerate(self.test_cases, 1):
            print(f"    [{i}/{len(self.test_cases)}] {case['question'][:60]}...")
            response = rag_service.ask(case["question"])

            eval_data["question"].append(case["question"])
            eval_data["answer"].append(response.answer)
            eval_data["contexts"].append([c.text for c in response.sources])
            eval_data["ground_truth"].append(case["ground_truth"])

        print(f"  Computing RAGAS metrics (this calls an LLM judge for each metric)...")
        result = evaluate(Dataset.from_dict(eval_data), metrics=metrics)

        return dict(result)

    def run_per_case(self, rag_service, metrics: list | None = None) -> list[dict]:
        """Like run(), but returns per-question scores instead of just aggregates."""
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        metrics = metrics or [faithfulness, answer_relevancy]

        eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
        for case in self.test_cases:
            response = rag_service.ask(case["question"])
            eval_data["question"].append(case["question"])
            eval_data["answer"].append(response.answer)
            eval_data["contexts"].append([c.text for c in response.sources])
            eval_data["ground_truth"].append(case["ground_truth"])

        result = evaluate(Dataset.from_dict(eval_data), metrics=metrics)
        df = result.to_pandas()
        return df.to_dict(orient="records")


def load_baseline() -> dict | None:
    path = Path(BASELINE_FILE)
    if path.exists():
        return json.loads(path.read_text())
    return None


def save_baseline(scores: dict):
    Path(BASELINE_FILE).write_text(json.dumps(scores, indent=2))


def check_regression(current: dict, baseline: dict, threshold: float = REGRESSION_THRESHOLD) -> dict:
    """Compare current scores against baseline, flag regressions."""
    report = {}
    for metric, current_score in current.items():
        baseline_score = baseline.get(metric)
        if baseline_score is None:
            report[metric] = {"current": current_score, "baseline": None, "regression": False}
            continue
        delta = current_score - baseline_score
        report[metric] = {
            "current": current_score,
            "baseline": baseline_score,
            "delta": round(delta, 4),
            "regression": delta < -threshold,
        }
    return report


def print_report(current: dict, regression_report: dict):
    print(f"\n{'='*62}")
    print(f"  RAGAS EVALUATION REPORT")
    print(f"{'='*62}")
    any_regression = False
    for metric, score in current.items():
        info = regression_report.get(metric, {})
        if info.get("baseline") is not None:
            delta = info["delta"]
            flag = "🔴 REGRESSION" if info["regression"] else ("🟢" if delta >= 0 else "🟡")
            print(f"  {metric:20s} {score:.3f}  (baseline: {info['baseline']:.3f}, Δ={delta:+.3f}) {flag}")
            if info["regression"]:
                any_regression = True
        else:
            print(f"  {metric:20s} {score:.3f}  (no baseline yet)")
    print(f"{'='*62}")
    if any_regression:
        print("  ⚠ REGRESSIONS DETECTED — review before deploying")
    else:
        print("  ✅ No regressions detected")
    return not any_regression


def main():
    from rag_service import RAGService

    print("Setting up RAG service for evaluation...")
    service = RAGService(collection_name="eval_docs", persist_dir="./eval_chroma_db")

    # Index a sample document if the collection is empty (for standalone demo runs)
    if service.stats()["total_chunks"] == 0:
        sample_doc = Path("sample_handbook.txt")
        if not sample_doc.exists():
            sample_doc.write_text(SAMPLE_HANDBOOK_TEXT)
        service.index(str(sample_doc))

    suite = EvaluationSuite()
    current_scores = suite.run(service)

    baseline = load_baseline()
    if baseline is None:
        print("\n  No baseline found — saving current scores as the new baseline.")
        save_baseline(current_scores)
        regression_report = check_regression(current_scores, current_scores)
    else:
        regression_report = check_regression(current_scores, baseline)

    passed = print_report(current_scores, regression_report)

    if baseline is None or passed:
        save_baseline(current_scores)

    return passed


SAMPLE_HANDBOOK_TEXT = """
Customer Handbook

Refunds: Annual subscriptions are eligible for a prorated refund within the
first 30 days of purchase. Monthly subscriptions are not eligible for
partial-month refunds, but you can cancel anytime to stop future charges.

Password Reset: Use the 'Forgot Password' link on the login page. You'll
receive a reset email within a few minutes. Check your spam folder if it
doesn't arrive.

Data Retention: If you cancel your account, your data is retained for 30
days in case you want to reactivate. After 30 days, all data is permanently
and irreversibly deleted from our systems.

Plan Changes: You can upgrade your plan at any time, and the change takes
effect immediately with a prorated charge for the difference. Downgrades
take effect at the end of your current billing cycle, with no interruption
to your current plan's features until then.
"""


def test_no_regression():
    """pytest entry point: `pytest evaluation.py -v` collects this (and, before
    this function existed, nothing else in this file — pytest only discovers
    `test_*` functions/classes, so the invocation silently ran zero tests
    despite exiting 0). Fails the test if main() reports a regression."""
    assert main(), "RAGAS regression detected — see the printed report above."


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
