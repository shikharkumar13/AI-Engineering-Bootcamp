"""
evaluation.py — Content-quality regression gate

Follows the same baseline/regression pattern as Phase 8's evaluation.py,
adapted the way Phase 8's own README suggests for a non-RAG service:
"swap RAGAS for DeepEval custom metrics if not RAG-specific." Here that
means an LLM-as-judge coherence/on-topic score per platform plus a
structural length check (each platform has a real length constraint —
LinkedIn posts and tweets that blow past their limits are broken output,
not a matter of taste), instead of RAGAS's retrieval-grounded metrics.

Run standalone:
    python evaluation.py

Run as a pytest gate:
    pytest evaluation.py -v
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASELINE_FILE = "eval_baseline.json"
REGRESSION_THRESHOLD = 0.05

DEFAULT_TOPICS = [
    "The rise of small, efficient language models",
    "Why observability matters for AI products",
]

PLATFORM_LIMITS = {
    "blog": (400, 900),      # words
    "linkedin": (0, 1300),   # characters
    "twitter": (0, 2000),    # characters, roughly a 5-7 tweet thread
}

_judge_client = OpenAI()


def _length_ok(platform: str, content: str) -> bool:
    lo, hi = PLATFORM_LIMITS[platform]
    length = len(content.split()) if platform == "blog" else len(content)
    return lo <= length <= hi


def _judge_score(topic: str, platform: str, content: str) -> float:
    """LLM-as-judge: is this content coherent and actually about the topic?
    Returns a 0-1 score. Mirrors Phase 7's evaluate.py judge pattern."""
    prompt = (
        f"Rate this {platform} content on a 0-10 scale for: (a) staying on "
        f"topic ('{topic}'), (b) internal coherence, (c) being appropriately "
        f"formatted for {platform}. Respond with ONLY the number.\n\n"
        f"Content:\n{content[:2000]}"
    )
    r = _judge_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5,
    )
    try:
        return max(0.0, min(1.0, float(r.choices[0].message.content.strip()) / 10))
    except ValueError:
        return 0.0


class EvaluationSuite:
    def __init__(self, topics: list[str] | None = None):
        self.topics = topics or DEFAULT_TOPICS

    def run(self, desk) -> dict:
        """Runs each topic through the full pipeline and scores every
        platform's output. Returns per-platform average judge scores plus
        an aggregate structural pass rate."""
        judge_scores = {"blog": [], "linkedin": [], "twitter": []}
        structural_passes = 0
        structural_total = 0

        for i, topic in enumerate(self.topics, 1):
            print(f"  [{i}/{len(self.topics)}] generating + scoring: {topic[:50]}...")
            result = desk.generate(topic)

            for platform, content in result.platform_content.items():
                judge_scores[platform].append(_judge_score(topic, platform, content))
                structural_total += 1
                if _length_ok(platform, content):
                    structural_passes += 1

        scores = {
            f"{platform}_quality": (sum(vals) / len(vals) if vals else 0.0)
            for platform, vals in judge_scores.items()
        }
        scores["structural_pass_rate"] = (
            structural_passes / structural_total if structural_total else 0.0
        )
        return scores


def load_baseline() -> dict | None:
    path = Path(BASELINE_FILE)
    return json.loads(path.read_text()) if path.exists() else None


def save_baseline(scores: dict):
    Path(BASELINE_FILE).write_text(json.dumps(scores, indent=2))


def check_regression(current: dict, baseline: dict, threshold: float = REGRESSION_THRESHOLD) -> dict:
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


def print_report(current: dict, regression_report: dict) -> bool:
    print(f"\n{'=' * 62}")
    print("  CONTENT DESK EVALUATION REPORT")
    print(f"{'=' * 62}")
    any_regression = False
    for metric, score in current.items():
        info = regression_report.get(metric, {})
        if info.get("baseline") is not None:
            delta = info["delta"]
            flag = "REGRESSION" if info["regression"] else ("OK" if delta >= 0 else "watch")
            print(f"  {metric:24s} {score:.3f}  (baseline: {info['baseline']:.3f}, "
                  f"delta={delta:+.3f}) [{flag}]")
            any_regression = any_regression or info["regression"]
        else:
            print(f"  {metric:24s} {score:.3f}  (no baseline yet)")
    print(f"{'=' * 62}")
    print("  REGRESSIONS DETECTED — review before deploying" if any_regression
          else "  No regressions detected")
    return not any_regression


def main() -> bool:
    from crew import ContentDesk

    desk = ContentDesk(verbose=False)
    suite = EvaluationSuite()
    current_scores = suite.run(desk)

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


def test_no_regression():
    """pytest entry point: `pytest evaluation.py -v` collects this (and, before
    this function existed, nothing else in this file — pytest only discovers
    `test_*` functions/classes, so the invocation silently ran zero tests
    despite exiting 0). Fails the test if main() reports a regression."""
    assert main(), "Content quality regression detected — see the printed report above."


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
