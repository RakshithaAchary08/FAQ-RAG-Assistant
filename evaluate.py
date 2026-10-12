"""Evaluate FAQ retrieval against tests/eval_questions.json."""

import json
from pathlib import Path

from dotenv import load_dotenv

from database import ensure_index
from rag import search_faq

BASE_DIR = Path(__file__).resolve().parent
EVAL_PATH = BASE_DIR / "tests" / "eval_questions.json"


def main() -> None:
    """Print retrieval and fallback metrics with per-question failures."""
    load_dotenv(BASE_DIR / ".env")
    ensure_index()
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    in_scope = [case for case in cases if not case["should_fallback"]]
    out_of_scope = [case for case in cases if case["should_fallback"]]

    hit_at_1 = 0
    hit_at_3 = 0
    retrieval_failures: list[tuple[dict[str, object], list[int]]] = []
    fallback_hits = 0
    fallback_failures: list[tuple[dict[str, object], list[int]]] = []

    for case in in_scope:
        results = search_faq(case["question"], top_k=3)
        result_ids = [int(result["id"]) for result in results]
        expected_id = case["expected_faq_id"]
        if result_ids and result_ids[0] == expected_id:
            hit_at_1 += 1
        if expected_id in result_ids[:3]:
            hit_at_3 += 1
        else:
            retrieval_failures.append((case, result_ids))

    for case in out_of_scope:
        results = search_faq(case["question"], top_k=3)
        result_ids = [int(result["id"]) for result in results]
        if not results:
            fallback_hits += 1
        else:
            fallback_failures.append((case, result_ids))

    print(f"hit@1: {hit_at_1 / len(in_scope):.3%} ({hit_at_1}/{len(in_scope)})")
    print(f"hit@3: {hit_at_3 / len(in_scope):.3%} ({hit_at_3}/{len(in_scope)})")
    print(
        "fallback accuracy: "
        f"{fallback_hits / len(out_of_scope):.3%} "
        f"({fallback_hits}/{len(out_of_scope)})"
    )

    if retrieval_failures:
        print("\nRetrieval failures:")
        for case, result_ids in retrieval_failures:
            print(
                f"- expected FAQ {case['expected_faq_id']}; got {result_ids}: "
                f"{case['question']}"
            )
    if fallback_failures:
        print("\nOut-of-scope questions that did not trigger fallback:")
        for case, result_ids in fallback_failures:
            print(f"- got {result_ids}: {case['question']}")


if __name__ == "__main__":
    main()
