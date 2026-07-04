# tests/test_evals.py
import httpx
import respx

from photon.evals.golden import GoldenCase, GoldenSet, check_case
from photon.evals.runner import run_golden_set


def make_case(**overrides) -> GoldenCase:
    base = dict(id="c1", messages=[{"role": "user", "content": "capital of France?"}])
    base.update(overrides)
    return GoldenCase(**base)


def test_check_case_contains_and_regex_pass():
    case = make_case(must_contain=["paris"], must_match=r"[Pp]aris")
    result = check_case(case, "Paris is the capital.", latency_ms=100.0)
    assert result.passed
    assert result.failures == []


def test_check_case_forbidden_substring_fails():
    case = make_case(must_not_contain=["cannot help"])
    result = check_case(case, "I cannot help with that.", latency_ms=100.0)
    assert not result.passed
    assert "forbidden" in result.failures[0]


def test_check_case_latency_budget_fails():
    case = make_case(max_latency_ms=500.0)
    result = check_case(case, "anything", latency_ms=900.0)
    assert not result.passed


@respx.mock
async def test_run_golden_set_end_to_end():
    respx.post("http://gw.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "Paris is the capital."}}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            },
        )
    )
    golden = GoldenSet(name="t", cases=[make_case(must_contain=["paris"])])
    report = await run_golden_set("http://gw.test", golden)
    assert report.total == 1
    assert report.passed == 1
