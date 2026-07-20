# tests/core/test_pipeline.py
import pytest

from photon.core.contract import RouteTarget
from photon.core.pipeline import PipelineOrchestrator, PipelineSpec, StageSpec
from photon.core.serving import MockServingBackend


def _resp(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def _two_stage_spec(budget=None) -> PipelineSpec:
    return PipelineSpec(
        id="praxiom-mini",
        stages=[
            StageSpec(name="parse", target=RouteTarget(model_id="parser")),
            StageSpec(name="generate", target=RouteTarget(model_id="generator")),
        ],
        latency_budget_ms=budget,
    )


async def test_stages_run_in_order_and_chain_context():
    backend = MockServingBackend(canned={"parser": _resp("PARSED"), "generator": _resp("FINAL")})
    result = await PipelineOrchestrator(backend).execute(
        _two_stage_spec(), {"messages": [{"role": "user", "content": "goal"}]}
    )
    assert result.completed is True
    assert [o.name for o in result.stage_outputs] == ["parse", "generate"]
    assert [o.content for o in result.stage_outputs] == ["PARSED", "FINAL"]
    # the generator stage saw the parser's output as chained context
    assert [c.model_id for c in backend.calls] == ["parser", "generator"]


async def test_per_stage_routing_targets_respected():
    backend = MockServingBackend(canned={"parser": _resp("a"), "generator": _resp("b")})
    spec = PipelineSpec(
        id="p",
        stages=[
            StageSpec(name="s1", target=RouteTarget(model_id="parser", adapter_id="intent-v3")),
            StageSpec(name="s2", target=RouteTarget(model_id="generator")),
        ],
    )
    await PipelineOrchestrator(backend).execute(spec, {"messages": []})
    assert backend.calls[0].adapter_id == "intent-v3"
    assert backend.calls[1].adapter_id is None


async def test_budget_aborts_between_stages():
    # a budget of 0 is exhausted before the first stage runs
    backend = MockServingBackend(canned={"parser": _resp("a"), "generator": _resp("b")})
    result = await PipelineOrchestrator(backend).execute(
        _two_stage_spec(budget=0.0), {"messages": []}
    )
    assert result.completed is False
    assert result.aborted_at_stage == "parse"
    assert result.stage_outputs == []
    assert backend.calls == []  # nothing ran


def test_pipeline_spec_requires_at_least_one_stage():
    with pytest.raises(Exception):
        PipelineSpec(id="empty", stages=[])
