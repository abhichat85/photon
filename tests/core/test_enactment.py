# tests/core/test_enactment.py
import httpx
import pytest
import respx

from photon.core.enactment import FleetEnactor, VLLMAdapterControl
from photon.core.fleet import AdapterSpec, FleetManager, FleetSpec


@respx.mock
async def test_control_load_unload_list():
    load = respx.post("http://s.test/v1/load_lora_adapter").mock(
        return_value=httpx.Response(200, text="Success")
    )
    unload = respx.post("http://s.test/v1/unload_lora_adapter").mock(
        return_value=httpx.Response(200, text="Success")
    )
    respx.get("http://s.test/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [
            {"id": "qwen-1.5b"}, {"id": "legal-v3"},
        ]})
    )
    async with httpx.AsyncClient() as client:
        ctl = VLLMAdapterControl("http://s.test/v1", client)
        assert await ctl.load("legal-v3", "/adapters/legal-v3") is True
        assert await ctl.unload("fin-v2") is True
        assert await ctl.list_served() == ["qwen-1.5b", "legal-v3"]
    import json
    assert json.loads(load.calls.last.request.content) == {
        "lora_name": "legal-v3", "lora_path": "/adapters/legal-v3"}
    assert json.loads(unload.calls.last.request.content) == {"lora_name": "fin-v2"}


@respx.mock
async def test_control_load_failure_returns_false():
    respx.post("http://s.test/v1/load_lora_adapter").mock(
        return_value=httpx.Response(400, text="runtime lora updating disabled")
    )
    async with httpx.AsyncClient() as client:
        ctl = VLLMAdapterControl("http://s.test/v1", client)
        assert await ctl.load("x", "/p") is False


def _spec():
    return FleetSpec(
        base_models=["qwen-1.5b"],
        adapters=[
            AdapterSpec(name="legal-v3", base="qwen-1.5b", pinned=True, path="/adapters/legal-v3"),
            AdapterSpec(name="fin-v2", base="qwen-1.5b", pinned=False, path="/adapters/fin-v2"),
        ],
        slot_capacity=2,  # 1 base + 1 adapter slot → fin-v2 pages out
    )


@respx.mock
async def test_enactor_loads_resident_unloads_paged():
    respx.post("http://s.test/v1/load_lora_adapter").mock(
        return_value=httpx.Response(200, text="Success"))
    respx.post("http://s.test/v1/unload_lora_adapter").mock(
        return_value=httpx.Response(200, text="Success"))
    spec = _spec()
    plan = FleetManager().plan(spec)
    async with httpx.AsyncClient() as client:
        enactor = FleetEnactor({"qwen-1.5b": VLLMAdapterControl("http://s.test/v1", client)})
        report = await enactor.enact(plan, spec)
    assert report.loaded == ["legal-v3"]
    assert report.unloaded == ["fin-v2"]
    assert report.skipped == []
    assert report.errors == []


@respx.mock
async def test_enactor_skips_unmapped_base_and_pathless_adapter():
    respx.post("http://s.test/v1/unload_lora_adapter").mock(
        return_value=httpx.Response(200, text="Success"))
    spec = FleetSpec(
        base_models=["qwen-1.5b", "unserved-base"],
        adapters=[
            AdapterSpec(name="a1", base="unserved-base", pinned=True, path="/p"),
            AdapterSpec(name="a2", base="qwen-1.5b", pinned=True),  # no path
        ],
        slot_capacity=4,
    )
    plan = FleetManager().plan(spec)
    async with httpx.AsyncClient() as client:
        enactor = FleetEnactor({"qwen-1.5b": VLLMAdapterControl("http://s.test/v1", client)})
        report = await enactor.enact(plan, spec)
    assert report.loaded == []
    assert sorted(report.skipped) == ["a1", "a2"]  # unmapped base / missing path
    assert report.errors == []


def test_apply_fleet_enact_flag(client):
    # API: enact=true returns an enactment report alongside the plan
    with respx.mock:
        respx.post("http://big.test/v1/load_lora_adapter").mock(
            return_value=httpx.Response(200, text="Success"))
        respx.post("http://small.test/v1/load_lora_adapter").mock(
            return_value=httpx.Response(200, text="Success"))
        r = client.post("/photon/v1/fleet?enact=true", json={
            "base_models": ["qwen-72b"],
            "adapters": [{"name": "legal-v3", "base": "qwen-72b", "pinned": True,
                          "path": "/adapters/legal-v3"}],
            "slot_capacity": 3,
        })
    assert r.status_code == 200
    body = r.json()
    assert body["plan"]["resident_adapters"] == ["legal-v3"]
    assert body["enactment"]["loaded"] == ["legal-v3"]
