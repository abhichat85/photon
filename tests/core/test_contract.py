# tests/core/test_contract.py
from photon.core.contract import RouteTarget


def test_route_target_holds_model_adapter_params():
    t = RouteTarget(model_id="qwen-7b", adapter_id="praxiom-intent-v3", decode_params={"temperature": 0.2})
    assert t.model_id == "qwen-7b"
    assert t.adapter_id == "praxiom-intent-v3"
    assert t.decode_params["temperature"] == 0.2


def test_route_target_adapter_optional():
    t = RouteTarget(model_id="qwen-72b")
    assert t.adapter_id is None
    assert t.decode_params == {}
