# tests/core/test_api_fleet.py
def test_apply_fleet_and_read_residency(client):
    spec = {
        "base_models": ["qwen-1.5b", "qwen-14b"],
        "adapters": [
            {"name": "legal-v3", "base": "qwen-1.5b", "pinned": True},
            {"name": "fin-v2", "base": "qwen-1.5b", "pinned": False},
        ],
        "slot_capacity": 3,
    }
    r = client.post("/photon/v1/fleet", json=spec)
    assert r.status_code == 200
    plan = r.json()
    assert "legal-v3" in plan["resident_adapters"]  # pinned resident under pressure

    # dynamic status now reflects the applied plan's residency
    status = client.get("/photon/v1/fleet/status").json()
    assert status["residency"]["resident_adapters"] == plan["resident_adapters"]
    assert status["residency"]["paged_adapters"] == plan["paged_adapters"]


def test_apply_invalid_fleet_is_422(client):
    bad = {"base_models": ["qwen-1.5b"],
           "adapters": [{"name": "x", "base": "nope", "pinned": False}],
           "slot_capacity": 3}
    r = client.post("/photon/v1/fleet", json=bad)
    assert r.status_code == 422


def test_status_residency_null_before_any_apply(client):
    # a fresh app that hasn't had a fleet applied reports null residency, not a crash
    status = client.get("/photon/v1/fleet/status").json()
    assert status["residency"] is None
