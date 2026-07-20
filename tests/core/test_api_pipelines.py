# tests/core/test_api_pipelines.py
from photon.core.serving import MockServingBackend


def _resp(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


PIPELINE = {
    "id": "praxiom-mini",
    "stages": [
        {"name": "parse", "target": {"model_id": "parser"}},
        {"name": "generate", "target": {"model_id": "generator"}},
    ],
}


def test_register_list_execute_roundtrip(client):
    r = client.post("/photon/v1/pipelines", json=PIPELINE)
    assert r.status_code == 200
    assert client.get("/photon/v1/pipelines").json() == {"pipelines": ["praxiom-mini"]}

    # execute against an injected mock serving backend
    client.app.state.serving_backend = MockServingBackend(
        canned={"parser": _resp("PARSED"), "generator": _resp("FINAL")}
    )
    result = client.post(
        "/photon/v1/pipelines/praxiom-mini",
        json={"messages": [{"role": "user", "content": "goal"}]},
    ).json()
    assert result["completed"] is True
    assert [o["content"] for o in result["stage_outputs"]] == ["PARSED", "FINAL"]


def test_execute_unknown_pipeline_is_404(client):
    r = client.post("/photon/v1/pipelines/nope", json={"messages": []})
    assert r.status_code == 404


def test_register_invalid_spec_is_422(client):
    r = client.post("/photon/v1/pipelines", json={"id": "empty", "stages": []})
    assert r.status_code == 422


def test_default_serving_backend_is_wired(client):
    # lifespan constructs a VLLMServingBackend from the fleet config
    from photon.core.serving import VLLMServingBackend

    assert isinstance(client.app.state.serving_backend, VLLMServingBackend)
