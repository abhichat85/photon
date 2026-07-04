# tests/test_api_streaming.py
import httpx
import pytest
import respx

# Simulated vLLM SSE stream: two content chunks, a usage chunk, then [DONE]
SSE_BODY = (
    b'data: {"id":"c1","object":"chat.completion.chunk","model":"qwen-72b",'
    b'"choices":[{"index":0,"delta":{"content":"he"},"finish_reason":null}]}\n\n'
    b'data: {"id":"c1","object":"chat.completion.chunk","model":"qwen-72b",'
    b'"choices":[{"index":0,"delta":{"content":"llo"},"finish_reason":"stop"}]}\n\n'
    b'data: {"id":"c1","object":"chat.completion.chunk","model":"qwen-72b",'
    b'"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n'
    b"data: [DONE]\n\n"
)


@respx.mock
def test_streaming_relays_sse_and_records_usage(client):
    respx.post("http://big.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=SSE_BODY, headers={"content-type": "text/event-stream"}
        )
    )
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "photon-auto",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"X-Photon-Tenant": "praxiom"},
    ) as r:
        assert r.status_code == 200
        body = b"".join(r.iter_bytes())
    assert b'"content":"he"' in body
    assert b"data: [DONE]" in body

    decisions = client.get(
        "/photon/v1/routing/decisions", params={"tenant": "praxiom"}
    ).json()["decisions"]
    assert decisions[0]["status"] == "ok"
    assert decisions[0]["prompt_tokens"] == 10
    assert decisions[0]["completion_tokens"] == 5
    assert decisions[0]["cost_usd"] is not None


@respx.mock
def test_streaming_upstream_error_is_502(client):
    respx.post("http://big.test/v1/chat/completions").mock(
        return_value=httpx.Response(500, text="boom")
    )
    r = client.post(
        "/v1/chat/completions",
        json={"model": "photon-auto", "stream": True, "messages": []},
        headers={"X-Photon-Tenant": "praxiom"},
    )
    assert r.status_code == 502


@respx.mock
def test_midstream_failure_recorded_as_error(client):
    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield (
                b'data: {"id":"c1","object":"chat.completion.chunk","model":"qwen-72b",'
                b'"choices":[{"index":0,"delta":{"content":"he"},"finish_reason":null}]}\n\n'
            )
            raise httpx.ReadError("connection lost")

    respx.post("http://big.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, stream=BrokenStream(), headers={"content-type": "text/event-stream"}
        )
    )
    with pytest.raises(Exception):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "photon-auto", "stream": True,
                  "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Photon-Tenant": "praxiom"},
        ) as r:
            b"".join(r.iter_bytes())

    decisions = client.get(
        "/photon/v1/routing/decisions", params={"tenant": "praxiom"}
    ).json()["decisions"]
    assert decisions[0]["status"] == "error"


@respx.mock
def test_streaming_logs_shadow_candidate(client):
    respx.post("http://big.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=SSE_BODY, headers={"content-type": "text/event-stream"}
        )
    )
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "photon-auto", "stream": True,
              "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Photon-Tenant": "shadowstream"},
    ) as r:
        b"".join(r.iter_bytes())
    d = client.get(
        "/photon/v1/routing/decisions", params={"tenant": "shadowstream"}
    ).json()["decisions"][0]
    assert d["shadow_backend"] == "small"
    assert d["shadow_est_cost_usd"] == pytest.approx((10 * 0.08 + 5 * 0.08) / 1_000_000)


@respx.mock
def test_client_stream_options_merged_not_clobbered(client):
    import json as _json

    route = respx.post("http://big.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=SSE_BODY, headers={"content-type": "text/event-stream"}
        )
    )
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "photon-auto", "stream": True,
              "stream_options": {"continuous_usage_stats": False},
              "messages": [{"role": "user", "content": "hi"}]},
    ) as r:
        b"".join(r.iter_bytes())
    sent = _json.loads(route.calls.last.request.content)
    assert sent["stream_options"]["include_usage"] is True
    assert sent["stream_options"]["continuous_usage_stats"] is False
