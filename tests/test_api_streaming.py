# tests/test_api_streaming.py
import httpx
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
