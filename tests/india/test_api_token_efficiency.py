# tests/india/test_api_token_efficiency.py
"""The live measurement path: real requests accumulate tokenizer-efficiency
ratios per (backend, script), and the admin readout exposes the Indic penalty."""
import httpx
import respx


def _resp(prompt_tokens: int) -> dict:
    return {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 5,
                  "total_tokens": prompt_tokens + 5},
    }


@respx.mock
def test_hindi_request_records_devanagari_efficiency(client):
    hindi = "मुझे इस उत्पाद के बारे में विस्तार से जानकारी चाहिए"  # ~50 chars
    respx.post("http://big.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_resp(prompt_tokens=50)))
    client.post("/v1/chat/completions",
                json={"model": "photon-auto", "messages": [{"role": "user", "content": hindi}]})
    rows = client.get("/photon/v1/india/token-efficiency", params={"backend": "big"}).json()
    scripts = {r["script"]: r for r in rows["backends"]["big"]}
    assert "devanagari" in scripts
    assert scripts["devanagari"]["chars_per_token"] == len(hindi) / 50
    assert scripts["devanagari"]["samples"] == 1


@respx.mock
def test_indic_penalty_surfaces_across_scripts(client):
    # English: 200 chars / 50 tokens = 4.0 c/t. Hindi: 50 chars / 50 tokens = 1.0.
    # → devanagari penalty vs latin = 4.0x (the structural over-charge)
    english = "a" * 200
    hindi = "क" * 50
    respx.post("http://big.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_resp(prompt_tokens=50)))
    for text in (english, hindi):
        client.post("/v1/chat/completions",
                    json={"model": "photon-auto", "messages": [{"role": "user", "content": text}]})
    rows = client.get("/photon/v1/india/token-efficiency").json()["backends"]["big"]
    scripts = {r["script"]: r for r in rows}
    assert scripts["latin"]["chars_per_token"] == 4.0
    assert scripts["devanagari"]["chars_per_token"] == 1.0
    assert scripts["devanagari"]["indic_penalty_vs_latin"] == 4.0


def test_readout_lists_all_backends_when_unfiltered(client):
    body = client.get("/photon/v1/india/token-efficiency").json()
    assert set(body["backends"].keys()) == {"big", "small"}
