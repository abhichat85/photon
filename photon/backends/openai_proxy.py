import time

import httpx

from photon.config import BackendConfig


class BackendError(Exception):
    def __init__(self, backend: str, detail: str):
        self.backend = backend
        self.detail = detail
        super().__init__(f"{backend}: {detail}")


class OpenAIProxy:
    """Forwards OpenAI-format chat payloads to a backend's OpenAI-compatible
    server (vLLM's `vllm serve` exposes exactly this). Replaces the client's
    model field with the backend's concrete model id."""

    def __init__(self, client: httpx.AsyncClient):
        self._client = client

    async def chat_completions(
        self, backend: BackendConfig, payload: dict
    ) -> tuple[dict, float]:
        body = {**payload, "model": backend.model}
        started = time.perf_counter()
        try:
            resp = await self._client.post(
                f"{backend.base_url}/chat/completions", json=body
            )
        except httpx.HTTPError as exc:
            raise BackendError(backend.name, str(exc)) from exc
        latency_ms = (time.perf_counter() - started) * 1000
        if resp.status_code >= 400:
            raise BackendError(
                backend.name, f"upstream status {resp.status_code}: {resp.text[:200]}"
            )
        return resp.json(), latency_ms
