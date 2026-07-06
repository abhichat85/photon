"""ServingBackend implementations for C0. MockServingBackend for tests;
VLLMServingBackend is a thin adapter-aware wrapper over the existing OpenAI proxy
pattern (works today with vLLM --enable-lora, where a LoRA module is served under
its own model name). The Tier-2 dense multi-adapter engine implements the same
interface later without touching the Router."""
from __future__ import annotations

import time

import httpx
from pydantic import BaseModel

from photon.config import BackendConfig
from photon.core.contract import RouteTarget


class _Call(BaseModel):
    model_id: str
    adapter_id: str | None


class MockServingBackend:
    def __init__(self, canned: dict[str, dict]):
        self._canned = canned
        self.calls: list[_Call] = []

    async def generate(self, target: RouteTarget, payload: dict) -> tuple[dict, float]:
        self.calls.append(_Call(model_id=target.model_id, adapter_id=target.adapter_id))
        return self._canned[target.model_id], 0.0


class VLLMServingBackend:
    def __init__(self, backends: dict[str, BackendConfig], client: httpx.AsyncClient):
        self._backends = backends
        self._client = client

    async def generate(self, target: RouteTarget, payload: dict) -> tuple[dict, float]:
        backend = self._backends[target.model_id]
        # when an adapter is selected, vLLM serves it under the adapter name
        served_model = target.adapter_id or backend.model
        body = {**payload, "model": served_model}
        started = time.perf_counter()
        resp = await self._client.post(f"{backend.base_url}/chat/completions", json=body)
        latency_ms = (time.perf_counter() - started) * 1000
        resp.raise_for_status()
        return resp.json(), latency_ms
