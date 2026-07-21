# photon/core/enactment.py
"""Fleet ENACTMENT — the step from advisory PlacementPlans to real ones.

vLLM exposes runtime LoRA management (POST /v1/load_lora_adapter and
/v1/unload_lora_adapter) when the server is started with
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True. VLLMAdapterControl wraps that API;
FleetEnactor diffs a PlacementPlan against it: resident adapters are loaded,
paged adapters are unloaded. Adapters whose base isn't served by any known
backend, or that carry no filesystem path, are SKIPPED (reported, not errored) —
a plan may legitimately reference bases this pod doesn't serve.

Solo-mode scope (DECISIONS.md D13): this is vLLM-native adapter management,
benchmark-gated — fork/kernel work only if measured density numbers demand it."""
from __future__ import annotations

import httpx
from pydantic import BaseModel, Field

from photon.core.fleet import FleetSpec, PlacementPlan


class VLLMAdapterControl:
    """Thin client for one vLLM server's runtime-LoRA endpoints."""

    def __init__(self, base_url: str, client: httpx.AsyncClient):
        self._base_url = base_url
        self._client = client

    async def load(self, lora_name: str, lora_path: str) -> bool:
        try:
            resp = await self._client.post(
                f"{self._base_url}/load_lora_adapter",
                json={"lora_name": lora_name, "lora_path": lora_path},
            )
            return resp.status_code < 400
        except httpx.HTTPError:
            return False

    async def unload(self, lora_name: str) -> bool:
        try:
            resp = await self._client.post(
                f"{self._base_url}/unload_lora_adapter",
                json={"lora_name": lora_name},
            )
            return resp.status_code < 400
        except httpx.HTTPError:
            return False

    async def list_served(self) -> list[str]:
        """Model ids the server currently serves (bases + loaded LoRA modules)."""
        resp = await self._client.get(f"{self._base_url}/models")
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]


class EnactmentReport(BaseModel):
    loaded: list[str] = Field(default_factory=list)
    unloaded: list[str] = Field(default_factory=list)
    already_resident: list[str] = Field(default_factory=list)  # idempotent no-op
    already_absent: list[str] = Field(default_factory=list)     # idempotent no-op
    skipped: list[str] = Field(default_factory=list)  # unmapped base or no path
    errors: list[str] = Field(default_factory=list)   # control call failed
    warnings: list[str] = Field(default_factory=list)  # loud surface for skips

    @property
    def clean(self) -> bool:
        return not self.errors and not self.skipped


class FleetEnactor:
    def __init__(self, controls: dict[str, VLLMAdapterControl]):
        # keyed by base-model identifier as used in FleetSpec.adapters[].base
        self._controls = controls

    async def _served(self, control: VLLMAdapterControl, cache: dict) -> set[str]:
        key = id(control)
        if key not in cache:
            try:
                cache[key] = set(await control.list_served())
            except Exception:  # noqa: BLE001 — a failed probe → assume nothing served
                cache[key] = set()
        return cache[key]

    async def enact(self, plan: PlacementPlan, spec: FleetSpec) -> EnactmentReport:
        """Idempotent: diffs the plan against what each server ACTUALLY serves
        (list_served), so re-enacting a stable plan is a no-op rather than a
        stream of false 'already loaded' errors."""
        report = EnactmentReport()
        adapters = {a.name: a for a in spec.adapters}
        served_cache: dict = {}

        for name in plan.resident_adapters:
            adapter = adapters[name]
            control = self._controls.get(adapter.base)
            if control is None or adapter.path is None:
                report.skipped.append(name)
                report.warnings.append(
                    f"resident adapter {name!r}: "
                    + ("no path" if control is not None else f"base {adapter.base!r} not served here")
                )
                continue
            if name in await self._served(control, served_cache):
                report.already_resident.append(name)
                continue
            if await control.load(name, adapter.path):
                report.loaded.append(name)
            else:
                report.errors.append(name)

        for name in plan.paged_adapters:
            adapter = adapters[name]
            control = self._controls.get(adapter.base)
            if control is None:
                report.skipped.append(name)
                report.warnings.append(f"paged adapter {name!r}: base {adapter.base!r} not served here")
                continue
            if name not in await self._served(control, served_cache):
                report.already_absent.append(name)
                continue
            if await control.unload(name):
                report.unloaded.append(name)
            else:
                report.errors.append(name)

        return report
