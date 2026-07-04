# photon/config.py
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelPricing(BaseModel):
    """USD per 1M tokens. For our own GPUs these are amortized estimates
    derived from instance cost / measured throughput; for external
    baselines they are the provider's listed prices."""

    input_per_1m: float
    output_per_1m: float


class BackendConfig(BaseModel):
    name: str
    base_url: str  # OpenAI-compatible root, e.g. http://vllm-big:8000/v1
    model: str  # model id the backend expects in the payload
    pricing: ModelPricing


class ShadowPolicyConfig(BaseModel):
    enabled: bool = False
    candidate_backend: str | None = None
    max_prompt_chars: int = 2000


class CanaryConfig(BaseModel):
    backend: str
    weight: float  # fraction of photon-auto traffic in (0, 1]


class RoutingConfig(BaseModel):
    default_backend: str
    aliases: dict[str, str] = Field(default_factory=dict)
    shadow: ShadowPolicyConfig = Field(default_factory=ShadowPolicyConfig)
    canary: CanaryConfig | None = None


class PhotonConfig(BaseModel):
    backends: list[BackendConfig]
    routing: RoutingConfig

    @model_validator(mode="after")
    def _check_references(self) -> "PhotonConfig":
        names = {b.name for b in self.backends}
        if self.routing.default_backend not in names:
            raise ValueError(
                f"default_backend {self.routing.default_backend!r} is not a configured backend"
            )
        for alias, target in self.routing.aliases.items():
            if target not in names:
                raise ValueError(f"alias {alias!r} points at unknown backend {target!r}")
        shadow = self.routing.shadow
        if shadow.enabled and shadow.candidate_backend not in names:
            raise ValueError(
                f"shadow candidate_backend {shadow.candidate_backend!r} is not a configured backend"
            )
        if self.routing.canary is not None:
            if self.routing.canary.backend not in names:
                raise ValueError(
                    f"canary backend {self.routing.canary.backend!r} is not a configured backend"
                )
            if not (0 < self.routing.canary.weight <= 1):
                raise ValueError("canary weight must be in (0, 1]")
        return self

    def backend(self, name: str) -> BackendConfig:
        for b in self.backends:
            if b.name == name:
                return b
        raise KeyError(name)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PhotonConfig":
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))
