"""Cheap, deterministic request features for the policy model (D-R1: no
embedding model in C0). The feature vector order is FIXED — the policy model is
trained and scored against this exact order."""
from __future__ import annotations

from pydantic import BaseModel


class RequestFeatures(BaseModel):
    prompt_chars: int
    message_count: int
    has_tool_use: bool = False
    tenant: str = "default"
    route_hint: str = "auto"
    pipeline_stage: str | None = None
    tenant_recent_accept_rate: float = 0.0

    def to_vector(self) -> list[float]:
        """FIXED order — do not reorder without retraining every policy model."""
        return [
            float(self.prompt_chars),
            float(self.message_count),
            float(self.has_tool_use),
            float(self.tenant_recent_accept_rate),
        ]


def _content_chars(messages: list[dict]) -> int:
    return sum(len(m.get("content")) for m in messages if isinstance(m.get("content"), str))


def extract_features(
    messages: list[dict],
    tenant: str = "default",
    route_hint: str = "auto",
    pipeline_stage: str | None = None,
    tenant_recent_accept_rate: float = 0.0,
    tools: list | None = None,
) -> RequestFeatures:
    return RequestFeatures(
        prompt_chars=_content_chars(messages),
        message_count=len(messages),
        has_tool_use=bool(tools),
        tenant=tenant,
        route_hint=route_hint,
        pipeline_stage=pipeline_stage,
        tenant_recent_accept_rate=tenant_recent_accept_rate,
    )
