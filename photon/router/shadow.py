# photon/router/shadow.py
from photon.config import PhotonConfig


class ShadowPolicy:
    """Phase 0 shadow study: a deliberately crude heuristic (prompt length)
    that nominates a cheaper candidate backend. Its ONLY output is telemetry —
    there is no code path from here to backend selection. The offline study
    evaluates candidate quality; the heuristic's job is coverage, not accuracy."""

    def __init__(self, config: PhotonConfig):
        self._config = config

    def candidate(self, routed_backend_name: str, messages: list[dict]) -> str | None:
        shadow = self._config.routing.shadow
        if not shadow.enabled or shadow.candidate_backend is None:
            return None
        if routed_backend_name == shadow.candidate_backend:
            return None
        chars = sum(
            len(m.get("content"))
            for m in messages
            if isinstance(m.get("content"), str)
        )
        if chars <= shadow.max_prompt_chars:
            return shadow.candidate_backend
        return None
