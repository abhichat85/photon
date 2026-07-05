# tests/test_static_router.py
import pytest

from photon.config import PhotonConfig
from photon.router.static import AUTO_MODEL, StaticRouter, UnknownModelError
from tests.test_config import VALID


@pytest.fixture
def router() -> StaticRouter:
    return StaticRouter(PhotonConfig.model_validate(VALID))


def test_auto_resolves_to_default_backend(router):
    decision = router.resolve(AUTO_MODEL)
    assert decision.backend.name == "big"
    assert decision.reason == "default"


def test_alias_resolves(router):
    decision = router.resolve("praxiom-chat")
    assert decision.backend.name == "big"
    assert decision.reason == "alias"


def test_backend_name_resolves_directly(router):
    decision = router.resolve("small")
    assert decision.backend.name == "small"
    assert decision.reason == "direct"


def test_backend_model_id_resolves_directly(router):
    decision = router.resolve("qwen-7b")
    assert decision.backend.name == "small"
    assert decision.reason == "direct"


def test_unknown_model_raises(router):
    with pytest.raises(UnknownModelError):
        router.resolve("gpt-nonexistent")


# append to tests/test_static_router.py
from photon.router.static import StaticRouter as _SR  # noqa: F401 (already imported above)


class FixedRng:
    def __init__(self, value: float):
        self._value = value

    def random(self) -> float:
        return self._value


def make_canary_router(draw: float, weight: float = 0.25) -> StaticRouter:
    cfg = {
        **VALID,
        "routing": {**VALID["routing"], "canary": {"backend": "small", "weight": weight}},
    }
    return StaticRouter(PhotonConfig.model_validate(cfg), rng=FixedRng(draw))


def test_canary_hits_when_draw_below_weight():
    decision = make_canary_router(draw=0.1).resolve(AUTO_MODEL)
    assert decision.backend.name == "small"
    assert decision.reason == "canary"


def test_canary_misses_when_draw_above_weight():
    decision = make_canary_router(draw=0.9).resolve(AUTO_MODEL)
    assert decision.backend.name == "big"
    assert decision.reason == "default"


def test_canary_never_applies_to_alias_or_direct():
    router = make_canary_router(draw=0.0)  # would always fire on auto
    assert router.resolve("praxiom-chat").reason == "alias"
    assert router.resolve("small").reason == "direct"


def test_allow_canary_false_forces_default():
    router = make_canary_router(draw=0.0)  # would always fire on auto
    d = router.resolve(AUTO_MODEL, allow_canary=False)
    assert d.backend.name == "big"
    assert d.reason == "default"
