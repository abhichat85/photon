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
