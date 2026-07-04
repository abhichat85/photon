# tests/test_shadow_policy.py
import pytest

from photon.config import PhotonConfig
from photon.router.shadow import ShadowPolicy
from tests.test_config import VALID

SHORT = [{"role": "user", "content": "hello"}]
LONG = [{"role": "user", "content": "x" * 5000}]


@pytest.fixture
def policy() -> ShadowPolicy:
    return ShadowPolicy(PhotonConfig.model_validate(VALID))


def test_short_prompt_on_big_backend_gets_candidate(policy):
    assert policy.candidate("big", SHORT) == "small"


def test_long_prompt_gets_no_candidate(policy):
    assert policy.candidate("big", LONG) is None


def test_already_on_candidate_backend_gets_none(policy):
    assert policy.candidate("small", SHORT) is None


def test_disabled_shadow_gets_none():
    cfg = {**VALID, "routing": {**VALID["routing"], "shadow": {"enabled": False}}}
    policy = ShadowPolicy(PhotonConfig.model_validate(cfg))
    assert policy.candidate("big", SHORT) is None


def test_non_string_content_is_ignored_not_crashed(policy):
    # OpenAI content can be a list of parts (multimodal); we skip those chars
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert policy.candidate("big", messages) == "small"
