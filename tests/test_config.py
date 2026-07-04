# tests/test_config.py
import pytest
from pydantic import ValidationError

from photon.config import PhotonConfig

VALID = {
    "backends": [
        {
            "name": "big",
            "base_url": "http://big.test/v1",
            "model": "qwen-72b",
            "pricing": {"input_per_1m": 0.9, "output_per_1m": 0.9},
        },
        {
            "name": "small",
            "base_url": "http://small.test/v1",
            "model": "qwen-7b",
            "pricing": {"input_per_1m": 0.08, "output_per_1m": 0.08},
        },
    ],
    "routing": {
        "default_backend": "big",
        "aliases": {"praxiom-chat": "big"},
        "shadow": {"enabled": True, "candidate_backend": "small", "max_prompt_chars": 2000},
    },
}


def test_valid_config_parses():
    cfg = PhotonConfig.model_validate(VALID)
    assert cfg.backend("big").model == "qwen-72b"
    assert cfg.routing.default_backend == "big"


def test_unknown_default_backend_rejected():
    bad = {**VALID, "routing": {**VALID["routing"], "default_backend": "nope"}}
    with pytest.raises(ValidationError):
        PhotonConfig.model_validate(bad)


def test_alias_to_unknown_backend_rejected():
    bad = {**VALID, "routing": {**VALID["routing"], "aliases": {"x": "nope"}}}
    with pytest.raises(ValidationError):
        PhotonConfig.model_validate(bad)


def test_shadow_candidate_must_exist_when_enabled():
    bad = {
        **VALID,
        "routing": {**VALID["routing"], "shadow": {"enabled": True, "candidate_backend": "nope"}},
    }
    with pytest.raises(ValidationError):
        PhotonConfig.model_validate(bad)


def test_yaml_roundtrip(tmp_path):
    import yaml

    p = tmp_path / "fleet.yaml"
    p.write_text(yaml.safe_dump(VALID))
    cfg = PhotonConfig.from_yaml(p)
    assert {b.name for b in cfg.backends} == {"big", "small"}


def test_backend_lookup_missing_raises():
    cfg = PhotonConfig.model_validate(VALID)
    with pytest.raises(KeyError):
        cfg.backend("missing")


# append to tests/test_config.py
def test_canary_weight_must_be_in_unit_interval():
    bad = {
        **VALID,
        "routing": {**VALID["routing"], "canary": {"backend": "small", "weight": 1.5}},
    }
    with pytest.raises(ValidationError):
        PhotonConfig.model_validate(bad)
