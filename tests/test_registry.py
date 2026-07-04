# tests/test_registry.py
import pytest

from photon.registry.store import RegistryStore


@pytest.fixture
def registry(tmp_path) -> RegistryStore:
    return RegistryStore(tmp_path / "registry.db")


def test_versions_increment_per_name(registry):
    v1 = registry.register("praxiom-intent", "Qwen/Qwen2.5-1.5B-Instruct", "/adapters/v1")
    v2 = registry.register("praxiom-intent", "Qwen/Qwen2.5-1.5B-Instruct", "/adapters/v2")
    other = registry.register("praxiom-critic", "deberta-v3", None)
    assert (v1.version, v2.version, other.version) == (1, 2, 1)
    assert v1.status == "draft"


def test_promote_marks_production_and_demotes_previous(registry):
    v1 = registry.register("m", "base", "/a/1")
    v2 = registry.register("m", "base", "/a/2")
    registry.promote("m", v1.version)
    registry.promote("m", v2.version)
    assert registry.production("m").version == 2
    history = {mv.version: mv.status for mv in registry.history("m")}
    assert history == {1: "retired", 2: "production"}


def test_production_none_when_never_promoted(registry):
    registry.register("m", "base", "/a/1")
    assert registry.production("m") is None


def test_promote_unknown_version_raises(registry):
    registry.register("m", "base", "/a/1")
    with pytest.raises(KeyError):
        registry.promote("m", 99)


def test_attach_eval_report(registry):
    v1 = registry.register("m", "base", "/a/1")
    registry.attach_eval("m", v1.version, '{"passed": 3, "total": 3}')
    assert registry.get("m", v1.version).eval_report == '{"passed": 3, "total": 3}'
