# tests/core/test_shadow_store.py
import pytest

from photon.core.contract import RouteTarget
from photon.core.router import ShadowDecision
from photon.core.shadow_store import ShadowDecisionStore


def _decision(rid: str, actual: str = "big", would: str = "cheap", reason: str = "policy-cheap"):
    return ShadowDecision(
        request_id=rid,
        actual_backend=actual,
        would_route=RouteTarget(model_id=would, adapter_id=None),
        reason=reason,
    )


@pytest.fixture
def store(tmp_path) -> ShadowDecisionStore:
    return ShadowDecisionStore(tmp_path / "shadow.db")


def test_insert_and_read_back(store):
    store.insert(_decision("r1"))
    rows = store.recent(limit=10)
    assert len(rows) == 1
    assert rows[0]["request_id"] == "r1"
    assert rows[0]["actual_backend"] == "big"
    assert rows[0]["would_model"] == "cheap"
    assert rows[0]["reason"] == "policy-cheap"


def test_recent_orders_newest_first_and_limits(store):
    store.insert(_decision("old"))
    store.insert(_decision("new"))
    rows = store.recent(limit=1)
    assert rows[0]["request_id"] == "new"


def test_summary_agreement_and_distribution(store):
    store.insert(_decision("a", actual="big", would="big"))     # agree
    store.insert(_decision("b", actual="big", would="cheap"))   # disagree
    store.insert(_decision("c", actual="big", would="cheap"))   # disagree
    s = store.summary()
    assert s["total"] == 3
    assert s["agreement_share"] == pytest.approx(1 / 3)
    assert s["would_route_counts"] == {"big": 1, "cheap": 2}


def test_summary_empty(store):
    s = store.summary()
    assert s == {"total": 0, "agreement_share": 0.0, "would_route_counts": {}}


def test_insert_is_a_valid_shadow_router_sink(store):
    # the store's insert satisfies the ShadowRouter sink signature end-to-end
    from photon.core.policy import PolicyModel
    from photon.core.router import LearnedRouter, ShadowRouter

    class Stub(PolicyModel):
        def predict_acceptable(self, features):
            return 0.99

    learned = LearnedRouter(Stub(), 0.6, RouteTarget(model_id="cheap"), RouteTarget(model_id="big"))
    shadow = ShadowRouter(learned, sink=store.insert)
    from photon.core.features import RequestFeatures

    served = shadow.observe(
        actual_backend_name="big",
        features=RequestFeatures(prompt_chars=5, message_count=1),
        request_id="rX",
    )
    assert served == "big"
    assert store.recent(1)[0]["request_id"] == "rX"
