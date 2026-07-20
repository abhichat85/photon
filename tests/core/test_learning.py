# tests/core/test_learning.py
from photon.core.features import RequestFeatures
from photon.core.learning import LabeledRow, ReplayHarness


def _rows():
    # synthetic replay: short prompts were acceptable-cheap, long were not
    rows = []
    for c in (5, 8, 10, 12, 15):
        rows.append(LabeledRow(features=RequestFeatures(prompt_chars=c, message_count=1),
                               cheap_acceptable=True, cheap_cost=0.001, big_cost=0.010))
    for c in (900, 1100, 1400, 1800, 2200):
        rows.append(LabeledRow(features=RequestFeatures(prompt_chars=c, message_count=1),
                               cheap_acceptable=False, cheap_cost=0.001, big_cost=0.010))
    return rows


def test_harness_trains_and_reports_regret_and_savings():
    report = ReplayHarness(threshold=0.6).run(_rows())
    assert report.trained_on == 10
    # a separable pattern → the trained router should beat always-big on cost
    assert report.routed_cheap_share > 0.3
    assert report.regret < report.always_big_regret
    assert 0.0 <= report.est_savings_vs_always_big <= 1.0


def test_harness_handles_empty():
    report = ReplayHarness(threshold=0.6).run([])
    assert report.trained_on == 0
    assert report.routed_cheap_share == 0.0


def test_report_tracks_oracle_agreement():
    report = ReplayHarness(threshold=0.6).run(_rows())
    assert report.oracle_matches > 0
    assert report.oracle_match_share == report.oracle_matches / report.trained_on
    assert report.oracle_match_share > 0.8  # separable pattern → high agreement


def test_register_policy_version_flows_through_gated_registry(tmp_path):
    # the ROUTER is versioned + promotion-gated like any other model
    from photon.core.learning import register_policy_version
    from photon.core.policy import PolicyModel
    from photon.registry.store import RegistryStore

    rows = _rows()
    report = ReplayHarness(threshold=0.6).run(rows)
    policy = PolicyModel()
    policy.fit([r.features for r in rows], [int(r.cheap_acceptable) for r in rows])

    registry = RegistryStore(tmp_path / "r.db")
    mv = register_policy_version(registry, policy, report, tmp_path / "policy.joblib")
    assert mv.name == "router-policy"
    assert mv.status == "draft"
    assert (tmp_path / "policy.joblib").exists()

    # promotion passes the same eval gate adapters face (oracle agreement >= bar)
    registry.promote("router-policy", mv.version, min_pass_rate=0.8)
    assert registry.production("router-policy").version == mv.version
    # and the persisted artifact round-trips into a usable policy
    loaded = PolicyModel.load(mv.adapter_path)
    assert 0.0 <= loaded.predict_acceptable(rows[0].features) <= 1.0


def test_harness_handles_single_class_window():
    # a telemetry window where EVERY request was cheap-acceptable — sklearn
    # cannot fit a classifier on one class. The harness must not crash; with no
    # trained policy it fails closed (routes nothing cheap).
    rows = [
        LabeledRow(features=RequestFeatures(prompt_chars=c, message_count=1),
                   cheap_acceptable=True, cheap_cost=0.001, big_cost=0.010)
        for c in (5, 8, 10, 12)
    ]
    report = ReplayHarness(threshold=0.6).run(rows)  # must not raise
    assert report.trained_on == 4
    assert report.routed_cheap_share == 0.0  # fail closed on an untrainable window
