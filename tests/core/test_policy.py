from photon.core.features import RequestFeatures
from photon.core.policy import PolicyModel


def _feat(chars: int, accept_rate: float = 0.5) -> RequestFeatures:
    return RequestFeatures(prompt_chars=chars, message_count=1, tenant_recent_accept_rate=accept_rate)


def test_untrained_policy_returns_neutral():
    p = PolicyModel()
    # before fit, predict is a safe 0.0 (never route to cheap) — fail closed
    assert p.predict_acceptable(_feat(10)) == 0.0


def test_policy_learns_separable_pattern():
    # short prompts acceptable to the cheap model (label 1), long prompts not (0)
    X = [_feat(c) for c in (5, 8, 10, 12)] + [_feat(c) for c in (900, 1200, 1500, 2000)]
    y = [1, 1, 1, 1, 0, 0, 0, 0]
    p = PolicyModel()
    p.fit(X, y)
    assert p.predict_acceptable(_feat(9)) > 0.6   # short → likely acceptable
    assert p.predict_acceptable(_feat(1400)) < 0.4  # long → likely not


def test_policy_roundtrip_save_load(tmp_path):
    X = [_feat(c) for c in (5, 10)] + [_feat(c) for c in (1000, 2000)]
    y = [1, 1, 0, 0]
    p = PolicyModel()
    p.fit(X, y)
    path = tmp_path / "policy.joblib"
    p.save(path)
    loaded = PolicyModel.load(path)
    assert abs(loaded.predict_acceptable(_feat(8)) - p.predict_acceptable(_feat(8))) < 1e-9
