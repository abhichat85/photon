# tests/core/test_features.py
from photon.core.features import RequestFeatures, extract_features


def test_extract_basic_features():
    f = extract_features(
        messages=[{"role": "user", "content": "hello world"}],
        tenant="praxiom",
        route_hint="auto",
        pipeline_stage="parse",
        tenant_recent_accept_rate=0.75,
    )
    assert isinstance(f, RequestFeatures)
    assert f.prompt_chars == len("hello world")
    assert f.message_count == 1
    assert f.tenant == "praxiom"
    assert f.pipeline_stage == "parse"
    assert f.tenant_recent_accept_rate == 0.75


def test_extract_handles_missing_and_multimodal_content():
    f = extract_features(
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}, {"role": "user", "content": "yo"}],
        tenant="t",
    )
    # non-string (multimodal) parts contribute 0 chars; "yo" contributes 2
    assert f.prompt_chars == 2
    assert f.message_count == 2
    assert f.has_tool_use is False


def test_feature_vector_is_numeric_and_stable_order():
    f = extract_features(messages=[{"role": "user", "content": "x" * 10}], tenant="t")
    vec = f.to_vector()
    assert vec == [10.0, 1.0, 0.0, 0.0]  # [prompt_chars, message_count, has_tool_use, tenant_recent_accept_rate]
