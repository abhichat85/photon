# tests/india/test_residency.py
import httpx
import pytest
import respx

from photon.india.residency import (
    BackendResidency,
    ResidencyEnforcer,
    ResidencyPolicy,
    ResidencyViolation,
)

CHAT_RESPONSE = {
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
}


def _enforcer(**policy_kw):
    return ResidencyEnforcer(
        policies={"bank": ResidencyPolicy(tenant="bank", **policy_kw)},
        backends={
            "big": BackendResidency(backend="big", country="india", region="e2e-in-north",
                                    operator_jurisdiction="india"),
            "small": BackendResidency(backend="small", country="usa", region="us-east-1",
                                      operator_jurisdiction="usa"),
            # AWS Mumbai: physically in India, US-operated. Region name encodes
            # NO country — this is why country is declared, not inferred.
            "mumbai-aws": BackendResidency(backend="mumbai-aws", country="india",
                                           region="ap-south-1", operator_jurisdiction="usa"),
            "undeclared": BackendResidency(backend="undeclared", country="", region=""),
        },
    )


def test_indian_backend_allowed_foreign_blocked():
    e = _enforcer()
    e.check("bank", "big")
    with pytest.raises(ResidencyViolation, match="not in"):
        e.check("bank", "small")


def test_country_is_declared_not_guessed_from_region_name():
    """ap-south-1 / centralindia / asia-south1 are all India; no prefix rule
    gets that right, so the config states the country outright."""
    e = _enforcer()
    e.check("bank", "mumbai-aws")  # ap-south-1 accepted because country="india"


def test_fails_closed_on_undeclared_residency():
    e = _enforcer()
    with pytest.raises(ResidencyViolation, match="undeclared"):
        e.check("bank", "undeclared")


def test_operator_jurisdiction_separates_region_from_sovereignty():
    """ap-south-1 is physically in India but operated by a US entity. A tenant
    that only needs in-country compute passes; one that also requires an Indian
    operator does not. Both are legitimate asks and the model keeps them apart."""
    region_only = _enforcer()
    region_only.check("bank", "mumbai-aws")  # in-country → OK

    sovereign = _enforcer(required_operator_jurisdiction="india")
    with pytest.raises(ResidencyViolation, match="jurisdiction"):
        sovereign.check("bank", "mumbai-aws")


def test_tenant_without_policy_is_unrestricted():
    e = _enforcer()
    e.check("startup", "small")  # no policy → no-op, no exception


def test_permitted_backends_lists_only_compliant_ones():
    e = _enforcer()
    assert set(e.permitted_backends("bank")) == {"big", "mumbai-aws"}
    assert e.permitted_backends("startup") is None  # unrestricted


def test_attestation_is_auditor_ready():
    a = _enforcer(required_operator_jurisdiction="india").attestation("bank")
    assert a["restricted"] is True
    assert a["allowed_countries"] == ["india"]
    assert a["permitted_backends"] == ["big"]  # mumbai-aws excluded: US operator
    assert "rejected before dispatch" in a["enforcement"]


@respx.mock
def test_live_request_to_foreign_backend_is_blocked_before_dispatch(client):
    """The control that matters: a violating request never reaches the wire."""
    route = respx.post("http://small.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=CHAT_RESPONSE))
    client.app.state.residency = ResidencyEnforcer(
        policies={"bank": ResidencyPolicy(tenant="bank")},
        backends={"small": BackendResidency(backend="small", country="usa", region="us-east-1")},
    )
    r = client.post("/v1/chat/completions",
                    json={"model": "small", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"X-Photon-Tenant": "bank"})
    assert r.status_code == 451  # Unavailable For Legal Reasons
    assert route.call_count == 0  # nothing left the gateway


@respx.mock
def test_unrestricted_tenant_unaffected(client):
    respx.post("http://big.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=CHAT_RESPONSE))
    client.app.state.residency = ResidencyEnforcer(
        policies={"bank": ResidencyPolicy(tenant="bank")},
        backends={"big": BackendResidency(backend="big", country="india", region="e2e-in-north")},
    )
    r = client.post("/v1/chat/completions",
                    json={"model": "photon-auto", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"X-Photon-Tenant": "someone-else"})
    assert r.status_code == 200


def test_residency_endpoint_reports_unconfigured(client):
    body = client.get("/photon/v1/india/residency", params={"tenant": "bank"}).json()
    assert body["restricted"] is False
