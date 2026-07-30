# photon/india/residency.py
"""Data-residency enforcement for Indian deployments.

NOT LEGAL ADVICE. This module provides *engineering controls* that a compliance
position can be built on; whether your specific processing satisfies the DPDP
Act 2023, RBI's payment-data localisation directive, or a sectoral regulator is
a question for your counsel. What Photon can honestly offer is a mechanism that
makes the technical claim checkable: a request tagged as residency-restricted
CANNOT be served by a backend outside the permitted region, and every such
decision leaves an audit trail.

The design point that matters commercially: 'our data stays in India' is the
single most common blocker in Indian enterprise and public-sector AI deals, and
it is usually answered with a policy document. Answering it with an enforced
control plus an audit endpoint is a different conversation.

Note the distinction the module deliberately preserves: `region` (where the
compute physically sits) and `operator_jurisdiction` (whose law the operating
company answers to) are separate fields. An in-India region run by a foreign
entity satisfies the first and not necessarily the second — sovereignty-
sensitive buyers ask about both."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ResidencyPolicy(BaseModel):
    """Per-tenant residency requirement, expressed in COUNTRIES."""

    tenant: str
    allowed_countries: list[str] = Field(default_factory=lambda: ["india"])
    # if set, also require the operating entity's jurisdiction to match
    required_operator_jurisdiction: str | None = None


class BackendResidency(BaseModel):
    """Where a backend actually runs, and who runs it.

    `country` is DECLARED, never inferred from `region`. Cloud region names do
    not encode country in any consistent way — AWS Mumbai is 'ap-south-1',
    Azure's is 'centralindia', GCP's is 'asia-south1' — so a prefix heuristic
    would silently mis-classify real deployments. The operator states the
    country; `region` is a descriptive label for humans."""

    backend: str
    country: str                      # e.g. "india", "usa" — authoritative
    region: str = ""                  # e.g. "ap-south-1", "centralindia"
    operator_jurisdiction: str = ""   # e.g. "india", "usa"


class ResidencyViolation(Exception):
    def __init__(self, tenant: str, backend: str, reason: str):
        self.tenant, self.backend, self.reason = tenant, backend, reason
        super().__init__(f"residency: tenant {tenant!r} may not use {backend!r}: {reason}")


def _country_matches(actual: str, allowed: list[str]) -> bool:
    return (actual or "").strip().lower() in {a.strip().lower() for a in allowed}


class ResidencyEnforcer:
    """Checks a (tenant, backend) pairing against policy. Fails CLOSED: a
    tenant with a policy may not use a backend whose residency is unknown."""

    def __init__(
        self,
        policies: dict[str, ResidencyPolicy] | None = None,
        backends: dict[str, BackendResidency] | None = None,
    ):
        self._policies = policies or {}
        self._backends = backends or {}

    def policy_for(self, tenant: str) -> ResidencyPolicy | None:
        return self._policies.get(tenant)

    def check(self, tenant: str, backend: str) -> None:
        """Raise ResidencyViolation if this pairing is not permitted. No policy
        for the tenant → unrestricted (no-op)."""
        policy = self._policies.get(tenant)
        if policy is None:
            return
        residency = self._backends.get(backend)
        if residency is None or not residency.country:
            raise ResidencyViolation(tenant, backend, "backend residency is undeclared")
        if not _country_matches(residency.country, policy.allowed_countries):
            raise ResidencyViolation(
                tenant, backend,
                f"country {residency.country!r} not in {policy.allowed_countries}",
            )
        required = policy.required_operator_jurisdiction
        if required and residency.operator_jurisdiction.lower() != required.lower():
            raise ResidencyViolation(
                tenant, backend,
                f"operator jurisdiction {residency.operator_jurisdiction!r} != {required!r}",
            )

    def permitted_backends(self, tenant: str) -> list[str] | None:
        """Backends this tenant may use, or None if unrestricted — lets the
        router filter candidates instead of discovering a violation late."""
        if tenant not in self._policies:
            return None
        out = []
        for name in self._backends:
            try:
                self.check(tenant, name)
            except ResidencyViolation:
                continue
            out.append(name)
        return out

    def attestation(self, tenant: str) -> dict:
        """Machine-readable statement of what is enforced for this tenant — the
        artifact to hand an auditor or attach to a security questionnaire."""
        policy = self._policies.get(tenant)
        if policy is None:
            return {"tenant": tenant, "restricted": False,
                    "note": "no residency policy configured; requests may use any backend"}
        return {
            "tenant": tenant,
            "restricted": True,
            "allowed_countries": policy.allowed_countries,
            "required_operator_jurisdiction": policy.required_operator_jurisdiction,
            "permitted_backends": self.permitted_backends(tenant),
            "enforcement": "requests to non-permitted backends are rejected before dispatch",
        }
