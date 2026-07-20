# Photon Ops — Decision Record

Deliberate engineering decisions for Photon Ops v1, especially where the
implementation diverges from the tech named in the spec
(`03-photon-inference-engine.md`). The spec named a *target* stack; this records
what v1 actually uses and why, so no choice is a silent substitution.

Format: each decision states the spec expectation, what we did, why, and the
trigger that would change it.

---

## D1 — SQLite for telemetry and registry (not Postgres)

**Spec (§5):** registry uses "Postgres metadata"; telemetry implied durable.
**Decision:** both the telemetry store and the model/adapter registry are
SQLite (WAL, per-call connection).
**Why:** Phase 0 runs a **single gateway replica** (see D3), so a per-pod
embedded store is correct, zero-ops, and fast. Postgres adds a network hop, a
service to run, and connection-pool management for no Phase-0 benefit.
**Durability:** "embedded" must not be confused with "ephemeral". Both files sit
on a ReadWriteOnce PVC (`persistence.enabled`, default on); the registry is
pointed at it explicitly via `PHOTON_REGISTRY_DB=/data/registry.db`, because its
fallback is a *relative* path that lands in the image WORKDIR and dies with the
container. This decision only holds while the store it names actually survives a
restart — telemetry is the audit truth and the registry is the record of what
reached production. The PVC forces `strategy: Recreate` (RWO cannot be attached
to two pods during a rolling update), which costs seconds of downtime and buys
nothing back at replicas: 1.
**Change trigger:** the moment the gateway needs >1 replica, telemetry and
registry must move to Postgres (a shared store) — this is the *same* trigger as
D3 and the spec's own "Postgres-backed telemetry unlocks replicas" note. That
migration also retires the PVC and the Recreate strategy together. Until then,
SQLite is the deliberate choice, not a shortcut.

## D2 — W&B lives in fine-tuning, not the registry

**Spec (§5):** registry row says "W&B (already in use)".
**Decision:** there is no W&B integration in the registry. W&B is wired where it
belongs — the fine-tuning layer: `axolotl_config.py` emits `wandb_project` so
training runs report to W&B.
**Why:** W&B is experiment tracking (training curves, runs), not a model-version
store. The registry's job is versioning + promote/rollback + eval-report
attachment, which is relational state, not experiment telemetry. Conflating them
would be a category error.
**Change trigger:** none expected. If we later want the registry UI to deep-link
to the W&B run that produced an adapter, we store the run URL as a registry
field — no W&B dependency in the gateway.

## D3 — Single gateway replica (per-pod SQLite)

**Decision:** Helm pins `replicas: 1`; HPA is not configured for the gateway.
**Why:** telemetry/registry are per-pod SQLite (D1); multiple replicas would
each hold a partial, unmergeable audit log. The gateway is I/O-bound (it proxies
to vLLM), so one replica handles substantial throughput.
**Change trigger:** move telemetry+registry to Postgres (D1), then the gateway
is stateless and CPU-HPA-safe. Documented in `helm/photon/values.yaml`.

## D4 — Trusted `X-Photon-Tenant` header (no auth yet)

**Decision:** the gateway trusts the `X-Photon-Tenant` request header for tenant
attribution; there is no authentication.
**Why:** Phase 0 serves our own products inside our own network perimeter. Auth
is real work (key management, per-tenant secrets) that isn't needed to get
Praxiom off external inference.
**Change trigger:** HARD prerequisite before Enterprise Deployment #1 exposes
Photon to any customer network. Until auth ships, Photon must not be reachable
from an untrusted network. This is the single most important deferral to
remember.

## D5 — Quantization is fleet metadata + operator responsibility

**Spec (§5):** "quantization applied per fleet-spec, validated by eval harness."
**Decision:** `BackendConfig.quantization` is a validated field (awq/gptq/fp8/
bitsandbytes). The gateway does **not** quantize — vLLM does. The field (a)
documents what each backend serves, (b) is surfaced on `/photon/v1/fleet/status`,
(c) reminds the operator to launch vLLM with the matching `--quantization` arg.
**Validation:** "validated by the eval harness" = after enabling a method, the
golden gate (`run_evals.py` / `registry.promote` gate) must still pass. The
field + the gate together are the spec's intent.
**Change trigger:** if Photon ever manages vLLM lifecycle directly (it doesn't
in Ops), it would pass the arg itself.

## D6 — `photon` request block: Ops honors routing, records the rest

**Spec (§6):** requests may carry `quality_bar`, `latency_slo_ms`, `budget`,
`route`, `audit`.
**Decision:** at Ops the gateway parses the block and:
- honors `route: pin` (disables canary) and `route: auto`;
- **rejects** `route: cascade` with 400 — cascade needs the learned router (Core);
- **records** `quality_bar`/`latency_slo_ms`/`budget` to telemetry as
  forward-compat training data, but does **not** enforce them;
- treats `audit: true` as a no-op (every request is already audited).
**Why:** enforcement of quality/latency/budget is exactly the Photon Core router
job. Recording them now means the day Core lands, it has real labeled data. The
API contract is honest: it never silently ignores a field or pretends to enforce
one it can't.
**Change trigger:** Photon Core implements enforcement; the recorded fields
become its inputs.

## D7 — Observability integrations are gated-optional

**Spec (§5):** OpenTelemetry, Sentry, Loki.
**Decision:** structured JSON logging is built-in. OTel and Sentry are **gated**:
no-op unless BOTH the env var (`OTEL_EXPORTER_OTLP_ENDPOINT` / `SENTRY_DSN`) and
the optional extra (`pip install '.[otel]'` / `'.[sentry]'`) are present. Loki +
promtail ship in docker-compose.
**Why:** dev and test stay dependency-light; production opts in with zero code
change. Forcing OTel/Sentry into the base install would bloat every environment
for a capability most don't use in Phase 0.
**Change trigger:** none — this is the intended steady state.

## D8 — Terraform is the IaC reference; GKE/AKS are CLI-first

**Spec (§5):** EKS + GKE + AKS; Terraform.
**Decision:** all three clouds have runbooks (`EKS.md`, `GKE.md`, `AKS.md`).
Terraform IaC exists for EKS (`deploy/terraform/`), the primary cloud; GKE/AKS
are CLI-first (gcloud/az) for now.
**Why:** the first enterprise deployment's cloud isn't yet known; EKS is the most
likely and got the full IaC treatment. Writing speculative Terraform for three
clouds before knowing the target is waste.
**Change trigger:** once Enterprise Deployment #1's cloud is known, that cloud
gets the full IaC module (GKE/AKS Terraform follows the same module shape).

## D9 — Manual vLLM scaling in v1

**Decision:** vLLM replica counts are set manually; no queue-depth HPA.
**Why:** correct autoscaling for LLM serving (custom metrics on queue depth /
GPU utilization) is real work that shouldn't be improvised during a deployment.
**Change trigger:** traffic that justifies it; then a custom-metrics HPA, planned
deliberately. Documented in every cloud runbook's §5.

## D10 — Shadow persistence + the acceptance-rate proxy

**Decision:** shadow decisions persist to a dedicated SQLite store
(`PHOTON_SHADOW_DB`, surfaced at `GET /photon/v1/shadow/decisions`), and the
policy's tenant-history feature is `recent_ok_rate` — the share of the tenant's
recent requests that *succeeded*.
**Why the proxy:** true "acceptance" labels (was the cheap model's answer good
enough?) only exist after the Tier-3 study grades counterfactuals. Success rate
is the honest signal available today; it feeds the same feature slot and is
replaced by graded labels when they exist. Naming it a proxy here prevents it
from quietly being treated as ground truth.
**Change trigger:** Tier-3 grading pipeline produces real acceptance labels.

## D11 — Learned routing is a capability, not a default

**Decision:** `LearnedRoutingAdapter` makes the learned engine a drop-in for
`resolve()` (proven end-to-end in tests), but `create_app` always installs the
static router. Enabling learned routing in production is a deliberate act,
gated on the Tier-3 shadow-study result, and rolls out shadow → canary → full.
Fail-safe by construction: alias/direct/pin/featureless requests always fall
back to the static path, and the router policy itself must pass the registry's
promotion gate (oracle agreement as its eval report) like any other model.
**Change trigger:** shadow study ≥ 25% confidently-routable (spec §7 exit gate).

## D12 — Pipelines are per-process, sequential-chain, budget-enforced

**Decision:** pipeline specs register in-memory per boot (config-like, same
posture as the fleet plan); execution is a sequential chain with per-stage
RouteTargets and an end-to-end latency budget enforced between stages.
**Why:** the sequential chain is Praxiom-1's actual shape; branching DAGs and
on-GPU cross-stage KV/prefix reuse belong to the Tier-2 engine that owns GPU
memory — building them against a mocked backend would be shape without
substance.
**Change trigger:** the Tier-2 dense engine lands behind ServingBackend; the
orchestrator then gains what only real GPU residency makes meaningful.

---

## The Ops / Core line (what is deliberately NOT here)

Photon Ops v1 is the serving + deployment + quality substrate. The **moat** —
Photon Core — is deliberately deferred, gated on the Phase 0 shadow study and a
founding inference-engineer hire:

- **Fabric** — many-adapter serving at density (S-LoRA-class unified paging,
  Triton kernels, the fleet manager's dynamic residency). Ops covers single/
  few-adapter serving via vLLM `--enable-lora`; Fabric is hundreds at near-zero
  swap latency.
- **Learned Router** — feature extraction + policy model + cascade controller +
  regret loop. Ops ships the *seam* (static router behind `resolve()`, the
  `photon` block recorded, the shadow study collecting data); Core replaces the
  static table without an API change.

These are not gaps in Ops — they are the next phase, and the whole point of
finishing Ops to 100% first.
