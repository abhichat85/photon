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

## D13 — Solo mode: vLLM-native LoRA, fork only on a measured number

**Context change:** there is no founding inference engineer. It is one operator.
This retires the earlier lean (D-F1 in spec 04: "patched vLLM fork → S-LoRA").
**Decision:** the Fabric substrate is vLLM's **native runtime LoRA management**
(`load_lora_adapter`/`unload_lora_adapter`, enabled by
`VLLM_ALLOW_RUNTIME_LORA_UPDATING`). `FleetEnactor` drives it; a plan is now
*enacted*, not merely advisory. A patched fork or custom Triton kernels are
taken up ONLY if `scripts/benchmark.py` shows native LoRA missing the §9
targets at the target adapter density — i.e. gated on a measurement, never on
ambition. A solo operator cannot maintain a fork on speculation; they can read
a benchmark and decide.
**Change trigger:** `benchmark.py` reports routing-tax or density shortfall at
the density we actually need. Not before.

## D14 — The §9 numbers are code, not prose

**Decision:** the spec's DoD numbers (≤15% routing tax, <3ms selection overhead,
≥3× cost reduction) live in `photon/core/benchmark.py` as scored functions with
the targets baked in, driven by `scripts/benchmark.py` against a live gateway
(exit 1 on any miss). "Benchmark precedes the number" is executable.
**Why:** solo, the discipline that stops wishful claims has to be automated —
you can't peer-review yourself. The gate does it.
**Change trigger:** none — this is the intended steady state.

## D15 — Two things stay physical; everything around them is one command

**Decision:** exactly two steps require the operator's money/judgment and cannot
be pre-executed: (1) renting a GPU (`deploy/GPU_SETUP.md`, ~$1–2/hr) and (2)
cutting live Praxiom traffic over (`deploy/PRAXIOM_CUTOVER.md`, staged +
reversible). Everything bracketing them — enactment, preflight
(`scripts/preflight.py`), benchmarking, shadow enable (`scripts/enable_shadow.py`),
the go/no-go analysis — is built, tested, and single-command.
**Why:** minimize the surface where a solo operator can fumble a high-stakes,
money- or production-touching action. The code does everything that isn't
irreducibly yours.
**Change trigger:** none.

## D16 — Cost is denominated per CHARACTER for Indic traffic, not per token

**Decision:** Photon measures chars-per-token per (backend, script) from live
traffic and routes Indic requests on cost-per-1,000-characters, not $/token.
**Why:** a token is not a constant amount of meaning. Latin-optimised BPEs
fragment Indic scripts, so per-token pricing over-charges Hindi/Tamil/etc for
identical semantic content — and per-token *routing* therefore picks the wrong
model on exactly the traffic an Indian product serves most. Under the
per-character unit, a model that is dearer per token can be genuinely cheaper
for Hindi; that inversion is the arbitrage `IndicAwareRouter` takes.
**Discipline:** an unmeasured (backend, script) pair is ineligible — no assumed
ratios, including vendor Indic-efficiency claims.
**Change trigger:** none — this is the correct unit. If tokenizers converge on
Indic parity the penalty simply measures as ~1.0 and routing stops inverting.

## D17 — Rupee-native, with derived (not quoted) GPU economics

**Decision:** ₹ with GST split is a first-class output; FX rates require
provenance; ₹/1M tokens is derived from ₹/hour ÷ *measured* tokens/second.
**Why:** an Indian buyer cannot act on a USD number, and a cost model built on
vendor throughput claims isn't a cost model. The shipped provider book is
placeholders only, enforced by a test — the India margin story has to be
computed from your quotes and your benchmark, or it isn't real.
**Change trigger:** none.

## D18 — Residency is an enforced control; country is declared, not inferred

**Decision:** residency-restricted tenants are blocked before dispatch (451,
zero upstream calls) with a generated attestation endpoint. Backend `country` is
explicit config, and `operator_jurisdiction` is a separate field.
**Why:** "our data stays in India" is the most common Indian enterprise blocker
and is usually answered with a PDF; an enforced control plus an audit artifact
is a different conversation. Country cannot be inferred from region names (AWS
Mumbai = `ap-south-1`, Azure = `centralindia`, GCP = `asia-south1`) — I wrote
that prefix heuristic first and a test caught it mis-classifying real
deployments. Region and operator jurisdiction are separate questions:
in-country compute run by a foreign entity satisfies localisation but not
sovereignty, and buyers ask about both.
**Change trigger:** none. Note this is engineering, not legal advice.

---

## Tier boundary — solo-mode restatement (what is genuinely NOT here)

Everything CPU-buildable against spec 03/04 is built and tested. What remains is
not deferred-for-a-hire; it is deferred-for-hardware-and-traffic, and the code
is shaped so each slots in with no rework:

- **Tier 2 (needs a GPU box):** the *measured* Fabric numbers and, only if the
  benchmark demands it, fork/kernel work. The enactment path, benchmark harness,
  and isolation checks are all built and unit-tested; they light up the hour a
  GPU exists (`deploy/GPU_SETUP.md`).
- **Tier 3 (needs the Praxiom cutover):** the ≥40%-savings validation and going
  the learned router live. Every instrument — shadow store, history feature,
  replay harness, promotion gate, `enable_shadow.py` — is built and waiting; the
  cutover runbook drives it (`deploy/PRAXIOM_CUTOVER.md`).

No hire is on the critical path. Money (a GPU) and a decision (the cutover) are.
