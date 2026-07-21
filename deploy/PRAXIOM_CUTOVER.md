# Praxiom → Photon Cutover — the other physical thing only you can do

Pointing a live paying product at a new inference layer is a judgment call, not
a code change — so this is a runbook, staged and reversible at every step. It is
the Tier-3 gate: it produces the real traffic the shadow study needs, and only
after that study passes does the learned router ever serve a decision.

Prereq: the GPU box is up (deploy/GPU_SETUP.md) and `scripts/preflight.py` passes.

## Stage 0 — Baseline (measure before you touch anything)

For 2 weeks, record Praxiom's CURRENT external-provider cost and its golden-set
pass rate. This is the number every later step is compared against; skipping it
means you can never prove Photon was better (parent spec anti-goal: benchmark
precedes the number).

```bash
python -m scripts.run_evals <current-provider-url> evals/praxiom_golden.yaml --min-pass-rate 1.0
# record the pass rate + your provider bill
```

## Stage 1 — Mirror (Photon serves, but Praxiom doesn't depend on it yet)

Point a COPY of Praxiom's traffic at Photon, or run Photon in your infra and send
it synthetic replays of real prompts. Confirm parity:

```bash
python -m scripts.preflight --gateway http://photon:8080 --vllm http://vllm:8000/v1
python -m scripts.run_evals http://photon:8080 evals/praxiom_golden.yaml --min-pass-rate <stage-0-rate>
```

Green = Photon serves Praxiom's workload at ≥ the baseline quality.

## Stage 2 — Cut 10% (real dependency, small blast radius)

In Praxiom's LLM client, send 10% of production requests to Photon with
`model: "photon-auto"`. Watch the Grafana dashboards (error rate, p95, cost/hr)
and the golden gate. Roll back instantly by flipping the 10% to 0 — it's a
config value, not a deploy.

Ramp 10% → 50% → 100% only while the golden set stays green at each step.

## Stage 3 — Turn ON the shadow study (collect the Tier-3 data)

Once Praxiom is fully on Photon, enable shadow logging so the learned router
starts recording counterfactuals against real traffic:

```python
# in your Photon deployment's startup (see scripts/enable_shadow.py for the wiring)
from scripts.enable_shadow import install_shadow_router
install_shadow_router(app)   # logs would-route decisions to the shadow store
```

Let it run ≥ 2 weeks. Then read the study:

```bash
curl http://photon:8080/photon/v1/shadow/decisions | python3 -m json.tool
python -m scripts.cost_report shadow.db praxiom   # actual vs counterfactual cost
```

## Stage 4 — The go/no-go (spec §7 exit gate)

Grade a sample of shadow-flagged cheap-routes against the golden set. If ≥ 25%
of traffic is confidently routable to a ≥ 5× cheaper model at flat quality:
**go** — train a policy, register it (`register_policy_version`), pass the
promotion gate, and install a `LearnedRoutingAdapter` (canary first). If not:
the honest answer is Fabric/pipeline value only, and you say so. Either way, the
number decides — not the roadmap.
