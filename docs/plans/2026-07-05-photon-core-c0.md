# Photon Core — Phase C0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the CPU-tractable slice of Photon Core — a learned routing engine (shadow-mode) and a Fabric control-plane against a mocked serving backend — that plugs into the Ops seams with zero production behavior change.

**Architecture:** A `RoutingEngine` decides `(model_id, adapter_id, decode_params)` from cheap request features via a calibrated policy model + cascade controller; a `ServingBackend` interface abstracts execution (Mock + thin-vLLM impls now, dense GPU engine later). A `FleetManager` turns a declarative `FleetSpec` into a `PlacementPlan`. Everything ships in shadow mode: the learned router logs what it *would* route alongside the live static router's actual decision, enforcing nothing, so the routing bet can be measured on real traffic before it goes live.

**Tech Stack:** Python 3.11+, Pydantic v2, scikit-learn + numpy (policy model), the existing Photon Ops modules (telemetry, registry, config, static router, admin API), pytest.

**Decisions locked (spec §7):** D-S1 = build the full C0 slice now (shadow + mock). D-F1 = Tier-2 substrate target is a patched vLLM fork → S-LoRA (the `ServingBackend` interface is designed for it, but C0 only implements Mock + thin-vLLM). D-R1 = cheap features only in C0 (length + metadata + tenant history; no embedding model).

**Scope boundary:** C0 is Tier 1 only (spec §1). NOT in this plan: the dense multi-adapter GPU serving engine, Triton kernels, on-GPU pipeline KV reuse (Tier 2, needs the hire + GPUs); the ≥40%-savings validation (Tier 3, needs Praxiom cutover). The learned router NEVER changes which backend serves a request in C0 — it only logs counterfactual decisions.

Spec: `docs/superpowers/specs/2026-07-03-einstein-labs-full-stack-strategy/04-photon-core-implementation.md` (in the Einstein-Labs workspace). All paths below are relative to the `photon/` repo root.

---

## File Structure

```
photon/core/
├── __init__.py
├── contract.py        # RouteTarget, RoutingEngine (Protocol), ServingBackend (Protocol)
├── features.py        # RequestFeatures + extract_features()
├── policy.py          # PolicyModel: fit / predict_acceptable / save / load (sklearn)
├── cascade.py         # CascadeController + EscalationSignals; decide()
├── regret.py          # router_regret() over replay rows
├── learning.py        # ReplayHarness: telemetry rows → training set → fit → EvalReport
├── router.py          # LearnedRouter (decide) + ShadowRouter (wraps StaticRouter, log-only)
├── fleet.py           # FleetSpec, PlacementPlan, FleetManager.plan()
└── serving.py         # ServingBackend impls: MockServingBackend, VLLMServingBackend
tests/core/
├── __init__.py
├── test_contract.py
├── test_features.py
├── test_policy.py
├── test_cascade.py
├── test_regret.py
├── test_learning.py
├── test_router.py
├── test_fleet.py
├── test_serving.py
└── test_api_fleet.py   # POST /photon/v1/fleet + dynamic /fleet/status
```

Plus: `pyproject.toml` gains a `[core]` extra (scikit-learn, numpy) added to `dev` so tests run; `photon/api/admin.py` gains the fleet-apply endpoint; `photon/api/app.py` wires a `FleetManager` + shadow router into state.

Design rules: the Router never imports Fabric and vice-versa — they meet only at `contract.py`. The policy model is offline (training/eval), never in the hot request path beyond `predict`. Shadow mode is structurally enforced: `ShadowRouter` has no code path that returns a different served backend than the wrapped `StaticRouter`.

---

### Task 0: Core package scaffold + ML deps

**Files:**
- Create: `photon/core/__init__.py`, `tests/core/__init__.py` (both empty)
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the `[core]` extra and fold it into `dev`.** In `pyproject.toml`, under `[project.optional-dependencies]`, add:

```toml
core = [
    "scikit-learn>=1.4",
    "numpy>=1.26",
]
```

and add `"scikit-learn>=1.4"`, `"numpy>=1.26"` to the existing `dev` list so the test suite can train a policy model.

- [ ] **Step 2: Create the empty package files** `photon/core/__init__.py` and `tests/core/__init__.py`.

- [ ] **Step 3: Install and verify**

Run: `pip install -e ".[dev]" && python -c "import sklearn, numpy; print('ml ok')"`
Expected: `ml ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml photon/core/__init__.py tests/core/__init__.py
git commit -m "chore(core): scaffold photon.core package + ml deps"
```

---

### Task 1: The Router→Fabric contract

**Files:**
- Create: `photon/core/contract.py`
- Test: `tests/core/test_contract.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_contract.py
from photon.core.contract import RouteTarget


def test_route_target_holds_model_adapter_params():
    t = RouteTarget(model_id="qwen-7b", adapter_id="praxiom-intent-v3", decode_params={"temperature": 0.2})
    assert t.model_id == "qwen-7b"
    assert t.adapter_id == "praxiom-intent-v3"
    assert t.decode_params["temperature"] == 0.2


def test_route_target_adapter_optional():
    t = RouteTarget(model_id="qwen-72b")
    assert t.adapter_id is None
    assert t.decode_params == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_contract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'photon.core.contract'`

- [ ] **Step 3: Implement `photon/core/contract.py`**

```python
# photon/core/contract.py
"""The seam between the Router (selection) and the Fabric (execution). Defined
first so the two subsystems can be built independently and so the Tier-2 GPU
serving engine can drop in behind ServingBackend later without touching either."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class RouteTarget(BaseModel):
    """What the Router decides: which base model, which adapter (if any), and the
    decode params to run with."""

    model_id: str
    adapter_id: str | None = None
    decode_params: dict = Field(default_factory=dict)


@runtime_checkable
class RoutingEngine(Protocol):
    """Owned by the Router. Given extracted features, choose a RouteTarget.
    The learned engine and the static fallback both satisfy this."""

    def decide(self, features: "object") -> RouteTarget: ...


@runtime_checkable
class ServingBackend(Protocol):
    """Owned by the Fabric. Execute a RouteTarget. MockServingBackend and
    VLLMServingBackend implement it now; the dense GPU engine implements it later."""

    async def generate(self, target: RouteTarget, payload: dict) -> tuple[dict, float]: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_contract.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add photon/core/contract.py tests/core/test_contract.py
git commit -m "feat(core): Router-Fabric contract (RouteTarget, RoutingEngine, ServingBackend)"
```

---

### Task 2: Cheap feature extraction (D-R1)

**Files:**
- Create: `photon/core/features.py`
- Test: `tests/core/test_features.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_features.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `photon/core/features.py`**

```python
# photon/core/features.py
"""Cheap, deterministic request features for the policy model (D-R1: no
embedding model in C0). The feature vector order is FIXED — the policy model is
trained and scored against this exact order."""
from __future__ import annotations

from pydantic import BaseModel


class RequestFeatures(BaseModel):
    prompt_chars: int
    message_count: int
    has_tool_use: bool = False
    tenant: str = "default"
    route_hint: str = "auto"
    pipeline_stage: str | None = None
    tenant_recent_accept_rate: float = 0.0

    def to_vector(self) -> list[float]:
        """FIXED order — do not reorder without retraining every policy model."""
        return [
            float(self.prompt_chars),
            float(self.message_count),
            float(self.has_tool_use),
            float(self.tenant_recent_accept_rate),
        ]


def _content_chars(messages: list[dict]) -> int:
    return sum(len(m.get("content")) for m in messages if isinstance(m.get("content"), str))


def extract_features(
    messages: list[dict],
    tenant: str = "default",
    route_hint: str = "auto",
    pipeline_stage: str | None = None,
    tenant_recent_accept_rate: float = 0.0,
    tools: list | None = None,
) -> RequestFeatures:
    return RequestFeatures(
        prompt_chars=_content_chars(messages),
        message_count=len(messages),
        has_tool_use=bool(tools),
        tenant=tenant,
        route_hint=route_hint,
        pipeline_stage=pipeline_stage,
        tenant_recent_accept_rate=tenant_recent_accept_rate,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_features.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add photon/core/features.py tests/core/test_features.py
git commit -m "feat(core): cheap deterministic feature extraction"
```

---

### Task 3: Policy model (P(cheap model acceptable))

**Files:**
- Create: `photon/core/policy.py`
- Test: `tests/core/test_policy.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_policy.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_policy.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `photon/core/policy.py`**

```python
# photon/core/policy.py
"""The routing policy: predicts P(the cheap model's output is acceptable) from
request features. Calibrated logistic regression — fast, interpretable, and 3+
orders of magnitude cheaper than the inference it gates (parent spec anti-goal:
never an LLM-as-router). Trained offline on replay data (learning.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from photon.core.features import RequestFeatures


class PolicyModel:
    def __init__(self):
        self._model = None  # sklearn estimator, set by fit()/load()

    def fit(self, features: list[RequestFeatures], labels: list[int]) -> None:
        from sklearn.linear_model import LogisticRegression

        X = np.array([f.to_vector() for f in features], dtype=float)
        y = np.array(labels, dtype=int)
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        self._model = model

    def predict_acceptable(self, features: RequestFeatures) -> float:
        """Return P(cheap acceptable) in [0, 1]. Fails closed (0.0) when
        untrained, so an unfit policy never routes anything to the cheap model."""
        if self._model is None:
            return 0.0
        X = np.array([features.to_vector()], dtype=float)
        return float(self._model.predict_proba(X)[0][1])

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump(self._model, path)

    @classmethod
    def load(cls, path: str | Path) -> "PolicyModel":
        import joblib

        inst = cls()
        inst._model = joblib.load(path)
        return inst
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_policy.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add photon/core/policy.py tests/core/test_policy.py
git commit -m "feat(core): calibrated logistic-regression routing policy"
```

---

### Task 4: Cascade controller

**Files:**
- Create: `photon/core/cascade.py`
- Test: `tests/core/test_cascade.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_cascade.py
from photon.core.cascade import CascadeController, EscalationSignals
from photon.core.contract import RouteTarget
from photon.core.features import RequestFeatures
from photon.core.policy import PolicyModel


class StubPolicy(PolicyModel):
    def __init__(self, score: float):
        super().__init__()
        self._score = score

    def predict_acceptable(self, features):
        return self._score


CHEAP = RouteTarget(model_id="qwen-1.5b")
BIG = RouteTarget(model_id="qwen-72b")


def _feat():
    return RequestFeatures(prompt_chars=10, message_count=1)


def test_high_confidence_routes_cheap():
    c = CascadeController(StubPolicy(0.9), threshold=0.6, cheap=CHEAP, big=BIG)
    d = c.decide(_feat())
    assert d.target.model_id == "qwen-1.5b"
    assert d.reason == "policy-cheap"


def test_low_confidence_routes_big():
    c = CascadeController(StubPolicy(0.2), threshold=0.6, cheap=CHEAP, big=BIG)
    d = c.decide(_feat())
    assert d.target.model_id == "qwen-72b"
    assert d.reason == "policy-escalate"


def test_critic_score_forces_escalation_even_if_policy_confident():
    c = CascadeController(StubPolicy(0.95), threshold=0.6, cheap=CHEAP, big=BIG)
    d = c.decide(_feat(), signals=EscalationSignals(critic_score=0.1, critic_floor=0.5))
    assert d.target.model_id == "qwen-72b"
    assert d.reason == "critic-escalate"


def test_schema_failure_forces_escalation():
    c = CascadeController(StubPolicy(0.95), threshold=0.6, cheap=CHEAP, big=BIG)
    d = c.decide(_feat(), signals=EscalationSignals(schema_valid=False))
    assert d.reason == "schema-escalate"


def test_budget_exhausted_forces_cheap():
    c = CascadeController(StubPolicy(0.2), threshold=0.6, cheap=CHEAP, big=BIG)
    d = c.decide(_feat(), signals=EscalationSignals(budget_remaining=0.0))
    assert d.target.model_id == "qwen-1.5b"
    assert d.reason == "budget-cheap"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_cascade.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `photon/core/cascade.py`**

```python
# photon/core/cascade.py
"""Default-to-cheap with escalation. Escalation triggers (in priority order):
per-tenant budget exhausted (forces cheap — hard cost cap wins), a failed schema
/ output signal, Praxiom-1's critic score below a floor (the coupling advantage —
no horizontal router has this signal), then the policy model's own confidence."""
from __future__ import annotations

from pydantic import BaseModel

from photon.core.contract import RouteTarget
from photon.core.features import RequestFeatures
from photon.core.policy import PolicyModel


class EscalationSignals(BaseModel):
    critic_score: float | None = None       # Praxiom-1 critic output, if available
    critic_floor: float = 0.5
    schema_valid: bool = True                # False = cheap output failed validation
    budget_remaining: float | None = None    # per-tenant $ left; 0 forces cheap


class CascadeDecision(BaseModel):
    target: RouteTarget
    reason: str
    policy_score: float


class CascadeController:
    def __init__(
        self,
        policy: PolicyModel,
        threshold: float,
        cheap: RouteTarget,
        big: RouteTarget,
    ):
        self._policy = policy
        self._threshold = threshold
        self._cheap = cheap
        self._big = big

    def decide(
        self, features: RequestFeatures, signals: EscalationSignals | None = None
    ) -> CascadeDecision:
        signals = signals or EscalationSignals()
        score = self._policy.predict_acceptable(features)

        # hard cost cap wins over everything
        if signals.budget_remaining is not None and signals.budget_remaining <= 0:
            return CascadeDecision(target=self._cheap, reason="budget-cheap", policy_score=score)
        if not signals.schema_valid:
            return CascadeDecision(target=self._big, reason="schema-escalate", policy_score=score)
        if signals.critic_score is not None and signals.critic_score < signals.critic_floor:
            return CascadeDecision(target=self._big, reason="critic-escalate", policy_score=score)
        if score >= self._threshold:
            return CascadeDecision(target=self._cheap, reason="policy-cheap", policy_score=score)
        return CascadeDecision(target=self._big, reason="policy-escalate", policy_score=score)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_cascade.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add photon/core/cascade.py tests/core/test_cascade.py
git commit -m "feat(core): cascade controller with budget/schema/critic/policy triggers"
```

---

### Task 5: Router-regret metric

**Files:**
- Create: `photon/core/regret.py`
- Test: `tests/core/test_regret.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_regret.py
from photon.core.regret import ReplayRow, router_regret


def test_zero_regret_when_router_matches_oracle():
    # oracle picks the cheapest acceptable; router did too
    rows = [
        ReplayRow(chosen_cost=0.001, cheapest_acceptable_cost=0.001),
        ReplayRow(chosen_cost=0.010, cheapest_acceptable_cost=0.010),
    ]
    assert router_regret(rows) == 0.0


def test_regret_is_mean_excess_cost():
    rows = [
        ReplayRow(chosen_cost=0.010, cheapest_acceptable_cost=0.001),  # excess 0.009
        ReplayRow(chosen_cost=0.005, cheapest_acceptable_cost=0.005),  # excess 0.0
    ]
    assert router_regret(rows) == 0.0045


def test_empty_replay_is_zero_regret():
    assert router_regret([]) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_regret.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `photon/core/regret.py`**

```python
# photon/core/regret.py
"""Router regret: mean excess cost vs. an oracle that always picks the cheapest
model that would have been acceptable in hindsight. The north-star internal
metric for the learning loop — lower is better, 0 is oracle-optimal."""
from __future__ import annotations

from pydantic import BaseModel


class ReplayRow(BaseModel):
    chosen_cost: float               # what the router's decision actually cost
    cheapest_acceptable_cost: float  # oracle-in-hindsight cost


def router_regret(rows: list[ReplayRow]) -> float:
    if not rows:
        return 0.0
    excess = [max(0.0, r.chosen_cost - r.cheapest_acceptable_cost) for r in rows]
    return sum(excess) / len(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_regret.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add photon/core/regret.py tests/core/test_regret.py
git commit -m "feat(core): router-regret metric (mean excess cost vs oracle)"
```

---

### Task 6: Offline learning loop (replay harness)

**Files:**
- Create: `photon/core/learning.py`
- Test: `tests/core/test_learning.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_learning.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `photon/core/learning.py`**

```python
# photon/core/learning.py
"""Offline replay harness: take labeled historical rows (features + whether the
cheap model was acceptable + the two costs), train the policy, then evaluate the
resulting router against the replay — reporting router-regret and cost savings
vs. an always-big baseline. This is what runs nightly on real telemetry; here it
runs on any list of LabeledRow so it is fully testable on synthetic data."""
from __future__ import annotations

from pydantic import BaseModel, Field

from photon.core.cascade import CascadeController
from photon.core.contract import RouteTarget
from photon.core.features import RequestFeatures
from photon.core.policy import PolicyModel
from photon.core.regret import ReplayRow, router_regret


class LabeledRow(BaseModel):
    features: RequestFeatures
    cheap_acceptable: bool
    cheap_cost: float
    big_cost: float


class ReplayReport(BaseModel):
    trained_on: int
    routed_cheap_share: float
    regret: float
    always_big_regret: float
    est_savings_vs_always_big: float
    policy_scores: list[float] = Field(default_factory=list)


class ReplayHarness:
    def __init__(self, threshold: float = 0.6):
        self._threshold = threshold

    def run(self, rows: list[LabeledRow]) -> ReplayReport:
        if not rows:
            return ReplayReport(
                trained_on=0, routed_cheap_share=0.0, regret=0.0,
                always_big_regret=0.0, est_savings_vs_always_big=0.0,
            )
        policy = PolicyModel()
        policy.fit([r.features for r in rows], [int(r.cheap_acceptable) for r in rows])
        cheap = RouteTarget(model_id="cheap")
        big = RouteTarget(model_id="big")
        controller = CascadeController(policy, self._threshold, cheap, big)

        replay: list[ReplayRow] = []
        always_big: list[ReplayRow] = []
        cheap_count = 0
        chosen_total = 0.0
        big_total = 0.0
        scores: list[float] = []
        for r in rows:
            d = controller.decide(r.features)
            scores.append(d.policy_score)
            oracle = r.cheap_cost if r.cheap_acceptable else r.big_cost
            if d.target.model_id == "cheap":
                cheap_count += 1
                # a cheap route on an unacceptable row "costs" the big price (a retry)
                chosen = r.cheap_cost if r.cheap_acceptable else r.big_cost
            else:
                chosen = r.big_cost
            chosen_total += chosen
            big_total += r.big_cost
            replay.append(ReplayRow(chosen_cost=chosen, cheapest_acceptable_cost=oracle))
            always_big.append(ReplayRow(chosen_cost=r.big_cost, cheapest_acceptable_cost=oracle))

        savings = (big_total - chosen_total) / big_total if big_total else 0.0
        return ReplayReport(
            trained_on=len(rows),
            routed_cheap_share=cheap_count / len(rows),
            regret=router_regret(replay),
            always_big_regret=router_regret(always_big),
            est_savings_vs_always_big=max(0.0, savings),
            policy_scores=scores,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_learning.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add photon/core/learning.py tests/core/test_learning.py
git commit -m "feat(core): offline replay harness (train + regret + savings report)"
```

---

### Task 7: LearnedRouter + ShadowRouter (log-only)

**Files:**
- Create: `photon/core/router.py`
- Test: `tests/core/test_router.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_router.py
from photon.core.contract import RouteTarget
from photon.core.features import RequestFeatures
from photon.core.policy import PolicyModel
from photon.core.router import LearnedRouter, ShadowDecision, ShadowRouter


class StubPolicy(PolicyModel):
    def __init__(self, score):
        super().__init__()
        self._score = score

    def predict_acceptable(self, features):
        return self._score


def _feat(chars=10):
    return RequestFeatures(prompt_chars=chars, message_count=1)


def test_learned_router_decides_cheap_when_confident():
    r = LearnedRouter(
        policy=StubPolicy(0.9), threshold=0.6,
        cheap=RouteTarget(model_id="cheap"), big=RouteTarget(model_id="big"),
    )
    assert r.decide(_feat()).model_id == "cheap"


def test_shadow_router_returns_actual_but_logs_counterfactual():
    logged = []
    learned = LearnedRouter(
        policy=StubPolicy(0.9), threshold=0.6,
        cheap=RouteTarget(model_id="cheap"), big=RouteTarget(model_id="big"),
    )
    shadow = ShadowRouter(learned, sink=logged.append)
    # the actual served backend is whatever Ops chose ("big"); shadow must return it unchanged
    served = shadow.observe(actual_backend_name="big", features=_feat(), request_id="r1")
    assert served == "big"  # NEVER overridden in shadow mode
    assert len(logged) == 1
    d: ShadowDecision = logged[0]
    assert d.request_id == "r1"
    assert d.actual_backend == "big"
    assert d.would_route.model_id == "cheap"  # counterfactual only
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_router.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `photon/core/router.py`**

```python
# photon/core/router.py
"""LearnedRouter: a RoutingEngine (contract.py) driven by the policy + cascade.
ShadowRouter: wraps it for C0's shadow mode — it OBSERVES each request, logs the
counterfactual RouteTarget the learned engine WOULD have chosen, and returns the
actual backend Ops already selected, unchanged. There is deliberately no code
path from ShadowRouter to the served response: shadow mode cannot alter traffic.
Going live (Phase C1) means swapping ShadowRouter.observe for a decide that Ops
honors — gated on the Tier-3 validation."""
from __future__ import annotations

from typing import Callable

from pydantic import BaseModel

from photon.core.cascade import CascadeController, EscalationSignals
from photon.core.contract import RouteTarget
from photon.core.features import RequestFeatures
from photon.core.policy import PolicyModel


class LearnedRouter:
    def __init__(self, policy: PolicyModel, threshold: float, cheap: RouteTarget, big: RouteTarget):
        self._cascade = CascadeController(policy, threshold, cheap, big)

    def decide(self, features: RequestFeatures, signals: EscalationSignals | None = None) -> RouteTarget:
        return self._cascade.decide(features, signals).target


class ShadowDecision(BaseModel):
    request_id: str
    actual_backend: str
    would_route: RouteTarget
    reason: str


class ShadowRouter:
    def __init__(self, learned: LearnedRouter, sink: Callable[[ShadowDecision], None]):
        self._learned = learned
        self._sink = sink

    def observe(self, actual_backend_name: str, features: RequestFeatures, request_id: str) -> str:
        """Log what the learned router WOULD have chosen; return the actual
        backend unchanged. This is the whole of C0's routing behavior."""
        decision = self._learned._cascade.decide(features)
        self._sink(
            ShadowDecision(
                request_id=request_id,
                actual_backend=actual_backend_name,
                would_route=decision.target,
                reason=decision.reason,
            )
        )
        return actual_backend_name
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_router.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add photon/core/router.py tests/core/test_router.py
git commit -m "feat(core): LearnedRouter + shadow-mode wrapper (log-only, no traffic change)"
```

---

### Task 8: Fabric fleet manager (placement logic)

**Files:**
- Create: `photon/core/fleet.py`
- Test: `tests/core/test_fleet.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_fleet.py
import pytest

from photon.core.fleet import AdapterSpec, FleetManager, FleetSpec


def _spec(capacity):
    return FleetSpec(
        base_models=["qwen-1.5b", "qwen-14b"],
        adapters=[
            AdapterSpec(name="legal-v3", base="qwen-1.5b", pinned=True),
            AdapterSpec(name="fin-v2", base="qwen-1.5b", pinned=False),
            AdapterSpec(name="health-v1", base="qwen-14b", pinned=False),
        ],
        slot_capacity=capacity,
    )


def test_plan_places_all_when_capacity_allows():
    plan = FleetManager().plan(_spec(capacity=5))
    assert set(plan.resident_adapters) == {"legal-v3", "fin-v2", "health-v1"}
    assert plan.paged_adapters == []


def test_pinned_adapters_always_resident_under_pressure():
    # only 1 adapter slot beyond the 2 base models → pinned wins
    plan = FleetManager().plan(_spec(capacity=3))
    assert "legal-v3" in plan.resident_adapters  # pinned
    assert set(plan.paged_adapters)  # something got paged out
    assert "legal-v3" not in plan.paged_adapters


def test_plan_rejects_adapter_referencing_unknown_base():
    bad = FleetSpec(
        base_models=["qwen-1.5b"],
        adapters=[AdapterSpec(name="x", base="nonexistent", pinned=False)],
        slot_capacity=5,
    )
    with pytest.raises(ValueError, match="unknown base"):
        FleetManager().plan(bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_fleet.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `photon/core/fleet.py`**

```python
# photon/core/fleet.py
"""Fabric control-plane: turn a declarative FleetSpec into a PlacementPlan
(which adapters are resident vs. paged) using pin-priority then insertion order.
This is the placement LOGIC — the IP — independent of the GPU execution that
enacts it (that's the Tier-2 ServingBackend). Advisory in C0, load-bearing once
the dense engine lands."""
from __future__ import annotations

from pydantic import BaseModel


class AdapterSpec(BaseModel):
    name: str
    base: str
    pinned: bool = False


class FleetSpec(BaseModel):
    base_models: list[str]
    adapters: list[AdapterSpec]
    slot_capacity: int  # total resident slots (bases + adapters) on the pool


class PlacementPlan(BaseModel):
    resident_bases: list[str]
    resident_adapters: list[str]
    paged_adapters: list[str]


class FleetManager:
    def plan(self, spec: FleetSpec) -> PlacementPlan:
        bases = set(spec.base_models)
        for a in spec.adapters:
            if a.base not in bases:
                raise ValueError(f"adapter {a.name!r} references unknown base {a.base!r}")

        # bases are always resident; adapters fill remaining slots, pinned first
        adapter_slots = max(0, spec.slot_capacity - len(spec.base_models))
        ordered = sorted(
            spec.adapters, key=lambda a: (not a.pinned,)  # pinned (False sorts first)
        )
        resident = [a.name for a in ordered[:adapter_slots]]
        paged = [a.name for a in ordered[adapter_slots:]]
        return PlacementPlan(
            resident_bases=list(spec.base_models),
            resident_adapters=resident,
            paged_adapters=paged,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_fleet.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add photon/core/fleet.py tests/core/test_fleet.py
git commit -m "feat(core): fabric fleet manager placement logic (pin-priority paging)"
```

---

### Task 9: Serving backends (Mock + thin vLLM) behind the interface

**Files:**
- Create: `photon/core/serving.py`
- Test: `tests/core/test_serving.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_serving.py
import httpx
import respx

from photon.config import BackendConfig, ModelPricing
from photon.core.contract import RouteTarget, ServingBackend
from photon.core.serving import MockServingBackend, VLLMServingBackend

RESP = {"choices": [{"message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


async def test_mock_backend_is_a_serving_backend():
    m = MockServingBackend(canned={"cheap": {"ok": True}})
    assert isinstance(m, ServingBackend)
    resp, latency = await m.generate(RouteTarget(model_id="cheap"), {"messages": []})
    assert resp == {"ok": True}
    assert latency >= 0.0


async def test_mock_records_calls():
    m = MockServingBackend(canned={"cheap": {"ok": True}})
    await m.generate(RouteTarget(model_id="cheap", adapter_id="legal-v3"), {"messages": []})
    assert m.calls[0].model_id == "cheap"
    assert m.calls[0].adapter_id == "legal-v3"


@respx.mock
async def test_vllm_backend_posts_and_injects_adapter_as_model():
    backend = BackendConfig(name="small", base_url="http://s.test/v1", model="qwen-1.5b",
                            pricing=ModelPricing(input_per_1m=0.08, output_per_1m=0.08))
    route = respx.post("http://s.test/v1/chat/completions").mock(return_value=httpx.Response(200, json=RESP))
    async with httpx.AsyncClient() as client:
        b = VLLMServingBackend({"small": backend}, client)
        # adapter_id becomes the served model name (vLLM serves LoRA modules by name)
        resp, latency = await b.generate(
            RouteTarget(model_id="small", adapter_id="legal-v3"), {"messages": [{"role": "user", "content": "hi"}]}
        )
    assert resp["usage"]["prompt_tokens"] == 3
    import json
    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == "legal-v3"  # adapter name wins when present
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_serving.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `photon/core/serving.py`**

```python
# photon/core/serving.py
"""ServingBackend implementations for C0. MockServingBackend for tests;
VLLMServingBackend is a thin adapter-aware wrapper over the existing OpenAI proxy
pattern (works today with vLLM --enable-lora, where a LoRA module is served under
its own model name). The Tier-2 dense multi-adapter engine implements the same
interface later without touching the Router."""
from __future__ import annotations

import time

import httpx
from pydantic import BaseModel

from photon.config import BackendConfig
from photon.core.contract import RouteTarget


class _Call(BaseModel):
    model_id: str
    adapter_id: str | None


class MockServingBackend:
    def __init__(self, canned: dict[str, dict]):
        self._canned = canned
        self.calls: list[_Call] = []

    async def generate(self, target: RouteTarget, payload: dict) -> tuple[dict, float]:
        self.calls.append(_Call(model_id=target.model_id, adapter_id=target.adapter_id))
        return self._canned[target.model_id], 0.0


class VLLMServingBackend:
    def __init__(self, backends: dict[str, BackendConfig], client: httpx.AsyncClient):
        self._backends = backends
        self._client = client

    async def generate(self, target: RouteTarget, payload: dict) -> tuple[dict, float]:
        backend = self._backends[target.model_id]
        # when an adapter is selected, vLLM serves it under the adapter name
        served_model = target.adapter_id or backend.model
        body = {**payload, "model": served_model}
        started = time.perf_counter()
        resp = await self._client.post(f"{backend.base_url}/chat/completions", json=body)
        latency_ms = (time.perf_counter() - started) * 1000
        resp.raise_for_status()
        return resp.json(), latency_ms
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_serving.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add photon/core/serving.py tests/core/test_serving.py
git commit -m "feat(core): ServingBackend impls (mock + adapter-aware thin vLLM)"
```

---

### Task 10: Fleet API — POST /photon/v1/fleet + dynamic status

**Files:**
- Modify: `photon/api/app.py` (add a FleetManager + current-plan holder to state)
- Modify: `photon/api/admin.py` (POST /fleet applies a FleetSpec; extend /fleet/status with residency)
- Test: `tests/core/test_api_fleet.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_api_fleet.py
def test_apply_fleet_and_read_residency(client):
    spec = {
        "base_models": ["qwen-1.5b", "qwen-14b"],
        "adapters": [
            {"name": "legal-v3", "base": "qwen-1.5b", "pinned": True},
            {"name": "fin-v2", "base": "qwen-1.5b", "pinned": False},
        ],
        "slot_capacity": 3,
    }
    r = client.post("/photon/v1/fleet", json=spec)
    assert r.status_code == 200
    plan = r.json()
    assert "legal-v3" in plan["resident_adapters"]  # pinned resident under pressure

    # dynamic status now reflects the applied plan's residency
    status = client.get("/photon/v1/fleet/status").json()
    assert status["residency"]["resident_adapters"] == plan["resident_adapters"]
    assert status["residency"]["paged_adapters"] == plan["paged_adapters"]


def test_apply_invalid_fleet_is_422(client):
    bad = {"base_models": ["qwen-1.5b"],
           "adapters": [{"name": "x", "base": "nope", "pinned": False}],
           "slot_capacity": 3}
    r = client.post("/photon/v1/fleet", json=bad)
    assert r.status_code == 422


def test_status_residency_null_before_any_apply(client):
    # a fresh app that hasn't had a fleet applied reports null residency, not a crash
    status = client.get("/photon/v1/fleet/status").json()
    assert status["residency"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_api_fleet.py -v`
Expected: FAIL — `/photon/v1/fleet` returns 404 (endpoint not yet added)

- [ ] **Step 3: Wire a FleetManager into app state.** In `photon/api/app.py`, add the import with the other photon imports:

```python
from photon.core.fleet import FleetManager
```

and after `app.state.registry = RegistryStore(registry_db)` add:

```python
    app.state.fleet_manager = FleetManager()
    app.state.fleet_plan = None  # set by POST /photon/v1/fleet
```

- [ ] **Step 4: Add the endpoints in `photon/api/admin.py`.** Add imports at the top:

```python
from photon.core.fleet import FleetSpec
```

Add the apply endpoint (anywhere after `admin_router` is defined):

```python
@admin_router.post("/fleet")
async def apply_fleet(request: Request, spec: FleetSpec):
    try:
        plan = request.app.state.fleet_manager.plan(spec)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    request.app.state.fleet_plan = plan
    return plan.model_dump()
```

Then extend the existing `fleet_status` function's return dict to include residency. Change its `return {...}` to add a `"residency"` key:

```python
    plan = request.app.state.fleet_plan
    return {
        "backends": backends,
        "routing": {
            "default_backend": routing.default_backend,
            "aliases": routing.aliases,
            "canary": routing.canary.model_dump() if routing.canary else None,
            "shadow_enabled": routing.shadow.enabled,
        },
        "residency": plan.model_dump() if plan else None,
    }
```

(Replace the existing `return {"backends": ..., "routing": {...}}` block with the above — the `backends`/`routing` construction earlier in the function is unchanged.)

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/core/test_api_fleet.py -v`
Expected: 3 passed

- [ ] **Step 6: Run the full suite**

Run: `pytest -q`
Expected: all pass (91 Ops + 0..N Core; count = 91 + sum of Core tests: 2+3+3+5+3+2+3+3+3+3 = 30 new → 121 passed)

- [ ] **Step 7: Commit**

```bash
git add photon/api/app.py photon/api/admin.py tests/core/test_api_fleet.py
git commit -m "feat(core): POST /photon/v1/fleet apply + dynamic residency on /fleet/status"
```

---

### Task 11: Shadow-router integration into the request path

**Files:**
- Modify: `photon/api/app.py` (build a ShadowRouter into state, gated by env)
- Modify: `photon/api/chat.py` (observe after the real decision; log-only)
- Test: `tests/core/test_shadow_integration.py`

The learned router runs in shadow behind the live static router: after Ops resolves the actual backend, we extract features and record the counterfactual. Gated by `PHOTON_SHADOW_ROUTER=1` (default off) so it is opt-in and cannot affect the default path.

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_shadow_integration.py
import httpx
import respx

from photon.config import PhotonConfig
from photon.core.contract import RouteTarget
from photon.core.policy import PolicyModel
from photon.core.router import LearnedRouter, ShadowRouter

CHAT_RESPONSE = {
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


class StubPolicy(PolicyModel):
    def predict_acceptable(self, features):
        return 0.99  # would always route cheap


@respx.mock
def test_shadow_router_logs_but_does_not_change_backend(tmp_path):
    from fastapi.testclient import TestClient

    from photon.api.app import create_app
    from tests.test_config import VALID

    respx.post("http://big.test/v1/chat/completions").mock(return_value=httpx.Response(200, json=CHAT_RESPONSE))
    cfg = PhotonConfig.model_validate(VALID)
    app = create_app(config=cfg, db_path=str(tmp_path / "t.db"), registry_db=str(tmp_path / "r.db"))
    # inject a shadow router that would always pick "cheap"
    logged = []
    learned = LearnedRouter(StubPolicy(), 0.6, RouteTarget(model_id="cheap"), RouteTarget(model_id="big"))
    app.state.shadow_router = ShadowRouter(learned, sink=logged.append)

    with TestClient(app) as c:
        r = c.post("/v1/chat/completions",
                   json={"model": "photon-auto", "messages": [{"role": "user", "content": "hi"}]},
                   headers={"X-Photon-Tenant": "px"})
    assert r.status_code == 200
    assert r.headers["X-Photon-Backend"] == "big"  # UNCHANGED by shadow
    assert len(logged) == 1
    assert logged[0].actual_backend == "big"
    assert logged[0].would_route.model_id == "cheap"


def test_no_shadow_router_is_a_noop(client):
    # default app has no shadow_router set → request path works normally
    import respx as _respx
    import httpx as _httpx
    with _respx.mock:
        _respx.post("http://big.test/v1/chat/completions").mock(return_value=_httpx.Response(200, json=CHAT_RESPONSE))
        r = client.post("/v1/chat/completions",
                        json={"model": "photon-auto", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_shadow_integration.py -v`
Expected: FAIL — `AttributeError` on `app.state.shadow_router` usage in chat.py (not yet referenced) or the counterfactual isn't logged

- [ ] **Step 3: Default `shadow_router` to None in `photon/api/app.py`.** After `app.state.fleet_plan = None` add:

```python
    app.state.shadow_router = None  # set to a ShadowRouter to enable shadow logging
```

- [ ] **Step 4: Observe in `photon/api/chat.py`.** In `chat_completions`, immediately after `backend = decision.backend` and before the `record = RequestRecord(...)` construction, add:

```python
    shadow = getattr(state, "shadow_router", None)
    if shadow is not None:
        from photon.core.features import extract_features

        feats = extract_features(
            messages=payload.get("messages", []),
            tenant=tenant,
            route_hint=photon["route"],
        )
        shadow.observe(actual_backend_name=backend.name, features=feats, request_id=request_id)
```

This runs after the real routing decision and only logs — it never reassigns `backend`.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/core/test_shadow_integration.py -v`
Expected: 2 passed

- [ ] **Step 6: Run the full suite**

Run: `pytest -q`
Expected: 123 passed (121 + 2)

- [ ] **Step 7: Commit**

```bash
git add photon/api/app.py photon/api/chat.py tests/core/test_shadow_integration.py
git commit -m "feat(core): shadow-router observation in the request path (log-only, opt-in)"
```

---

## Post-plan work (Tier 2 / Tier 3 — not code tasks here)

1. **Tier 3 — validate the routing bet.** Once Praxiom is live on Photon (Ops post-plan item), collect real (features → cheap-acceptable → cost) rows from the shadow logs, run `ReplayHarness` on them, and check the §4.2 DoD (≥40% routable to ≥5× cheaper at flat quality). Go/no-go on turning the router live (Phase C1).
2. **Tier 2 — the dense serving engine.** The founding inference engineer implements a third `ServingBackend` (patched vLLM fork → S-LoRA, per D-F1) behind the same interface, plus Triton batched-adapter kernels and the on-GPU pipeline orchestrator. DoD from parent §4.1/§9. The `FleetManager.plan()` output becomes load-bearing (enacted, not advisory).
3. **Going live (Phase C1).** Replace `ShadowRouter.observe` with a decide the request path honors, gated behind canary + instant rollback (both already in Ops), only after step 1 passes.

## Explicitly deferred (gated, per spec §1)
- Dense multi-adapter paging, Triton kernels, on-GPU KV reuse — Tier 2 (hire + GPUs).
- The ≥40% savings claim — Tier 3 (real traffic). C0 builds the machinery and measures on synthetic/replay only; no production savings claim.
- Prompt-embedding features — D-R1 deferred; add only if replay data shows cheap features leave signal on the table.

---

## Self-Review

**Spec coverage (04 spec §1 Tier-1 items):** Router feature extraction ✓ (Task 2), policy model ✓ (Task 3), cascade controller + critic hook ✓ (Task 4), router-regret ✓ (Task 5), offline learning loop ✓ (Task 6), shadow-mode router behind resolve-shaped role ✓ (Task 7, 11), Fabric placement logic ✓ (Task 8), ServingBackend interface + Mock + thin-vLLM ✓ (Task 9), POST /fleet + dynamic /fleet/status ✓ (Task 10), the §5 contract ✓ (Task 1). Tier-2/Tier-3 correctly excluded and listed as post-plan. Shadow-only guarantee enforced structurally (Task 7 has no override path; Task 11 only logs).

**Placeholder scan:** no TBDs; every code step has complete code; error paths written (fleet 422 on unknown base, untrained policy fails closed to 0.0, empty-replay and empty-fleet handled).

**Type consistency:** `RouteTarget(model_id, adapter_id, decode_params)` identical across contract/cascade/router/serving/tests. `RequestFeatures.to_vector()` fixed 4-element order used identically by `PolicyModel.fit/predict`. `CascadeController(policy, threshold, cheap, big)` signature matches all call sites (cascade tests, learning harness, LearnedRouter). `ShadowRouter(learned, sink)` + `observe(actual_backend_name, features, request_id)` consistent between Task 7 and Task 11. `FleetManager().plan(FleetSpec) -> PlacementPlan` consistent between Task 8 and Task 10. Test-count arithmetic: Ops 91 + Core (2+3+3+5+3+2+3+3+3+3 = 30) = 121 at Task 10, +2 = 123 at Task 11 — matches the expected-output lines.
