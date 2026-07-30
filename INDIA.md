# Photon for India

Photon's India build is not a localisation layer bolted onto a US-shaped
inference engine. It exists because three things about the Indian market change
what the *correct engineering* is — and each one is implemented, tested, and
measurable rather than asserted.

---

## 1. Token pricing systematically over-charges Indian languages

This is the load-bearing insight.

LLM pricing is denominated per token, but a token is not a constant amount of
meaning. A Latin-optimised BPE tokenizer fragments Devanagari, Tamil, Telugu,
Bengali and the other Indic scripts into far more tokens per unit of semantic
content than English. The consequence is structural: **the same question, asked
in Hindi, can cost several times what it costs in English — on the same model,
for the same answer.**

Every horizontal router optimises cost-per-token. On Indian traffic that
optimises the wrong quantity, and it does so invisibly.

**What Photon does instead.** It measures chars-per-token per
(backend × script) from live traffic and compares models in a language-fair
unit — **cost per 1,000 characters of content**:

```
GET /photon/v1/india/token-efficiency
→ per backend, per script: chars_per_token, samples, indic_penalty_vs_latin
```

Under that unit, model ranking can invert. A model that is *dearer per token*
can be *cheaper for Hindi* because its tokenizer emits fewer tokens for the same
text. `IndicAwareRouter` takes that arbitrage on Indic requests. The end-to-end
test in `tests/india/test_indic_routing.py` demonstrates exactly this inversion.

**The discipline:** an unmeasured (backend, script) pair returns `None` and is
ineligible for routing. Photon never assumes a tokenizer ratio — including for
models whose vendors publish Indic-efficiency claims. Your ratios, measured on
your traffic.

## 2. The unit of account is the rupee, with GST

An Indian buyer budgets, approves and audits in ₹ — and their finance team needs
the GST split, because input-tax-credit eligibility depends on it being visible.
A dollar-denominated cost dashboard pushes conversion and tax treatment onto the
customer and is unusable in procurement.

- `photon/india/inr.py` — 18% services GST with net/GST/gross separation, Indian
  digit grouping (₹12,34,567 — the Western grouping reads as *wrong* to an
  Indian finance reader), and an FX rate type that **requires provenance**. A
  rate without a source is a guess with a decimal point.
- GPU cost per 1M tokens is **derived**: ₹/hour ÷ *measured* tokens/second.
  Vendor throughput claims are not an input.

## 3. Cheap in-country GPU capacity is the margin

Indian price points don't support US-shaped gross margins on hyperscaler
inference. The margin has to come from cost structure: Indian GPU providers
(E2E Networks, Jarvislabs, Yotta, Neysa), and eventually owned hardware.

```bash
python -m scripts.india_costing config/india_providers.yaml
# ₹/1M tokens per provider, incl GST, cheapest first
# + rent-vs-own break-even utilisation
```

`config/india_providers.example.yaml` ships as **placeholders only** — and a
test enforces that it produces no quotable price. Fill it from your own quotes
and your own benchmark run. Also worth checking: IndiaAI Mission subsidised
compute via empanelled providers can materially change these economics for
qualifying Indian entities.

## 4. "Does our data leave India?" — answered with a control, not a policy PDF

The most common blocker in Indian enterprise and public-sector AI deals. Usually
answered with a document. Photon answers it with enforcement:

- A tenant with a residency policy **cannot** be served by a non-permitted
  backend — the request is rejected with `451` before any bytes leave the
  gateway (`tests/india/test_residency.py` asserts zero upstream calls).
- `GET /photon/v1/india/residency?tenant=…` emits a machine-readable attestation
  generated from live config — the artifact for a security questionnaire.
- **`country` is declared, never inferred from region names.** AWS Mumbai is
  `ap-south-1`, Azure's is `centralindia`, GCP's is `asia-south1`; no prefix
  rule classifies those correctly.
- **`country` and `operator_jurisdiction` are separate fields**, because they
  are separate questions. An in-India region operated by a foreign entity
  satisfies data-localisation while leaving the sovereignty question open — and
  sovereignty-sensitive buyers ask about both. Photon lets a tenant require
  either or both.

> Not legal advice. These are engineering controls a compliance position can be
> built on; whether your processing satisfies the DPDP Act 2023, RBI's
> payment-data localisation directive, or a sectoral regulator is a question for
> your counsel.

---

## What is measured vs what is built

Consistent with the rest of this codebase (`DECISIONS.md`): **the benchmark
precedes the number.**

| | Status |
|---|---|
| Script detection, token-efficiency ledger, language-fair routing | **Built + tested** (214 tests) |
| ₹/GST accounting, derived GPU unit economics, rent-vs-own | **Built + tested** |
| Residency enforcement + attestation | **Built + tested** |
| Actual Indic penalty ratios for *your* models | **Unmeasured** — accumulates from live traffic |
| Actual ₹/1M-token costs | **Unmeasured** — needs your quotes + `scripts/benchmark.py` |

No Indic tokenizer ratio, GPU price, or FX rate in this repository is a claim.
They are all slots waiting for a measurement.

## Getting the numbers

1. Bring up a GPU box — `deploy/GPU_SETUP.md` (~₹100–200/hr on an Indian
   provider; the runbook is provider-agnostic).
2. Run `scripts/benchmark.py` for throughput → fill `measured_tokens_per_second`.
3. Fill `inr_per_hour` from your quotes → `scripts/india_costing.py` prints
   real ₹/1M-token economics.
4. Send Indic traffic through the gateway → `/india/token-efficiency` fills in
   with *your* penalty ratios → enable `IndicAwareRouter` once there's enough
   signal (shadow → canary → full, same discipline as the learned router).
