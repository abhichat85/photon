# Solo GPU Setup — the one physical thing only you can do

No hire, no cluster. This is the ~$1–2/hr, ~30-minute path to a real GPU box
serving the Photon stack, so the Tier-2 numbers and the multi-adapter Fabric
stop being theory. Everything Photon-side is already built and tested (148+
tests); this doc is only the parts that cost money or touch hardware.

## 0. What you're proving

- vLLM serving a base model + LoRA adapters at runtime (the Fabric substrate)
- `POST /fleet?enact=true` actually loading/unloading adapters on a live server
- `scripts/benchmark.py` reading real §9 numbers (routing tax, overhead, cost)

## 1. Rent a GPU (pick one — cheapest first)

**RunPod** (fastest, per-second billing):
- runpod.io → Deploy → an RTX 4090 (24GB, ~$0.44/hr) or L40S (48GB, ~$1/hr)
- Template: "RunPod PyTorch" or the official vLLM template
- Expose TCP ports 8000 (vLLM) and 8080 (Photon)

**Lambda / Vast.ai** are equivalent; any box with a recent NVIDIA GPU + Docker.

A 1.5B/7B model fits a 24GB 4090; a 14B needs the L40S/48GB. Start small (7B) —
you're validating the pipeline, not training.

## 2. Bring up vLLM WITH runtime LoRA enabled (this flag is the whole point)

```bash
# on the GPU box
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True   # <-- enables /load_lora_adapter
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --enable-lora --max-loras 4 --max-lora-rank 32 \
  --port 8000
```

Verify: `curl localhost:8000/v1/models` lists the base model.

## 3. Run Photon against it

```bash
# clone + install (host or a second container)
git clone https://github.com/abhichat85/photon && cd photon
pip install -e .
# Edit config/fleet.example.yaml so a backend matches the model you started in
# step 2. BOTH must line up:
#   - base_url  → your vLLM server (e.g. http://localhost:8000/v1)
#   - model     → the EXACT model id vLLM serves (Qwen/Qwen2.5-7B-Instruct)
# then:
PHOTON_CONFIG=config/fleet.example.yaml uvicorn photon.api.app:main_app --factory --port 8080
```

> **The one enactment gotcha:** in step 4 the adapter's `base` MUST equal a
> configured backend's `model` (or its `name`). If it doesn't, `enact` returns
> **200 with `skipped: [...]`** — a silent no-op, not an error. Always check the
> response says `loaded`, not `skipped`.

Run the preflight (checks everything below is wired before you spend time):

```bash
python -m scripts.preflight --gateway http://localhost:8080 --vllm http://localhost:8000/v1
```

## 4. Enact a fleet (load a real adapter at runtime)

Train or download a LoRA for the base, put it on the box, then:

```bash
# `base` here = the model id vLLM serves AND a backend's `model` in your config.
curl -X POST 'http://localhost:8080/photon/v1/fleet?enact=true' -H 'Content-Type: application/json' -d '{
  "base_models": ["Qwen/Qwen2.5-7B-Instruct"],
  "adapters": [{"name": "legal-v3", "base": "Qwen/Qwen2.5-7B-Instruct",
                "pinned": true, "path": "/adapters/legal-v3"}],
  "slot_capacity": 3
}'
# EXPECT: {"plan": {...}, "enactment": {"loaded": ["legal-v3"], "warnings": []}}
# If you see "skipped":["legal-v3"] with a warning about "base ... not served
# here", your config's backend `model` doesn't match "Qwen/Qwen2.5-7B-Instruct"
# — fix step 3 and retry. Re-running a completed enact is safe: it reports
# already_resident, not a duplicate load.
```

`curl localhost:8000/v1/models` now lists `legal-v3` — the adapter is live.

## 5. Measure the §9 numbers

```bash
python -m scripts.benchmark \
  --gateway http://localhost:8080 \
  --backend http://localhost:8000/v1 --backend-model Qwen/Qwen2.5-7B-Instruct \
  --requests 200 \
  --baseline-cost-usd 0.030 --measured-cost-usd 0.010
```

Exit 0 = the spec's targets are met on real hardware. That is the moment Tier-2
stops being a claim. Record the JSON in DECISIONS or a benchmark log.

## 6. Tear down (stop paying)

Destroy the pod/instance. Total spend for a validation run: a few dollars.

## When native vLLM LoRA isn't enough

If `benchmark.py` shows the routing tax or density falling short at your target
adapter count, THAT measurement is the trigger for the fork/kernel work
(DECISIONS.md D13) — not before. Don't fork on speculation; fork on a number.
