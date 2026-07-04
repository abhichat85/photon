# Photon GPU Smoke Test

Prereqs: a host with 1–2 NVIDIA GPUs (L40S/A100 class), Docker + NVIDIA
Container Toolkit installed, HF_TOKEN exported if the models are gated.

## 1. Bring up the stack

    cd photon/deploy
    docker compose up -d --build
    docker compose logs -f vllm-big     # wait for "Uvicorn running on ..."

Model download on first boot takes minutes; watch the logs.

## 2. Verify backends directly

    curl -s http://localhost:8000/v1/models | python3 -m json.tool
    curl -s http://localhost:8001/v1/models | python3 -m json.tool

Expected: each lists its model id.

## 3. Chat through Photon (non-streaming)

    curl -s http://localhost:8080/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -H 'X-Photon-Tenant: smoke' \
      -d '{"model": "photon-auto", "messages": [{"role": "user", "content": "Say hi in 5 words."}]}' \
      | python3 -m json.tool

Expected: a completion; response headers include X-Photon-Backend: big
(add -i to see headers).

## 4. Streaming

    curl -N http://localhost:8080/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -H 'X-Photon-Tenant: smoke' \
      -d '{"model": "photon-auto", "stream": true, "messages": [{"role": "user", "content": "Count to 5."}]}'

Expected: SSE chunks ending with `data: [DONE]`.

## 5. Telemetry, cost, shadow

    curl -s 'http://localhost:8080/photon/v1/costs?tenant=smoke' | python3 -m json.tool
    curl -s 'http://localhost:8080/photon/v1/routing/decisions?tenant=smoke&limit=5' | python3 -m json.tool

Expected: cost summary shows requests > 0 with non-zero cost_usd and
(for short prompts) non-zero shadow_est_cost_usd; decisions list each
request with routed_backend and token counts.

## 6. Latency sanity

    time curl -s http://localhost:8080/v1/chat/completions ... (as step 3)

Compare against hitting vllm-big directly (step 2 URL with the same
payload): the Photon overhead should be single-digit milliseconds.
