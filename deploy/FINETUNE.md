# Fine-tune → Register → Canary → Promote

The end-to-end adapter lifecycle. Steps 1 and 3–6 run anywhere; step 2 runs
on a GPU training box (A100/L40S class).

## 1. Prepare data

    python -m photon.finetune.dataprep raw_conversations.jsonl prepared/

Review the printed PrepReport (dropped_invalid / dropped_duplicate counts).
For enterprise engagements, add domain-specific PII scrubbing (Presidio)
before this step.

## 2. Train (GPU box)

    pip install axolotl        # training box only — never on the gateway
    python -c "
    from photon.finetune.axolotl_config import LoraJobSpec, write_config
    write_config(LoraJobSpec(
        base_model='Qwen/Qwen2.5-1.5B-Instruct',
        train_path='prepared/train.jsonl',
        output_dir='out/praxiom-intent-v1',
        wandb_project='photon-ft',
    ), 'job.yaml')"
    accelerate launch -m axolotl.cli.train job.yaml

The LoRA adapter lands in out/praxiom-intent-v1/.

## 3. Register the adapter

    python -m scripts.register_adapter registry.db praxiom-intent \
        Qwen/Qwen2.5-1.5B-Instruct out/praxiom-intent-v1

## 4. Serve as a candidate backend

Add the adapter to the vLLM service (compose or Helm values):

    command: ["--model", "Qwen/Qwen2.5-1.5B-Instruct", "--port", "8001",
              "--enable-lora",
              "--lora-modules", "praxiom-intent-v1=/adapters/praxiom-intent-v1"]

and mount the adapter directory. Then add a backend to the Photon fleet
config with `model: praxiom-intent-v1` (vLLM serves the LoRA module under
that name) and its own pricing entry.

## 5. Gate on the golden set

    python -m scripts.run_evals http://localhost:8080 evals/praxiom_golden.yaml \
        --model praxiom-intent-cand --min-pass-rate 1.0

Exit code 1 = regression -> stop, retrain or adjust. Store the report:

    python - <<'EOF'
    # attach the eval JSON to the registry version
    from photon.registry.store import RegistryStore
    RegistryStore("registry.db").attach_eval("praxiom-intent", 1, open("report.json").read())
    EOF

## 6. Canary, then promote

Set in the fleet config:

    routing:
      canary: {backend: praxiom-intent-cand, weight: 0.1}

Watch Grafana (error rate, p95 latency, cost/hr) for at least a day of
representative traffic. Then promote — promotion is GATED: it refuses unless the
version carries an eval report meeting the pass-rate threshold (attach the
report.json from step 5):

    python -m scripts.register_adapter registry.db praxiom-intent \
        Qwen/Qwen2.5-1.5B-Instruct out/praxiom-intent-v1 \
        --eval-report report.json --promote --min-pass-rate 1.0

To promote without a passing eval (deliberate operational override only):
add --force. The gate is enforced in registry.promote(), so the HTTP endpoint
(POST /photon/v1/adapters with promote=true) and any code path are covered too.

flip the alias/default to the new backend, and remove the canary block.
Rollback = re-promote the previous registry version and flip the config back.

## 7. Continuous drift monitoring (production)

The promotion gate is point-in-time. In production, schedule the drift check
(k8s CronJob / cron) to catch regressions in the live model:

    python -m scripts.drift_check https://gateway.internal evals/praxiom_golden.yaml \
        --min-pass-rate 0.95

It updates the photon_golden_pass_rate gauge (push to a Pushgateway via
PUSHGATEWAY_URL) which drives the PhotonGoldenQualityDrift alert.
