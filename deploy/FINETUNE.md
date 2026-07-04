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
representative traffic. Then:

    python -m scripts.register_adapter registry.db ... --promote   # or promote() in code

flip the alias/default to the new backend, and remove the canary block.
Rollback = re-promote the previous registry version and flip the config back.
