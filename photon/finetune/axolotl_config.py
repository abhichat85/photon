# photon/finetune/axolotl_config.py
"""Generate Axolotl QLoRA configs from a typed job spec.

Evaluation is deliberately NOT delegated to Axolotl (val_set_size: 0.0):
candidate adapters are judged by Photon's own golden-set harness and the
registry's eval reports, so the quality bar is identical for fine-tunes
and base-model changes."""
from pathlib import Path

import yaml
from pydantic import BaseModel


class LoraJobSpec(BaseModel):
    base_model: str  # HF model id
    train_path: str  # prepared train.jsonl (from dataprep)
    output_dir: str
    wandb_project: str | None = None
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    num_epochs: int = 3
    learning_rate: float = 2e-4
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    sequence_len: int = 2048


def build_axolotl_config(spec: LoraJobSpec) -> dict:
    cfg = {
        "base_model": spec.base_model,
        "load_in_4bit": True,
        "adapter": "qlora",
        "datasets": [
            {"path": spec.train_path, "type": "chat_template", "field_messages": "messages"}
        ],
        "val_set_size": 0.0,
        "sequence_len": spec.sequence_len,
        "sample_packing": True,
        "lora_r": spec.lora_r,
        "lora_alpha": spec.lora_alpha,
        "lora_dropout": spec.lora_dropout,
        "lora_target_linear": True,
        "gradient_accumulation_steps": spec.gradient_accumulation_steps,
        "micro_batch_size": spec.micro_batch_size,
        "num_epochs": spec.num_epochs,
        "learning_rate": spec.learning_rate,
        "optimizer": "adamw_torch",
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.05,
        "bf16": "auto",
        "logging_steps": 10,
        "saves_per_epoch": 1,
        "output_dir": spec.output_dir,
    }
    if spec.wandb_project:
        cfg["wandb_project"] = spec.wandb_project
    return cfg


def write_config(spec: LoraJobSpec, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(build_axolotl_config(spec), f, sort_keys=False)
