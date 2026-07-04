# tests/test_axolotl_config.py
import yaml

from photon.finetune.axolotl_config import LoraJobSpec, build_axolotl_config, write_config

SPEC = LoraJobSpec(
    base_model="Qwen/Qwen2.5-1.5B-Instruct",
    train_path="prepared/train.jsonl",
    output_dir="out/praxiom-intent-v1",
)


def test_core_keys_present():
    cfg = build_axolotl_config(SPEC)
    assert cfg["base_model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg["adapter"] == "qlora"
    assert cfg["datasets"] == [
        {"path": "prepared/train.jsonl", "type": "chat_template", "field_messages": "messages"}
    ]
    assert cfg["lora_r"] == 16
    assert cfg["output_dir"] == "out/praxiom-intent-v1"
    assert "wandb_project" not in cfg


def test_wandb_project_included_when_set():
    cfg = build_axolotl_config(SPEC.model_copy(update={"wandb_project": "photon-ft"}))
    assert cfg["wandb_project"] == "photon-ft"


def test_write_config_yaml_roundtrip(tmp_path):
    path = tmp_path / "job.yaml"
    write_config(SPEC, path)
    assert yaml.safe_load(path.read_text())["num_epochs"] == 3
