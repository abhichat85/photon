# scripts/register_adapter.py
"""Register (and optionally promote) a trained LoRA adapter.

Usage: python -m scripts.register_adapter <registry_db> <name> <base_model> <adapter_path> [--promote]
"""
import sys

from photon.registry.store import RegistryStore


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--promote"]
    promote = "--promote" in sys.argv
    db, name, base_model, adapter_path = args
    store = RegistryStore(db)
    mv = store.register(name, base_model, adapter_path)
    if promote:
        store.promote(name, mv.version)
    print(f"registered {name} v{mv.version}" + (" (production)" if promote else " (draft)"))


if __name__ == "__main__":
    main()
