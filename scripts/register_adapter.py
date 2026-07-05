# scripts/register_adapter.py
"""Register (and optionally promote) a trained LoRA adapter.

Usage:
    python -m scripts.register_adapter <registry_db> <name> <base_model> <adapter_path> \
        [--eval-report path.json] [--promote] [--min-pass-rate 1.0] [--force]

Promotion is gated: unless --force, the version must carry an eval report whose
pass rate >= --min-pass-rate (attach one with --eval-report, produced by
scripts/run_evals.py). This makes the golden-set gate a real gate.
"""
import sys

from photon.registry.store import RegistryStore


def _opt(argv: list[str], flag: str, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def main() -> None:
    argv = sys.argv[1:]
    promote = "--promote" in argv
    force = "--force" in argv
    min_pass_rate = float(_opt(argv, "--min-pass-rate", "1.0"))
    eval_report_path = _opt(argv, "--eval-report")

    # positional args are everything that isn't a flag or a flag's value
    consumed = set()
    for flag in ("--eval-report", "--min-pass-rate"):
        if flag in argv:
            i = argv.index(flag)
            consumed.update({i, i + 1})
    for flag in ("--promote", "--force"):
        if flag in argv:
            consumed.add(argv.index(flag))
    positional = [a for i, a in enumerate(argv) if i not in consumed]
    db, name, base_model, adapter_path = positional

    eval_report = open(eval_report_path).read() if eval_report_path else None
    store = RegistryStore(db)
    mv = store.register(name, base_model, adapter_path, eval_report)
    if promote:
        store.promote(name, mv.version, min_pass_rate=min_pass_rate, force=force)
    print(f"registered {name} v{mv.version}" + (" (production)" if promote else " (draft)"))


if __name__ == "__main__":
    main()
