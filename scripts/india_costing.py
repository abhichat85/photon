# scripts/india_costing.py
"""Print rupee unit economics from your provider book.

    python -m scripts.india_costing config/india_providers.yaml

Exits 1 if nothing is priced — a book of placeholders is not a cost model, and
printing an empty table with a zero would be worse than saying so."""
import sys

from photon.india.costing import ProviderBook, cost_table, rent_vs_own_breakeven_hours
from photon.india.inr import format_inr


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "config/india_providers.yaml"
    book = ProviderBook.from_yaml(path)
    rows = cost_table(book)

    rate = book.rate()
    print(f"FX: {rate.usd_to_inr} ₹/USD ({rate.source})" if rate else "FX: UNSET (set fx.usd_to_inr + source)")
    print(f"GST: {book.gst_rate:.0%}\n")

    if not rows:
        print("No fully-priced providers.", file=sys.stderr)
        print("Every entry needs BOTH inr_per_hour (your quote) and", file=sys.stderr)
        print("measured_tokens_per_second (scripts/benchmark.py). Placeholders are", file=sys.stderr)
        print("excluded on purpose — a derived price from a guess is not a price.", file=sys.stderr)
        sys.exit(1)

    width = max(len(r.name) for r in rows)
    print(f"{'provider'.ljust(width)}  {'residency':<10}  {'₹/1M tok':>12}  {'incl GST':>12}")
    for r in rows:
        print(f"{r.name.ljust(width)}  {r.data_residency:<10}  "
              f"{format_inr(r.inr_per_1m_tokens):>12}  {format_inr(r.inr_per_1m_tokens_with_gst):>12}")

    cheapest, dearest = rows[0], rows[-1]
    if len(rows) > 1 and cheapest.inr_per_1m_tokens > 0:
        factor = dearest.inr_per_1m_tokens / cheapest.inr_per_1m_tokens
        print(f"\nSpread: {cheapest.name} is {factor:.2f}x cheaper than {dearest.name}.")

    own = next((p for p in book.providers if p.name == "colo-owned" and p.is_priced), None)
    rent = next((p for p in book.providers if p.name != "colo-owned" and p.is_priced), None)
    if own and rent:
        hours = rent_vs_own_breakeven_hours(rent.inr_per_hour, own.inr_per_hour)
        if hours is None:
            print("Rent vs own: renting wins at every utilisation on these numbers.")
        else:
            print(f"Rent vs own: owning wins above ~{hours:.0f} h/month "
                  f"({hours / 730:.0%} utilisation) vs {rent.name}.")


if __name__ == "__main__":
    main()
