"""Run the full FinGraph pipeline on the IBM AML benchmark.

    python -m fingraph.data.run_benchmark data/benchmark/HI-Small_Trans.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fingraph.data.ibm_loader import IBMSource
from fingraph.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FinGraph on the IBM AML benchmark.")
    parser.add_argument("path", type=Path, help="Path to HI-Small_Trans.csv")
    parser.add_argument("--max-rows", type=int, default=300_000)
    args = parser.parse_args()

    print(f"Loading and analysing up to {args.max_rows:,} rows from {args.path} ...")
    pipeline = Pipeline(source=IBMSource(args.path, max_rows=args.max_rows)).build()

    metrics = pipeline.metrics()
    print("\nFinGraph on IBM AML benchmark (HI-Small)")
    print("-" * 44)
    for name, value in metrics.items():
        print(f"{name:>22}: {value}")

    dirty_rate = metrics["num_dirty_accounts"] / metrics["num_accounts"]
    print(f"\n  dirty-account base rate: {dirty_rate:.2%}")


if __name__ == "__main__":
    main()