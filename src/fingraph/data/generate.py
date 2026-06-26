"""Command-line entry point for generating a synthetic FinGraph dataset.

Examples:
    python -m fingraph.data.generate
    python -m fingraph.data.generate --accounts 1000 --legit 12000 --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fingraph.data.generator import GeneratorConfig, generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic transaction dataset.")
    parser.add_argument("--accounts", type=int, default=500, help="Number of accounts.")
    parser.add_argument("--legit", type=int, default=6000, help="Number of ordinary transactions.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--out", type=Path, default=Path("data"), help="Output directory.")
    args = parser.parse_args()

    config = GeneratorConfig(
        num_accounts=args.accounts,
        num_legit_transactions=args.legit,
        seed=args.seed,
    )
    accounts_df, tx_df = generate_dataset(config)

    args.out.mkdir(parents=True, exist_ok=True)
    accounts_path = args.out / "accounts.csv"
    tx_path = args.out / "transactions.csv"
    accounts_df.to_csv(accounts_path, index=False)
    tx_df.to_csv(tx_path, index=False)

    flagged = int(tx_df["is_laundering"].sum())
    print(f"Wrote {len(accounts_df):,} accounts      -> {accounts_path}")
    print(f"Wrote {len(tx_df):,} transactions -> {tx_path}")
    print(f"  {flagged:,} belong to planted laundering patterns "
          f"({flagged / len(tx_df):.1%} of all transactions)")


if __name__ == "__main__":
    main()