# FinGraph

A graph-based financial intelligence system that detects suspicious transaction
patterns, surfaces unusual account relationships, and explains *why* an account
was flagged — producing investigator-ready summaries instead of opaque scores.

## Why this matters

Money laundering rarely looks suspicious one transaction at a time. It hides in
*structure*: rapid movement through chains of accounts, funds that fan out and
quietly reconverge, tight cycles that return money to its source. Modeling
transactions as a graph makes these patterns visible in a way that row-by-row
analysis cannot — and pairing detection with explanation is what turns an alert
into something an investigator can actually act on.

## Status

Early development. Built in clean, reviewable stages — see commit history.

## Tech stack

Python · pandas · NetworkX · scikit-learn · pydantic · FastAPI · Streamlit

## Setup

```bash
git clone https://github.com/<you>/fingraph.git
cd fingraph
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## License

MIT