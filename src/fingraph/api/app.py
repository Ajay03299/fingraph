"""FinGraph REST API.

A thin service over the analysis pipeline. The pipeline is built once when the
app starts; every endpoint after that is a cheap read. Endpoints expose the
ranked alert queue, a single account's full investigation, and the overall
detection metrics.

Run with:
    uvicorn fingraph.api.app:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from fingraph.pipeline import Pipeline

pipeline = Pipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the analysis once at startup so requests don't trigger recomputation.
    pipeline.build()
    yield


app = FastAPI(
    title="FinGraph",
    description="Graph-based detection and explanation of suspicious transactions.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict[str, float]:
    return pipeline.metrics()


@app.get("/alerts")
def alerts(top_n: int = 25) -> list[dict]:
    return pipeline.alerts(top_n=top_n).to_dict(orient="records")


@app.get("/accounts/{account_id}")
def account(account_id: str) -> dict:
    if account_id not in pipeline.graph:
        raise HTTPException(status_code=404, detail=f"Unknown account: {account_id}")
    return pipeline.investigation(account_id).to_dict()
