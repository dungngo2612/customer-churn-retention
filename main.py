"""FastAPI application entry point for the churn-retention service."""

from fastapi import FastAPI

from churn_retention.entrypoints.api.routes.datasets import router as datasets_router

app = FastAPI(title="Customer Churn Retention")
app.include_router(datasets_router)


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}
