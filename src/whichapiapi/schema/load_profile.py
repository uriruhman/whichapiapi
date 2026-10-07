"""How the user will actually consume an offer. Drives the cost engine."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LoadProfile(BaseModel):
    requests_per_month: int = 1000
    input_tokens: int = 2000  # per request
    output_tokens: int = 500
    cached_share: float = Field(0.0, ge=0, le=1)  # share of input tokens served from prompt cache
    batch_share: float = Field(0.0, ge=0, le=1)  # share of requests that tolerate batch latency
    off_peak_share: float = Field(0.0, ge=0, le=1)
    allow_training_on_data: bool = False
