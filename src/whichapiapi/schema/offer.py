"""Offer: one concrete way to buy a capability (model/API x channel x terms)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, computed_field

ChannelType = Literal["official", "aggregator", "authorized_reseller", "user_added"]
Confidence = Literal["high", "medium", "low"]


class Provenance(BaseModel):
    source: str  # URL or adapter id the value came from
    verified_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    confidence: Confidence = "medium"
    note: str | None = None


class Price(BaseModel):
    """Prices in `currency`. Token prices per 1M tokens; `per_unit` for non-token APIs (minute, request, 1k searches)."""

    input_per_1m: float | None = None
    output_per_1m: float | None = None
    cache_read_per_1m: float | None = None
    cache_write_per_1m: float | None = None
    per_request: float | None = None
    per_unit: float | None = None
    unit: str | None = None
    currency: str = "USD"

    @property
    def is_known(self) -> bool:
        return any(
            v is not None for v in (self.input_per_1m, self.output_per_1m, self.per_request, self.per_unit)
        )


class OffPeak(BaseModel):
    window: str  # "16:30-00:30"
    tz: str = "UTC"
    discount: float  # fraction off, 0.5 = half price


class FreeTier(BaseModel):
    rpd: int | None = None
    tokens_per_day: int | None = None
    requires_credits: bool | None = None
    rpm: int | None = None
    tpm: int | None = None
    monthly_tokens: str | None = None  # as the source states it, e.g. "~25M"
    resets: str | None = None  # "utc_midnight", "pacific_midnight", …
    notes: list[str] = Field(default_factory=list)  # provider quirks worth knowing before relying on it


class Conditions(BaseModel):
    off_peak: OffPeak | None = None
    batch_discount: float | None = None  # fraction off for batch API
    batch_max_latency_h: float | None = None
    trains_on_data: bool | None = None  # True if prompts may be used for training at this price
    data_sharing_discount: float | None = None
    free_tier: FreeTier | None = None
    byok_fee: float | None = None  # fraction


class Billing(BaseModel):
    min_topup: float | None = None
    topup_fee: float | None = None  # fraction
    credits_expire_days: int | None = None
    payment_methods: list[str] = Field(default_factory=list)
    kyc_required: bool | None = None
    currency: str = "USD"
    fx_note: str | None = None


class Access(BaseModel):
    geo_blocked: list[str] = Field(default_factory=list)
    data_retention: Literal["zero", "30d", "unknown"] | str = "unknown"
    region_hosting: list[str] = Field(default_factory=list)
    commercial_use: bool | None = None


class Reliability(BaseModel):
    uptime_30d: float | None = None
    rate_limit: str | None = None
    ttft_ms: float | None = None  # time to first token, as measured by the source
    throughput_tps: float | None = None  # output tokens per second, as measured by the source


class Endpoint(BaseModel):
    """How to call this offer."""

    protocol: Literal["openai", "anthropic", "http", "none"] = "openai"
    base_url: str | None = None
    model_id: str | None = None  # id to send to the API, if different from `model`
    key_env: str | None = None


class Offer(BaseModel):
    capability: str = "llm.chat"
    model: str  # canonical-ish product/model id, e.g. "deepseek-v4.1-flash"
    vendor: str | None = None  # who makes the model/API ("openai", "deepseek")
    channel: str  # who sells access ("openrouter", "myreseller", "deepseek")
    channel_type: ChannelType = "official"
    group: str | None = None  # sub-channel inside a reseller (new-api group), if any
    price: Price = Field(default_factory=Price)
    conditions: Conditions = Field(default_factory=Conditions)
    billing: Billing = Field(default_factory=Billing)
    access: Access = Field(default_factory=Access)
    reliability: Reliability = Field(default_factory=Reliability)
    endpoint: Endpoint = Field(default_factory=Endpoint)
    context_window: int | None = None
    max_output: int | None = None
    features: list[str] = Field(default_factory=list)  # "tools", "json_schema", "vision", "reasoning"
    provenance: Provenance
    visibility: Literal["public", "user"] = "public"
    verified: bool = True

    @computed_field  # type: ignore[prop-decorator]
    @property
    def id(self) -> str:
        base = f"{self.channel}/{self.model}"
        return f"{base}@{self.group}" if self.group else base
