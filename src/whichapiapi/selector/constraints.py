"""Hard constraints on offers (ARCHITECTURE §2 selector). Unknown values are never silently treated as OK:
in `strict` mode they exclude the offer, otherwise the offer is kept with a warning."""

from __future__ import annotations

from pydantic import BaseModel, Field

from whichapiapi.schema.offer import Offer


class Constraints(BaseModel):
    channel_types: list[str] | None = None  # allowed, e.g. ["official", "aggregator"]
    exclude_user_channels: bool = False  # drop BYO / reseller channels (visibility=user)
    require_features: list[str] = Field(default_factory=list)  # "tools", "json_schema", "vision", "reasoning"
    min_context: int | None = None
    max_input_per_1m: float | None = None
    max_output_per_1m: float | None = None
    payment_method: str | None = None  # substring match against billing.payment_methods, e.g. "crypto"
    no_kyc: bool = False
    zero_retention: bool = False
    no_training_on_data: bool = False
    region: str | None = None  # must be in access.region_hosting
    min_uptime: float | None = None  # 0..1
    strict: bool = False


class Checked(BaseModel):
    offer: Offer
    warnings: list[str] = Field(default_factory=list)


def check(o: Offer, c: Constraints) -> tuple[bool, list[str], str | None]:
    """→ (keep, warnings, rejection reason)."""
    warn: list[str] = []

    def unknown(what: str) -> str | None:
        if c.strict:
            return f"{what} unknown"
        warn.append(f"{what} unknown")
        return None

    if c.channel_types is not None and o.channel_type not in c.channel_types:
        return False, warn, f"channel type {o.channel_type}"
    if c.exclude_user_channels and o.visibility == "user":
        return False, warn, "user/BYO channel"
    missing = [f for f in c.require_features if f not in o.features]
    if missing:
        if o.features:
            return False, warn, f"missing {missing}"
        if r := unknown("features"):
            return False, warn, r
    if c.min_context is not None:
        if o.context_window is None:
            if r := unknown("context window"):
                return False, warn, r
        elif o.context_window < c.min_context:
            return False, warn, f"context {o.context_window}"
    p = o.price
    for cap, value, what in (
        (c.max_input_per_1m, p.input_per_1m, "input price"),
        (c.max_output_per_1m, p.output_per_1m, "output price"),
    ):
        if cap is None:
            continue
        if value is None:
            if r := unknown(what):
                return False, warn, r
        elif value > cap:
            return False, warn, what
    if c.payment_method:
        methods = [m.lower() for m in o.billing.payment_methods]
        if not methods:
            if r := unknown("payment methods"):
                return False, warn, r
        elif not any(c.payment_method.lower() in m for m in methods):
            return False, warn, f"no {c.payment_method} payment"
    if c.no_kyc:
        if o.billing.kyc_required is None:
            if r := unknown("KYC"):
                return False, warn, r
        elif o.billing.kyc_required:
            return False, warn, "KYC required"
    if c.zero_retention:
        if o.access.data_retention == "unknown":
            if r := unknown("data retention"):
                return False, warn, r
        elif o.access.data_retention != "zero":
            return False, warn, f"retention {o.access.data_retention}"
    if c.no_training_on_data:
        if o.conditions.trains_on_data is None:
            if r := unknown("training on data"):
                return False, warn, r
        elif o.conditions.trains_on_data:
            return False, warn, "trains on your data"
    if c.region:
        if not o.access.region_hosting:
            if r := unknown("hosting region"):
                return False, warn, r
        elif c.region.lower() not in [x.lower() for x in o.access.region_hosting]:
            return False, warn, f"not hosted in {c.region}"
    if c.min_uptime is not None:
        if o.reliability.uptime_30d is None:
            if r := unknown("uptime"):
                return False, warn, r
        elif o.reliability.uptime_30d < c.min_uptime:
            return False, warn, f"uptime {o.reliability.uptime_30d:.1%}"
    return True, warn, None


def apply(offers: list[Offer], c: Constraints) -> tuple[list[Checked], dict[str, int]]:
    kept: list[Checked] = []
    rejected: dict[str, int] = {}
    for o in offers:
        ok, warn, why = check(o, c)
        if ok:
            kept.append(Checked(offer=o, warnings=warn))
        else:
            rejected[why or "?"] = rejected.get(why or "?", 0) + 1
    return kept, rejected
