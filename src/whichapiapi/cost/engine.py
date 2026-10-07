"""Cost engine: price x usage -> money. Pure functions (ARCHITECTURE §2)."""

from __future__ import annotations

from pydantic import BaseModel

from whichapiapi.schema.load_profile import LoadProfile
from whichapiapi.schema.offer import Offer, Price


def call_cost(
    price: Price | None,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    requests: int = 1,
    units: float = 0.0,
) -> float | None:
    """Cost of `requests` calls from their summed usage (default: one call). `output_tokens` already includes
    reasoning tokens (OpenAI semantics); `units` are non-token units such as audio minutes (`Price.per_unit`)."""
    if price is None or not price.is_known:
        return None
    inp = price.input_per_1m or 0.0
    cache_read = price.cache_read_per_1m if price.cache_read_per_1m is not None else inp
    cached = min(cached_tokens, input_tokens)
    total = (
        (input_tokens - cached) * inp / 1e6
        + cached * cache_read / 1e6
        + output_tokens * (price.output_per_1m or 0.0) / 1e6
        + (price.per_request or 0.0) * requests
        + (price.per_unit or 0.0) * units
    )
    return round(total, 8)


class CostBreakdown(BaseModel):
    offer_id: str
    per_request: float
    per_month: float
    list_per_month: float  # same volume at plain list price, no discounts or fees
    notes: list[str]


def monthly_cost(offer: Offer, profile: LoadProfile) -> CostBreakdown | None:
    """Effective monthly cost for a load profile, applying cache, batch, off-peak and billing conditions."""
    p = offer.price
    if not p.is_known:
        return None
    notes: list[str] = []
    cached = int(profile.input_tokens * profile.cached_share)
    base = call_cost(p, profile.input_tokens, profile.output_tokens, cached) or 0.0
    list_price = call_cost(p, profile.input_tokens, profile.output_tokens, 0) or 0.0
    if cached and p.cache_read_per_1m is not None:
        notes.append(f"prompt cache on {profile.cached_share:.0%} of input")

    c = offer.conditions
    discount = 0.0
    if c.batch_discount and profile.batch_share:
        discount += c.batch_discount * profile.batch_share
        notes.append(f"batch {c.batch_discount:.0%} off on {profile.batch_share:.0%} of requests")
    if c.off_peak and profile.off_peak_share:
        remaining = 1 - profile.batch_share
        discount += c.off_peak.discount * min(profile.off_peak_share, remaining)
        notes.append(f"off-peak {c.off_peak.discount:.0%} off ({c.off_peak.window} {c.off_peak.tz})")
    if c.trains_on_data and not profile.allow_training_on_data:
        notes.append("WARNING: this price assumes your data may be used for training")
    per_req = base * (1 - discount)

    fee = offer.billing.topup_fee or 0.0
    if fee:
        per_req *= 1 + fee
        notes.append(f"top-up fee {fee:.1%}")
    return CostBreakdown(
        offer_id=offer.id,
        per_request=round(per_req, 8),
        per_month=round(per_req * profile.requests_per_month, 4),
        list_per_month=round(list_price * profile.requests_per_month, 4),
        notes=notes,
    )
