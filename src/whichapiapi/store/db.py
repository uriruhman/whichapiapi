"""Local store (ADR-0004): offers with history, results cache, runs, spend ledger. SQLAlchemy Core."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
    select,
)
from sqlalchemy.engine import Engine

from whichapiapi.schema.offer import Offer

meta = MetaData()

offers = Table(
    "offers",
    meta,
    Column("id", String, primary_key=True),
    Column("channel", String, index=True),
    Column("model", String, index=True),
    Column("grp", String),
    Column("capability", String, index=True),
    Column("visibility", String),
    Column("data", JSON),
    Column("updated_at", String),
)

offer_versions = Table(
    "offer_versions",
    meta,
    Column("seq", Integer, primary_key=True, autoincrement=True),
    Column("offer_id", String, index=True),
    Column("data", JSON),
    Column("recorded_at", String),
)

cache = Table(
    "cache",
    meta,
    Column("key", String, primary_key=True),
    Column("value", JSON),
    Column("created_at", String),
)

runs = Table(
    "runs",
    meta,
    Column("id", String, primary_key=True),
    Column("suite", Text),
    Column("started_at", String),
    Column("finished_at", String),
    Column("status", String),
    Column("meta", JSON),
)

ledger = Table(
    "ledger",
    meta,
    Column("seq", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String, index=True),
    Column("ts", String),
    Column("channel", String, index=True),
    Column("model", String),
    Column("kind", String),  # candidate | judge | probe
    Column("cost_usd", Float),
    Column("source", String),  # computed | measured
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def default_url() -> str:
    if url := os.environ.get("WHICHAPIAPI_DB_URL"):
        return url
    home = Path(os.environ.get("WHICHAPIAPI_HOME", Path.home() / ".local/share/whichapiapi"))
    home.mkdir(mode=0o700, parents=True, exist_ok=True)  # cache holds outputs, ledger, learned prices
    return f"sqlite:///{home / 'whichapiapi.db'}"


class Store:
    def __init__(self, url: str | None = None):
        self.engine: Engine = create_engine(url or default_url())
        meta.create_all(self.engine)

    # ---------------------------------------------------------- offers
    def upsert_offers(self, items: Iterable[Offer]) -> tuple[int, int]:
        """Insert or update offers; append a version row whenever content changed. Returns (new, changed)."""
        new = changed = 0
        with self.engine.begin() as c:
            for o in items:
                data = o.model_dump(mode="json")
                cmp = _without_volatile(data)
                row = c.execute(select(offers.c.data).where(offers.c.id == o.id)).first()
                if row is None:
                    new += 1
                elif _without_volatile(row.data) == cmp:
                    # same content, seen again: refresh verified_at so re-pulled offers don't turn "stale"
                    c.execute(offers.update().where(offers.c.id == o.id).values(data=data, updated_at=_now()))
                    continue
                else:
                    changed += 1
                    c.execute(offers.delete().where(offers.c.id == o.id))
                c.execute(
                    offers.insert().values(
                        id=o.id,
                        channel=o.channel,
                        model=o.model,
                        grp=o.group,
                        capability=o.capability,
                        visibility=o.visibility,
                        data=data,
                        updated_at=_now(),
                    )
                )
                c.execute(offer_versions.insert().values(offer_id=o.id, data=data, recorded_at=_now()))
        return new, changed

    def find_offers(
        self,
        *,
        model: str | None = None,
        channel: str | None = None,
        capability: str | None = None,
        include_user: bool = True,
        limit: int = 500,
    ) -> list[Offer]:
        q = select(offers.c.data)
        if model:
            # "_" for "." and "-": a model key ("claude-haiku-4.5") also finds "claude-haiku-4-5" spellings
            q = q.where(offers.c.model.like("%" + model.replace(".", "_").replace("-", "_") + "%"))
        if channel:
            q = q.where(offers.c.channel == channel)
        if capability:
            q = q.where(offers.c.capability == capability)
        if not include_user:
            q = q.where(offers.c.visibility == "public")
        with self.engine.connect() as c:
            return [Offer.model_validate(r.data) for r in c.execute(q.limit(limit))]

    def get_offer(self, offer_id: str) -> Offer | None:
        with self.engine.connect() as c:
            row = c.execute(select(offers.c.data).where(offers.c.id == offer_id)).first()
        return Offer.model_validate(row.data) if row else None

    def offer_history(self, offer_id: str) -> list[dict[str, Any]]:
        with self.engine.connect() as c:
            rows = c.execute(
                select(offer_versions.c.data, offer_versions.c.recorded_at)
                .where(offer_versions.c.offer_id == offer_id)
                .order_by(offer_versions.c.seq)
            )
            return [{"recorded_at": r.recorded_at, **r.data} for r in rows]

    # ---------------------------------------------------------- cache
    def cache_get(self, key: str) -> dict[str, Any] | None:
        with self.engine.connect() as c:
            row = c.execute(select(cache.c.value).where(cache.c.key == key)).first()
        return row.value if row else None

    def cache_items(self, prefix: str) -> dict[str, dict[str, Any]]:
        with self.engine.connect() as c:
            rows = c.execute(select(cache.c.key, cache.c.value).where(cache.c.key.like(f"{prefix}%")))
            return {r.key: r.value for r in rows}

    def cache_put(self, key: str, value: dict[str, Any]) -> None:
        with self.engine.begin() as c:
            c.execute(cache.delete().where(cache.c.key == key))
            c.execute(cache.insert().values(key=key, value=value, created_at=_now()))

    # ---------------------------------------------------------- runs & ledger
    def start_run(self, run_id: str, suite: str, meta_: dict[str, Any]) -> None:
        with self.engine.begin() as c:
            c.execute(
                runs.insert().values(id=run_id, suite=suite, started_at=_now(), status="running", meta=meta_)
            )

    def finish_run(self, run_id: str, status: str, meta_: dict[str, Any]) -> None:
        with self.engine.begin() as c:
            c.execute(
                runs.update().where(runs.c.id == run_id).values(finished_at=_now(), status=status, meta=meta_)
            )

    def spend(self, run_id: str, channel: str, model: str, kind: str, cost: float, source: str) -> None:
        with self.engine.begin() as c:
            c.execute(
                ledger.insert().values(
                    run_id=run_id,
                    ts=_now(),
                    channel=channel,
                    model=model,
                    kind=kind,
                    cost_usd=cost,
                    source=source,
                )
            )

    def spent_total(self, *, channel: str | None = None, source: str = "computed") -> float:
        q = select(func.coalesce(func.sum(ledger.c.cost_usd), 0.0)).where(ledger.c.source == source)
        if channel:
            q = q.where(ledger.c.channel == channel)
        with self.engine.connect() as c:
            return float(c.execute(q).scalar_one())

    def real_spent_total(self) -> float:
        """Real money spent by evals: measured rows (per-run) + manual true-ups; excludes after-the-fact
        `reconcile` rows, which re-describe spend already counted."""
        q = select(func.coalesce(func.sum(ledger.c.cost_usd), 0.0)).where(
            ledger.c.source == "measured", ledger.c.kind != "reconcile"
        )
        with self.engine.connect() as c:
            return float(c.execute(q).scalar_one())

    def ledger_rows(self, run_id: str | None = None) -> list[dict[str, Any]]:
        q = select(ledger)
        if run_id:
            q = q.where(ledger.c.run_id == run_id)
        with self.engine.connect() as c:
            return [dict(r._mapping) for r in c.execute(q.order_by(ledger.c.seq))]


def _without_volatile(data: dict[str, Any]) -> str:
    """Content to compare for "changed": no timestamps, and no None fields (a new optional schema field is not a
    change of the offer)."""

    def prune(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: prune(v) for k, v in x.items() if v is not None}
        return [prune(v) for v in x] if isinstance(x, list) else x

    d = json.loads(json.dumps(data))
    d.get("provenance", {}).pop("verified_at", None)
    return json.dumps(prune(d), sort_keys=True)
