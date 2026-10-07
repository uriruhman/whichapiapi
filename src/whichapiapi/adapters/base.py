"""Adapter = one data source → normalised Offers. New sources are one module + an entry point."""

from __future__ import annotations

from collections.abc import Iterable
from importlib.metadata import entry_points
from typing import Protocol

from whichapiapi.schema.offer import Offer


class Adapter(Protocol):
    id: str
    license: str
    attribution: str

    def fetch(self) -> Iterable[Offer]: ...


def registry() -> dict[str, type]:
    return {ep.name: ep.load() for ep in entry_points(group="whichapiapi.adapters")}
