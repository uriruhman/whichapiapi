from whichapiapi.adapters.base import Adapter, registry
from whichapiapi.adapters.models_dev import ModelsDevAdapter
from whichapiapi.adapters.newapi import NewApiAdapter
from whichapiapi.adapters.openrouter import OpenRouterAdapter

__all__ = ["Adapter", "ModelsDevAdapter", "NewApiAdapter", "OpenRouterAdapter", "registry"]
