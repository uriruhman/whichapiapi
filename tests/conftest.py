from pathlib import Path

import pytest

from whichapiapi.store.db import Store

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 't.db'}")


@pytest.fixture
def fix():
    return FIX


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Activity logs (and anything else under WHICHAPIAPI_HOME) go to a temp dir, never the user's real one."""
    monkeypatch.setenv("WHICHAPIAPI_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WHICHAPIAPI_FIELD_LOGS", "")  # never read the real Tributary logs


@pytest.fixture(autouse=True)
def _channels(tmp_path, monkeypatch):
    """A neutral custom channel for tests (real custom channels live outside the repo, see whichapiapi.channels)."""
    f = tmp_path / "channels.yaml"
    f.write_text(
        "reseller:\n  base_url: https://api.reseller.test/v1\n  key_env: RESELLER_API_KEY\n  kind: newapi\n"
        "  primary: true\n  judge_model: gpt-6-luna\n  min_balance: 0.30\n"
    )
    monkeypatch.setenv("WHICHAPIAPI_CHANNELS", str(f))
    monkeypatch.delenv("WHICHAPIAPI_JUDGE", raising=False)
