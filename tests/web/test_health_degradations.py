"""/health surfaces active degraded-mode signals for monitoring."""

from __future__ import annotations

import asyncio
from importlib import import_module

import pytest

from xagent.web.services.ops_signals import (
    GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED,
    clear_degradation,
    register_degradation,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_degradation(GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED)
    yield
    clear_degradation(GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED)


def test_health_is_plain_ok_without_degradations() -> None:
    app_module = import_module("xagent.web.app")

    payload = asyncio.run(app_module.health_check())

    assert payload == {"status": "ok"}


def test_health_reports_active_degradations_but_stays_ok() -> None:
    """Degradations ride along for monitoring to alert on; the status stays
    healthy so container probes keep passing. Only signal names appear —
    /health is unauthenticated and the detail strings describe
    security-relevant misconfiguration."""
    app_module = import_module("xagent.web.app")
    register_degradation(GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED, "service account unset")

    payload = asyncio.run(app_module.health_check())

    assert payload["status"] == "ok"
    assert payload["degradations"] == [GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED]
    assert "service account unset" not in str(payload)
