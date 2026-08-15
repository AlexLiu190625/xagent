"""The ``create_task`` endpoint maps ``KnowledgeBaseScopeError`` to the
status and message the error itself carries, instead of letting its blanket
handler reclassify it as an internal error.

This is the one place anything reads ``status_code``/``safe_message`` off
the knowledge-base scope seam's typed error; every other handler (the two
on the run path, the two on the tool-build path) re-raises it unchanged so
those attributes survive to here. The arm sits directly beside the
``ConnectorRuntimeError`` arm that already does the same thing, and pins
the same three properties for it:

* the response status is the error's own ``status_code`` (503, "resolution
  failed, retryable"), not the funnel's generic 500;
* the response detail is ``safe_message``;
* ``code`` and ``details`` are diagnostic and stay out of the response.

The raise is injected at the endpoint's first statement rather than reached
organically: no step inside this endpoint resolves the team knowledge-base
layer today (resolution happens per search call, on the run path), so what
is under test is the funnel's classification, not any particular producer.
``validate_task_extension_requests`` is a convenient injection point
precisely because it runs before any database work -- and because its own
inner handler catches only ``TypeError``/``ValueError``, so a
``KnowledgeBaseScopeError`` (a ``RuntimeError``) reaches the outer funnel
unmodified.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from xagent.core.tools.core.knowledge_base_scope import KnowledgeBaseScopeError
from xagent.web.schemas.chat import TaskCreateRequest

_SAFE_MESSAGE = "Knowledge base team scope is unavailable."
_PRIVATE_DETAIL = "team_scope_resolution_failed"
_ERROR_CODE = "knowledge_base_scope_unavailable"


def _scope_error() -> KnowledgeBaseScopeError:
    return KnowledgeBaseScopeError(
        _ERROR_CODE,
        _SAFE_MESSAGE,
        details={"reason": _PRIVATE_DETAIL},
        status_code=503,
    )


@pytest.mark.asyncio
async def test_create_task_answers_knowledge_base_scope_error_with_its_own_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.api import chat as chat_module

    def _raise(*args: object, **kwargs: object) -> None:
        raise _scope_error()

    monkeypatch.setattr(chat_module, "validate_task_extension_requests", _raise)

    with pytest.raises(HTTPException) as excinfo:
        await chat_module.create_task(
            TaskCreateRequest(title="kb-scope-http-exit"),
            http_request=None,
            db=MagicMock(),
            user=MagicMock(id=1, is_admin=False),
        )

    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == _SAFE_MESSAGE


@pytest.mark.asyncio
async def test_create_task_keeps_knowledge_base_scope_details_out_of_the_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The diagnostic half of the error is logged, never returned."""
    from xagent.web.api import chat as chat_module

    def _raise(*args: object, **kwargs: object) -> None:
        raise _scope_error()

    monkeypatch.setattr(chat_module, "validate_task_extension_requests", _raise)

    with pytest.raises(HTTPException) as excinfo:
        await chat_module.create_task(
            TaskCreateRequest(title="kb-scope-http-exit"),
            http_request=None,
            db=MagicMock(),
            user=MagicMock(id=1, is_admin=False),
        )

    detail = str(excinfo.value.detail)
    assert _PRIVATE_DETAIL not in detail
    assert _ERROR_CODE not in detail
