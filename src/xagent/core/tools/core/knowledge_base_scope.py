"""Typed scope error for team-resolved knowledge bases.

MODULE CONSTRAINT: this module must not import any xagent module.
Zero dependencies outside the standard library.

That is what makes it importable at module level from BOTH sides of the
core/web boundary -- core/tools/core/document_search.py catches it, and
web/services/knowledge_base_team_scope.py raises it -- with no possible
import cycle. A single xagent import here reintroduces the ordering
problem this module exists to remove.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional


class KnowledgeBaseScopeError(RuntimeError):
    """A team-scoped knowledge-base resolution could not be completed.

    Raised when the application-owned team visibility hook is missing,
    malformed, or fails, and there is no safe way to fall back to a
    partial or empty answer -- the caller must be told resolution
    failed rather than receiving a result that silently omits the
    team layer. The two run-path handlers that would otherwise
    re-wrap any exception into a bare ``RuntimeError`` catch this type
    first and re-raise it unchanged, so its ``status_code`` and
    ``code`` survive to the caller.

    ``safe_message`` (not ``message``) on purpose: it mirrors
    ``ConnectorRuntimeError`` (``core/tools/adapters/vibe/connector_runtime.py``),
    the equivalent typed error on the connector team-scope seam, whose
    ``.safe_message`` is what its own catch site (``chat.py``) reads to
    build the HTTP response. Nothing in this codebase currently reads
    ``code``, ``status_code``, ``details``, or ``safe_message`` off this
    class -- both run-path handlers re-raise it unchanged rather than
    unpacking it -- but a future HTTP-facing catch site should find the
    same attribute name on either error, not two conventions for the same
    shape.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Optional[Mapping[str, Any]] = None,
        status_code: int = 503,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message
        self.details = dict(details) if details is not None else {}
        self.status_code = status_code

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"{self.__class__.__name__}(code={self.code!r}, "
            f"message={self.safe_message!r}, status_code={self.status_code!r})"
        )
