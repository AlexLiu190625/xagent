"""Client-visible projections of server-side failures.

Holds the fixed fallback strings used when a failure has nothing safe to
say, the per-exception adapters that pass a curated message through, and
``PublicErrorDetails`` -- the only structured payload allowed onto a
task_error frame, together with the reason allowlist that governs it.
"""

from dataclasses import dataclass
from enum import StrEnum

from ...core.tools.adapters.vibe.config import RequiredMCPUnavailableError
from ...core.tools.adapters.vibe.connector_runtime import (
    RUNTIME_SOURCE_KEY_RE,
    ConnectorRuntimeError,
)

CLIENT_SAFE_VALIDATION_ERROR = "The message could not be processed. Please try again."

# Task audiences did not necessarily initiate the failing operation, so a
# task-level failure uses neutral wording instead of the validation fallback.
CLIENT_SAFE_TASK_FAILURE = "Task execution failed."
CLIENT_SAFE_GUIDANCE_IN_PROGRESS = (
    "A previous guidance message is still being applied. Please wait for it to finish."
)


class ClientErrorCode(StrEnum):
    """Stable identifiers clients may localize without trusting server prose."""

    MESSAGE_PROCESSING_FAILED = "message_processing_failed"
    TASK_EXECUTION_FAILED = "task_execution_failed"
    GUIDANCE_IN_PROGRESS = "guidance_in_progress"
    MESSAGE_RATE_LIMITED = "message_rate_limited"
    MESSAGE_ID_CONFLICT = "message_id_conflict"
    MESSAGE_DELIVERY_FAILED = "message_delivery_failed"
    MESSAGE_CONTINUATION_UNSUPPORTED = "message_continuation_unsupported"
    TASK_PAUSE_IN_PROGRESS = "task_pause_in_progress"
    MESSAGE_ACCEPTANCE_PENDING = "message_acceptance_pending"
    TASK_UNAVAILABLE = "task_unavailable"
    TASK_BUSY = "task_busy"
    WORKFORCE_UNAVAILABLE = "workforce_unavailable"
    WORKFORCE_ARCHIVED = "workforce_archived"
    MESSAGE_ATTACHMENT_CORRUPT = "message_attachment_corrupt"
    MESSAGE_ATTACHMENT_UNAVAILABLE = "message_attachment_unavailable"
    TASK_CHECKPOINT_UNREADABLE = "task_checkpoint_unreadable"
    AUTHENTICATION_REQUIRED = "authentication_required"
    TASK_ACCESS_DENIED = "task_access_denied"
    INVALID_MESSAGE = "invalid_message"


def client_error_message(code: ClientErrorCode) -> str:
    """Return the fixed safe fallback for a stable client error code."""

    return {
        ClientErrorCode.MESSAGE_PROCESSING_FAILED: CLIENT_SAFE_VALIDATION_ERROR,
        ClientErrorCode.TASK_EXECUTION_FAILED: CLIENT_SAFE_TASK_FAILURE,
        ClientErrorCode.GUIDANCE_IN_PROGRESS: CLIENT_SAFE_GUIDANCE_IN_PROGRESS,
        ClientErrorCode.MESSAGE_RATE_LIMITED: (
            "You're sending messages too quickly. Please wait a moment and try again."
        ),
        ClientErrorCode.MESSAGE_ID_CONFLICT: (
            "Message id was already used for different content or files."
        ),
        ClientErrorCode.MESSAGE_DELIVERY_FAILED: (
            "The message could not be delivered. Please retry the draft."
        ),
        ClientErrorCode.MESSAGE_CONTINUATION_UNSUPPORTED: (
            "Task does not support message continuation."
        ),
        ClientErrorCode.TASK_PAUSE_IN_PROGRESS: (
            "Task pause is still being applied; please retry shortly."
        ),
        ClientErrorCode.MESSAGE_ACCEPTANCE_PENDING: (
            "Message acceptance is still being reconciled. Please retry shortly."
        ),
        ClientErrorCode.TASK_UNAVAILABLE: "Task is no longer available.",
        ClientErrorCode.TASK_BUSY: (
            "Task is currently busy; please wait for the previous turn to finish "
            "before sending another message."
        ),
        ClientErrorCode.WORKFORCE_UNAVAILABLE: (
            "This workforce conversation can no longer accept messages; "
            "please start a new conversation."
        ),
        ClientErrorCode.WORKFORCE_ARCHIVED: (
            "This workforce has been archived. Unarchive and publish it before "
            "starting a new conversation, or select an active workforce."
        ),
        ClientErrorCode.MESSAGE_ATTACHMENT_CORRUPT: (
            "A stored file for this message failed its integrity check "
            "and must be re-uploaded."
        ),
        ClientErrorCode.MESSAGE_ATTACHMENT_UNAVAILABLE: (
            "A stored file for this message could not be read. Please try again."
        ),
        ClientErrorCode.TASK_CHECKPOINT_UNREADABLE: (
            "The task's saved progress could not be read."
        ),
        ClientErrorCode.AUTHENTICATION_REQUIRED: (
            "Authentication is required to send this message."
        ),
        ClientErrorCode.TASK_ACCESS_DENIED: "You do not have access to this task.",
        ClientErrorCode.INVALID_MESSAGE: "The message format is invalid.",
    }[code]


def required_mcp_unavailable_client_message(
    error: BaseException,
    *,
    fallback: str = CLIENT_SAFE_VALIDATION_ERROR,
) -> str:
    """Adapt the curated required-MCP failure without opening a generic escape.

    The runtime check keeps this boundary fail-closed even if a future caller
    passes an incidental exception despite the function's specific name.
    """

    if not isinstance(error, RequiredMCPUnavailableError):
        return fallback
    message = str(error)
    if message.strip():
        return message
    return fallback


def connector_runtime_client_message(
    error: BaseException,
    *,
    fallback: str = CLIENT_SAFE_TASK_FAILURE,
) -> str:
    """Adapt the curated connector-runtime failure without a generic escape.

    The runtime check keeps this boundary fail-closed even if a future caller
    passes an incidental exception despite the function's specific name.
    """

    if not isinstance(error, ConnectorRuntimeError):
        return fallback
    message = error.safe_message
    if isinstance(message, str) and message.strip():
        return message
    return fallback


CONNECTOR_RUNTIME_PUBLIC_REASONS = frozenset(
    {
        # Missing values and binding.
        "not_provided",
        "store_lost",
        "connector_not_selected",
        "auth_selector_not_supported",
        "duplicate_ref",
        "undeclared_context_key",
        "undeclared_secrets_key",
        "undeclared_auth_selector_key",
        # Fixed 503 strings built by direct ConnectorRuntimeError construction
        # in three other modules. Each one states that a server-side component
        # is unavailable; none of them states who owns the task, or how an
        # authorization check resolved. Two further strings of exactly this
        # shape (runtime_task_identity_mismatch, runtime_owner_mismatch) are
        # deliberately absent for that reason -- see the class docstring below.
        "team_scope_resolution_failed",
        "team_env_resolution_failed",
        "runtime_view_resolution_failed",
        "custom_api_config_load_failed",
    }
)
# Every member above is raised somewhere in this repository today, and a test
# asserts that in both directions. Add a reason here in the same change that
# adds the site raising it, never ahead of it: a listed reason nothing produces
# is an allowance with no expiry date, and by the time the raising code arrives
# nobody remembers which audience the reason was judged against.
CONNECTOR_RUNTIME_PUBLIC_REASON_PREFIXES = frozenset(
    {
        "missing_context",
        "type_mismatch.context",
        "type_mismatch.secrets",
        "type_mismatch.auth_selector",
        "conflict.context",
        "conflict.secrets",
        "conflict.auth_selector",
    }
)


def _is_public_reason(reason: object) -> bool:
    """True when this reason may reach a client. Used by PublicErrorDetails.

    The key half of a prefixed reason is matched against the declared runtime
    key grammar itself, not a copy of it, so the two cannot drift apart.
    """

    if not isinstance(reason, str):
        return False
    if reason in CONNECTOR_RUNTIME_PUBLIC_REASONS:
        return True
    prefix, separator, key = reason.rpartition(".")
    if not separator:
        return False
    if prefix not in CONNECTOR_RUNTIME_PUBLIC_REASON_PREFIXES:
        return False
    return RUNTIME_SOURCE_KEY_RE.fullmatch(key) is not None


@dataclass(frozen=True)
class PublicErrorDetails:
    """The only shape allowed into a task_error frame's ``details``.

    ``reason`` is normalized on construction: a value that is not a listed
    enum member, and not ``<listed prefix>.<declared key name>``, becomes
    ``None``. Constructing this type and passing the reason whitelist are
    therefore the same act -- there is no path that produces an instance
    carrying free text, including a direct call from another module.

    Nulling rather than raising is deliberate: every construction site is on
    the reporting path of an already-failed task, and raising there would
    turn a diagnosable failure into an undiagnosable crash.

    The sink is ``broadcast_to_task``, whose audience includes anonymous
    widget and share-link visitors, so every listed reason and every new
    field must answer one question first: can a visitor who is not the task
    owner read the task's ownership, or the outcome of an authorization
    check, out of it? There is no ``connector_ref`` field because the answer
    for it is yes; two runtime reasons are omitted for the same answer.
    """

    reason: str | None

    def __post_init__(self) -> None:
        if self.reason is not None and not _is_public_reason(self.reason):
            object.__setattr__(self, "reason", None)

    def to_wire(self) -> dict[str, str]:
        return {"reason": self.reason} if self.reason is not None else {}


def connector_runtime_public_error(
    error: BaseException,
) -> tuple[str, PublicErrorDetails] | None:
    """Project a connector-runtime failure onto the wire-safe (code, details).

    Returns ``None`` for anything else, so a caller cannot widen the surface
    by passing an incidental exception. The reason filter itself lives in
    ``PublicErrorDetails``; this function only decides whether the exception
    is one we project at all.

    This is not the only client-visible projection of this exception.
    ``_raise_v1_connector_runtime_error`` (``web/api/v1/tasks.py``) projects it
    for the SDK surface and ships ``to_public_error()["details"]`` whole,
    ``connector_ref`` included. The two differ because their audiences do: that
    one answers an API key held by a caller already authorized for the task,
    while this one feeds ``broadcast_to_task``, which reaches every connection
    under the task id including anonymous widget and share-link visitors.
    Keep them as two projectors with one audience each; folding them into one
    that takes the audience as an argument puts the width of the output behind
    a caller-supplied flag, which fails open the first time it is passed wrong.
    """

    if not isinstance(error, ConnectorRuntimeError):
        return None
    details = error.details
    if not isinstance(details, dict):
        # ``__init__`` normalizes details to a dict, but it is a plain public
        # attribute anything can reassign afterwards. This is the last step
        # before the wire, so verify rather than assume: a payload of the wrong
        # shape means the instance is not trustworthy, and the safe answer is
        # to fall all the way back to the opaque failure rather than guess
        # which half of it is still readable.
        return None
    return error.code, PublicErrorDetails(reason=details.get("reason"))
