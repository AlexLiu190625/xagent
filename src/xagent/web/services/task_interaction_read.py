"""The read surface's tuple adapter: what one waiting task is asking,
projected down to the ``(question, interactions)`` pair its five
consumers already expect.

Two steps, in this order, and nothing else:

1. The protocol marker fast path. ``tasks.interaction_protocol_version``
   decides whether this task can have a native interaction row at all.
   When it is not the version this reader speaks, the interaction table
   is not queried at all -- the legacy transcript reader answers
   directly. Every consumer already holds a loaded ``Task`` row when it
   needs the answer, so this check costs one attribute access on a row
   that is already in memory, not a query.
2. Otherwise ``materialize_compatibility_view`` -- the single rich
   implementation of "what is this waiting task's question" -- decides
   the tier, and this function projects its result down to the tuple.

This function performs **no authorization**. It takes a ``Task`` object,
not an id, precisely so that a caller has to have resolved and
authorized that row through its own layer first. All five callers do
(two request-scoped handlers, one connection-scoped snapshot builder
that calls it twice, and one worker-owned short session that resolves
through ``_resolve_task_or_404``). A new caller that has not is a bug in that
caller, not something this function can detect.

What the tuple cannot carry, and why that is accepted:

* ``reason``. The rich result names *why* a question cannot currently be
  answered; a two-slot tuple has no room for it, so this projection
  drops it. #1079's endpoint consumes the rich result directly and keeps
  it. Lossy by design, not an oversight.
* A uniform ``interactions`` element shape. On the legacy tiers the
  elements are whatever the JSON column holds -- sparse dicts written by
  the transcript producer. On the native tier they are fixed-shape dumps
  of the parsed payload model. Both are ``list[dict]`` and every consumer
  treats them as opaque, but the key sets differ. This adapter does not
  converge them; a typed element model belongs with the endpoint that
  first needs to type them.

Both empty slots is a real outcome, not an error path. It means: an
active native row holds this task's answer slot, and this reader could
not read it. The consumer renders nothing to answer -- an empty field on
three call sites, the generic waiting message on the websocket replay
path, and a null ``pending_interaction`` on the v1 snapshot. That is a
dead end for the user, and it is only acceptable because the tier that
produces it raises an operational signal at the same moment, so someone
can look and fix it. If that alarm is ever removed or downgraded to
silence, this projection has to be decided again -- the two halves are
one decision.

Three facts about the compatibility seam that refuses a legacy-shaped
answer, written here because this adapter's projection depends on them:

* The seam does not read the anchor. It refuses on exactly two things:
  whether the task has an active native row, and whether the
  continuation command carries a well-formed receipt. So a question this
  adapter projects *with* its interaction controls -- the native tier,
  whose anchor did resolve -- is refused by the seam just the same. The
  window where the interface offers an answer the backend will not take
  is therefore wider than the unresolved-anchor tier this projection was
  originally reasoned about.
* That window is **currently unreachable, and what makes it unreachable
  is that no writer has been wired yet**, not this design. The seam only
  triggers when an active native row exists, and nothing in production
  writes one today, so every waiting read takes the marker fast path.
  The change that wires the first finalizer makes it reachable, on the
  day it merges.
* The seam is installed on the websocket resume path only. The a2a and
  v1 reply resume paths do not have it. So "no controls shown" and "a
  free-text answer would be refused anyway" agree on the websocket path
  and only there; answering the same question through v1 or a2a is not
  refused by anything.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..models.task_interaction import INTERACTION_PROTOCOL_VERSION
from .chat_history_service import get_latest_waiting_question
from .task_interaction_service import materialize_compatibility_view

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from ..models.task import Task

__all__ = ["get_pending_interaction_question"]


def get_pending_interaction_question(
    db: "Session", task: "Task"
) -> tuple[str | None, list[dict[str, Any]] | None]:
    """This waiting task's question and its interaction descriptors.

    ``task`` is a loaded ``Task`` row, not an id: step 0 reads
    ``interaction_protocol_version`` off it, and the caller is the layer
    that authorized it (see the module docstring).
    """

    marker = task.interaction_protocol_version
    if marker != INTERACTION_PROTOCOL_VERSION:
        # No native row can belong to this task under a marker this
        # reader does not speak, so the interaction table is not queried.
        # A NULL marker means no structured row was ever published for
        # this task's current wait, which is the one case where a
        # transcript row that a later structured publication superseded
        # is still the honest answer -- hence the second pass. Any other
        # unrecognized value leaves the structured side's state unknown,
        # so the second pass stays shut.
        return get_latest_waiting_question(
            db, int(task.id), allow_superseded=marker is None
        )

    view = materialize_compatibility_view(db, int(task.id), allow_superseded=True)
    if view.tier == "unanswerable":
        # The question text, when the tier could still read one, and no
        # controls: this question cannot be answered right now, so
        # offering controls for it would be a lie. The two tiers that
        # could not read the text at all carry ``question=None`` from the
        # view itself, which is where both slots come out empty.
        return view.question, None
    return view.question, view.interactions
