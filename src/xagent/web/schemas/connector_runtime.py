"""Connector runtime requirements: shared response shape and the values
request body.

Three producers share the response shape: the agent-keyed and task-keyed
read endpoints, and the values endpoint's 200 response. All three describe
a requirements report -- which runtime inputs a task's (or a prospective
task's) connectors declare, and whether each one already has a value --
never a stored value itself, and never a connector's transport or
authentication configuration.

Placed in its own module rather than ``schemas/chat.py`` or ``schemas/v1.py``
because it has three audiences, not one: folding it into either of those
modules would couple that module's own audience to the other two.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ConnectorRuntimeRefModel(BaseModel):
    """Wire identity of a connector, as returned in a requirements report."""

    connector_type: str
    connector_id: int


class ConnectorRuntimeInputModel(BaseModel):
    """One declared runtime input and whether it is currently satisfied.

    ``key`` is the raw key name a connector owner wrote when declaring the
    input -- there is no human-readable label anywhere in the declaration.
    ``type`` is already normalized server-side to ``"string"`` or
    ``"object"``; a client must not normalize it again or expect any other
    value. ``satisfied`` and ``expired`` are constants in this phase for the
    ``secrets``/``auth_selector`` sections: no secret store exists yet, so
    both are always ``False``.
    """

    section: str
    key: str
    type: str
    required: bool
    satisfied: bool
    expired: bool = False


class ConnectorRuntimeConnectorModel(BaseModel):
    """One connector's declared runtime inputs.

    ``name`` is the only piece of connector identity beyond the ref that is
    ever included -- never the connector's URL, headers, environment, or
    authentication configuration.
    """

    connector_ref: ConnectorRuntimeRefModel
    name: str
    inputs: list[ConnectorRuntimeInputModel]


class ConnectorRuntimeRequirementsModel(BaseModel):
    """A requirements report. Every field always appears.

    ``connectors`` is empty, never omitted, when nothing is selected or
    declares a runtime input. ``secrets_expires_at`` is a constant ``null``
    in this phase; a later phase gives it a real value without changing
    its meaning or making it optional.
    """

    satisfied: bool
    secrets_expires_at: str | None
    connectors: list[ConnectorRuntimeConnectorModel]


class ConnectorRuntimeValueItem(BaseModel):
    """One connector's caller-supplied context values.

    ``secrets`` and ``auth_selector`` are deliberately absent from this
    phase's request shape: ``extra="forbid"`` turns either one into a 422
    rather than silently accepting and discarding it.
    """

    model_config = ConfigDict(extra="forbid")

    connector_ref: ConnectorRuntimeRefModel
    context: dict[str, object] | None = None


class ConnectorRuntimeValuesRequest(BaseModel):
    """Body of ``POST /api/chat/task/{task_id}/connector-runtime-values``.

    No override switch of any kind belongs here: a stored value is never
    replaced, and ``extra="forbid"`` turns an attempt to add one (for
    example ``if_absent`` or ``force``) into a 422.
    """

    model_config = ConfigDict(extra="forbid")

    items: list[ConnectorRuntimeValueItem]
