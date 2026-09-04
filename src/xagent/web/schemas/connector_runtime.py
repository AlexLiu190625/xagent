"""Connector runtime requirements: the shared response shape both read
endpoints return.

The agent-keyed and task-keyed read endpoints both return this same
requirements report -- which runtime inputs a task's (or a prospective
task's) connectors declare, and whether each one already has a value --
never a stored value itself, and never a connector's transport or
authentication configuration.

Placed in its own module rather than ``schemas/chat.py`` because a values-
submission endpoint lands on top of it shortly and will share this same
response shape as its own 200 body; putting it here now avoids a later
move that would touch every existing importer.
"""

from __future__ import annotations

from pydantic import BaseModel


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
