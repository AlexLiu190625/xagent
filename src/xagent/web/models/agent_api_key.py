"""API key for SDK-side authentication, bound 1:1 to a published agent."""

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func, text

from .database import Base


class AgentApiKey(Base):  # type: ignore
    """SDK API key. One active key per agent, enforced by a partial unique index.

    Rotating a key inserts a new active row and marks the old one revoked;
    revoked rows stay in the table as an audit trail.
    """

    __tablename__ = "agent_api_keys"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(
        Integer,
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Public-safe lookup handle (6-char in practice; column sized to 12 for headroom).
    key_prefix = Column(String(12), nullable=False, unique=True, index=True)
    # bcrypt(full_key, cost=12). Never store the plaintext secret.
    key_hash = Column(String(128), nullable=False)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    agent = relationship("Agent")

    __table_args__ = (
        # At most one active (non-revoked) key per agent. Partial unique index
        # backed by both PostgreSQL (postgresql_where) and SQLite (sqlite_where)
        # dialect kwargs -- both engines support partial indexes natively and
        # we need the constraint identically on both so the SQLite-backed test
        # suite exercises the same rotation semantics as production. Without
        # ``sqlite_where`` SQLite would degrade to a plain unique index and
        # silently reject every legitimate key rotation.
        Index(
            "uq_agent_api_keys_agent_active",
            "agent_id",
            unique=True,
            postgresql_where=text("revoked_at IS NULL"),
            sqlite_where=text("revoked_at IS NULL"),
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<AgentApiKey(id={self.id}, agent_id={self.agent_id}, "
            f"key_prefix='{self.key_prefix}', revoked={self.revoked_at is not None})>"
        )
