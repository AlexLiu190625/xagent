from __future__ import annotations

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.sql import func

from .database import Base


class MCPOAuthClient(Base):  # type: ignore
    """OAuth client metadata for an existing HTTP MCP server."""

    __tablename__ = "mcp_oauth_clients"
    __table_args__ = (
        UniqueConstraint(
            "mcp_server_id",
            "issuer",
            "client_id",
            name="uq_mcp_oauth_clients_server_issuer_client",
        ),
    )

    id = Column(Integer, primary_key=True)
    mcp_server_id = Column(
        Integer,
        ForeignKey("mcp_servers.id", ondelete="CASCADE"),
        nullable=False,
    )
    issuer = Column(String(1000), nullable=False, index=True)
    authorization_endpoint = Column(String(1000), nullable=False)
    token_endpoint = Column(String(1000), nullable=False)
    client_id = Column(String(1000), nullable=False)
    client_secret = Column(Text, nullable=True)
    token_endpoint_auth_method = Column(String(100), nullable=False, default="none")
    redirect_uri = Column(String(1000), nullable=False)
    metadata_json = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    mcp_server = relationship("MCPServer")

    def __repr__(self) -> str:
        return (
            f"<MCPOAuthClient(id={self.id}, mcp_server_id={self.mcp_server_id}, "
            f"issuer='{self.issuer}')>"
        )


class MCPOAuthGrant(Base):  # type: ignore
    """Encrypted OAuth grant for an MCP resource owner."""

    __tablename__ = "mcp_oauth_grants"
    __table_args__ = (
        UniqueConstraint(
            "mcp_server_id",
            "user_id",
            "resource_owner_key",
            "mcp_oauth_client_id",
            "issuer",
            "resource",
            "scope",
            name="uq_mcp_oauth_grants_lookup",
        ),
    )

    id = Column(Integer, primary_key=True)
    mcp_server_id = Column(
        Integer,
        ForeignKey("mcp_servers.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    mcp_oauth_client_id = Column(
        Integer,
        ForeignKey("mcp_oauth_clients.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    resource_owner_key = Column(String(512), nullable=False)
    issuer = Column(String(1000), nullable=False)
    resource = Column(String(1000), nullable=False)
    scope = Column(Text, nullable=False, default="")
    access_token = Column(Text, nullable=False)
    refresh_token = Column(Text, nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
    token_type = Column(String(50), nullable=False, default="Bearer")
    status = Column(String(50), nullable=False, default="active")
    metadata_json = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    revoked_at = Column(DateTime(timezone=True), nullable=True)

    mcp_server = relationship("MCPServer")
    oauth_client = relationship("MCPOAuthClient")
    user = relationship("User")

    def __repr__(self) -> str:
        return (
            f"<MCPOAuthGrant(id={self.id}, mcp_server_id={self.mcp_server_id}, "
            f"user_id={self.user_id}, resource_owner_key='{self.resource_owner_key}')>"
        )


class MCPOAuthFlowState(Base):  # type: ignore
    """Short-lived OAuth state for MCP Authorization Code + PKCE."""

    __tablename__ = "mcp_oauth_flow_states"

    id = Column(Integer, primary_key=True)
    state = Column(String(255), nullable=False, unique=True)
    mcp_server_id = Column(
        Integer,
        ForeignKey("mcp_servers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    mcp_oauth_client_id = Column(
        Integer,
        ForeignKey("mcp_oauth_clients.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    resource_owner_key = Column(String(512), nullable=False)
    issuer = Column(String(1000), nullable=False)
    resource = Column(String(1000), nullable=False)
    scope = Column(Text, nullable=False, default="")
    code_verifier = Column(Text, nullable=False)
    redirect_after = Column(String(1000), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    consumed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    mcp_server = relationship("MCPServer")
    oauth_client = relationship("MCPOAuthClient")
    user = relationship("User")

    def __repr__(self) -> str:
        return (
            f"<MCPOAuthFlowState(id={self.id}, mcp_server_id={self.mcp_server_id}, "
            f"user_id={self.user_id})>"
        )
