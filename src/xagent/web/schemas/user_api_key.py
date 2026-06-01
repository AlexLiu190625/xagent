"""Schemas for user personal SDK management keys."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class PersonalAPIKeyCreateResponse(BaseModel):
    id: int
    full_key: str = Field(
        ...,
        description=(
            "Plaintext personal key in the format "
            "xag_personal_<6 chars>_<32 chars>. Returned exactly once."
        ),
    )
    key_prefix: str
    created_at: datetime
    expires_at: Optional[datetime] = None


class PersonalAPIKeyMetadata(BaseModel):
    id: int
    key_prefix: str
    masked_key: str
    revoked_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    created_at: datetime


class PersonalAPIKeyRevokeResponse(BaseModel):
    revoked: bool
    revoked_at: Optional[datetime] = None
