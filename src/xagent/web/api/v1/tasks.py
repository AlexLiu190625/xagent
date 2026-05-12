"""SDK task endpoints -- /v1/chat/tasks/* family.

Stub router; the actual endpoints land in upcoming commits:

  - D: POST /v1/chat/tasks
  - E: POST /v1/chat/tasks/{id}/messages, GET /v1/chat/tasks/{id}
  - F: GET /v1/chat/tasks/{id}/steps

This module exists in C only so ``v1_router.include_router(tasks.router)``
in __init__.py succeeds without referencing a missing import. The router
has no routes registered yet.
"""

from fastapi import APIRouter

router = APIRouter()
