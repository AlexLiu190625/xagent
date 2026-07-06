#!/usr/bin/env bash
# Check Alembic has exactly one head

set -e

if command -v uv >/dev/null 2>&1; then
    ALEMBIC_CMD=(uv run alembic)
else
    ALEMBIC_CMD=(python -m alembic)
fi

HEADS=$("${ALEMBIC_CMD[@]}" heads | grep -c " (head)" || true)

if [ "$HEADS" -eq 1 ]; then
    echo "Alembic: Single head confirmed"
    exit 0
else
    echo "Alembic: Expected 1 head, found $HEADS"
    echo "Run '${ALEMBIC_CMD[*]} heads' to see all heads"
    exit 1
fi
