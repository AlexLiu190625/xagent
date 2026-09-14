from __future__ import annotations

import pytest

from xagent.core.tools.adapters.vibe.current_time_tool import validate_local_time


def test_validate_local_time_unchanged_after_helper_extraction() -> None:
    with pytest.raises(ValueError) as exc:
        validate_local_time("2026-10-04T02:30", "EST")

    assert str(exc.value) == (
        "timezone must be a Region/City IANA name such as "
        "'Australia/Sydney' or 'UTC', not 'EST'"
    )
