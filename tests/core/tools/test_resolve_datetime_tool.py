from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from xagent.core.tools.adapters.vibe import current_time_tool as module
from xagent.core.tools.adapters.vibe.current_time_tool import (
    GRAMMAR_FORMS,
    RESOLUTION_REASONS,
    ResolveDatetimeResult,
    resolve_datetime,
    validate_local_time,
)

# Tuesday 2026-09-15 12:30 in Sydney (AEST), 10:30 in Shanghai.
FROZEN = datetime(2026, 9, 15, 2, 30, 0, tzinfo=timezone.utc)

SYDNEY = "Australia/Sydney"
SHANGHAI = "Asia/Shanghai"

# (phrase, zone, expected) where expected is (resolved, has_time) on success
# or ("REFUSED", resolution) on refusal.
CASES: list[tuple[str, str, tuple[str, object]]] = [
    ("today", SYDNEY, ("2026-09-15T00:00:00+10:00", False)),
    ("tomorrow", SYDNEY, ("2026-09-16T00:00:00+10:00", False)),
    ("tomorrow at 3pm", SYDNEY, ("2026-09-16T15:00:00+10:00", True)),
    ("yesterday", SYDNEY, ("2026-09-14T00:00:00+10:00", False)),
    ("day after tomorrow", SYDNEY, ("2026-09-17T00:00:00+10:00", False)),
    ("next Monday", SYDNEY, ("2026-09-21T00:00:00+10:00", False)),
    ("next tuesday", SYDNEY, ("2026-09-22T00:00:00+10:00", False)),
    ("last Friday", SYDNEY, ("2026-09-11T00:00:00+10:00", False)),
    ("Friday at 10:30", SYDNEY, ("2026-09-18T10:30:00+10:00", True)),
    ("Friday 10am", SYDNEY, ("2026-09-18T10:00:00+10:00", True)),
    ("in 3 days", SYDNEY, ("2026-09-18T00:00:00+10:00", False)),
    ("in 2 hours", SYDNEY, ("2026-09-15T14:30:00+10:00", True)),
    ("in 1 week", SYDNEY, ("2026-09-22T00:00:00+10:00", False)),
    ("15 Sep 2026 10:00", SYDNEY, ("2026-09-15T10:00:00+10:00", True)),
    ("15 Sep 2026", SYDNEY, ("2026-09-15T00:00:00+10:00", False)),
    ("1 Jan 1990", SYDNEY, ("1990-01-01T00:00:00+11:00", False)),
    ("1990-01-01", SYDNEY, ("1990-01-01T00:00:00+11:00", False)),
    ("2026-10-04", SYDNEY, ("2026-10-04T00:00:00+10:00", False)),
    ("2026-09-15T10:00:00+10:00", SHANGHAI, ("2026-09-15T08:00:00+08:00", True)),
    ("01/02/1990", SYDNEY, ("REFUSED", "ambiguous_date")),
    ("今天", SHANGHAI, ("2026-09-15T00:00:00+08:00", False)),
    ("明天下午三点", SHANGHAI, ("2026-09-16T15:00:00+08:00", True)),
    ("后天", SHANGHAI, ("2026-09-17T00:00:00+08:00", False)),
    ("昨天", SHANGHAI, ("2026-09-14T00:00:00+08:00", False)),
    ("下周三", SHANGHAI, ("2026-09-23T00:00:00+08:00", False)),
    ("上周五", SHANGHAI, ("2026-09-11T00:00:00+08:00", False)),
    ("周五上午十点半", SHANGHAI, ("2026-09-18T10:30:00+08:00", True)),
    ("3天后", SHANGHAI, ("2026-09-18T00:00:00+08:00", False)),
    ("两小时后", SHANGHAI, ("2026-09-15T12:30:00+08:00", True)),
    ("1990年1月1日", SHANGHAI, ("1990-01-01T00:00:00+08:00", False)),
    ("2026年9月20日 下午两点", SHANGHAI, ("2026-09-20T14:00:00+08:00", True)),
    ("soon", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("next month", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("the first Monday of October", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("Bibek khadka", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("2026-10-04 02:30", SYDNEY, ("REFUSED", "nonexistent_local_time")),
    ("2026-04-05 02:30", SYDNEY, ("REFUSED", "ambiguous_local_time")),
    ("tomorrow", "EST", ("REFUSED", "invalid_timezone")),
    ("3pm", SYDNEY, ("2026-09-15T15:00:00+10:00", True)),
    ("10:30", SYDNEY, ("2026-09-15T10:30:00+10:00", True)),
    ("3", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("下午三点", SHANGHAI, ("2026-09-15T15:00:00+08:00", True)),
    ("十点半", SHANGHAI, ("2026-09-15T10:30:00+08:00", True)),
    ("noon", SYDNEY, ("REFUSED", "unsupported_expression")),
]


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_now", lambda: FROZEN)


def test_validate_local_time_unchanged_after_helper_extraction() -> None:
    with pytest.raises(ValueError) as exc:
        validate_local_time("2026-10-04T02:30", "EST")

    assert str(exc.value) == (
        "timezone must be a Region/City IANA name such as "
        "'Australia/Sydney' or 'UTC', not 'EST'"
    )


def test_grammar_table_has_every_case() -> None:
    assert len(CASES) == 44


@pytest.mark.parametrize(
    ("phrase", "zone", "expected"), CASES, ids=[c[0] for c in CASES]
)
def test_resolve_datetime_grammar(
    phrase: str, zone: str, expected: tuple[str, object]
) -> None:
    result = resolve_datetime(phrase, zone)

    if expected[0] == "REFUSED":
        reason = expected[1]
        assert result["success"] is False
        assert result["tool_name"] == "resolve_datetime"
        assert result["resolution"] == reason
        assert reason in RESOLUTION_REASONS
        assert "status" not in result
        assert ("supported" in result) == (reason == "unsupported_expression")
    else:
        resolved, has_time = expected
        assert result == {"resolved": resolved, "has_time": has_time, "timezone": zone}


def test_resolve_result_has_no_quotable_clock() -> None:
    assert set(ResolveDatetimeResult.model_fields) == {
        "resolved",
        "has_time",
        "timezone",
    }

    supported = resolve_datetime("soon", "UTC")["supported"]
    for line in supported:
        assert not re.search(r"\d{4}-\d{2}-\d{2}", line)
        assert not re.search(r"\d{4}", line)
    assert len(supported) == len(GRAMMAR_FORMS)

    assert set(RESOLUTION_REASONS) == {
        "unsupported_expression",
        "ambiguous_date",
        "nonexistent_local_time",
        "ambiguous_local_time",
        "invalid_timezone",
    }


def test_resolve_datetime_dst_and_zone() -> None:
    skipped = resolve_datetime("2026-10-04 02:30", SYDNEY)
    assert skipped["success"] is False
    assert skipped["resolution"] == "nonexistent_local_time"

    repeated = resolve_datetime("2026-04-05 02:30", SYDNEY)
    assert repeated["success"] is False
    assert repeated["resolution"] == "ambiguous_local_time"

    utc = resolve_datetime("tomorrow", "UTC")
    assert utc["resolved"] == "2026-09-16T00:00:00+00:00"
    assert utc["timezone"] == "UTC"


@pytest.mark.parametrize("zone", ["EST", "Sydney", "Etc/GMT+10"])
def test_resolve_datetime_rejects_non_region_city_zone(zone: str) -> None:
    result = resolve_datetime("tomorrow", zone)

    assert result["success"] is False
    assert result["resolution"] == "invalid_timezone"
    assert "supported" not in result


def test_phrase_zone_name_is_refused() -> None:
    result = resolve_datetime("15 Sep 2026 10:00 EST", SYDNEY)

    assert result["success"] is False
    assert result["resolution"] == "unsupported_expression"
