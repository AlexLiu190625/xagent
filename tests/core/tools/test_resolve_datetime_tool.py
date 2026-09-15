from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from tests.core.tools.test_current_time_tool import _FakeConfig
from xagent.core.tools.adapters.vibe import current_time_tool as module
from xagent.core.tools.adapters.vibe.base import ToolCategory
from xagent.core.tools.adapters.vibe.current_time_tool import (
    GRAMMAR_FORMS,
    RESOLUTION_REASONS,
    CurrentTimeTool,
    ResolveDatetimeResult,
    ResolveDatetimeTool,
    ValidateLocalTimeTool,
    resolve_datetime,
    validate_local_time,
)
from xagent.core.tools.adapters.vibe.factory import ToolFactory, ToolRegistry
from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec

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
    # Hour twelve with a half-day word is refused rather than picked one of
    # two ways: it could be midnight (today ending or tomorrow beginning)
    # or, read as bare digits, noon.
    ("晚上12点", SHANGHAI, ("REFUSED", "unsupported_expression")),
    ("12 pm", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("上午12点", SHANGHAI, ("REFUSED", "unsupported_expression")),
    ("12 am", SYDNEY, ("REFUSED", "unsupported_expression")),
    # A supported form with extra words around it is the whole phrase
    # failing to match, not the supported word inside it being found.
    ("call me back tomorrow if you can", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("我昨天见过他", SHANGHAI, ("REFUSED", "unsupported_expression")),
    ("tomorrow at 3pm please", SYDNEY, ("REFUSED", "unsupported_expression")),
    # "this <weekday>" stays within the current calendar week, including
    # when today already is that weekday.
    ("this friday", SYDNEY, ("2026-09-18T00:00:00+10:00", False)),
    ("this tuesday", SYDNEY, ("2026-09-15T00:00:00+10:00", False)),
    # The minute half of "N hours/minutes later" (English and Chinese).
    ("in 30 minutes", SYDNEY, ("2026-09-15T13:00:00+10:00", True)),
    ("30分钟后", SHANGHAI, ("2026-09-15T11:00:00+08:00", True)),
    # The alternate character for Sunday folds to the same weekday
    # position as the other spelling.
    ("周天", SHANGHAI, ("2026-09-20T00:00:00+08:00", False)),
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
    assert len(CASES) == 56


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
        assert result["error"]
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


def test_resolve_datetime_rejects_sub_minute_offset() -> None:
    """Africa/Monrovia used a -00:44:30 offset before it standardised in
    1972, which RFC 3339 / JSON Schema 'date-time' cannot represent (only a
    ±HH:MM offset is legal). The value is refused rather than truncated or
    rounded into an approximation."""
    result = resolve_datetime("1 Jan 1970", "Africa/Monrovia")

    assert result["success"] is False
    assert result["resolution"] == "unsupported_expression"
    assert result["error"]


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


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (ToolSelectionSpec.from_raw(tool_categories=None), 1),  # ALL
        (ToolSelectionSpec.from_raw(tool_categories=["web_search"]), 1),  # non-basic
        (ToolSelectionSpec.from_raw(tool_categories=[]), 0),  # explicit NONE
    ],
)
async def test_resolve_datetime_is_intrinsic(
    spec: ToolSelectionSpec, expected: int
) -> None:
    """The full factory pipeline assembles exactly one usable
    resolve_datetime for any non-NONE selection -- including one that
    never picks the basic category -- and none for an explicit NONE."""
    tools = await ToolFactory.create_all_tools(
        _FakeConfig(spec), apply_user_override_filter=False
    )

    assert [t.name for t in tools].count("resolve_datetime") == expected


async def test_resolve_datetime_creator_is_skipped_for_explicit_none() -> None:
    """The registry gate must not even build the intrinsic tool for an
    explicit zero-tools agent: the NONE contract wins over always-on."""
    spec = ToolSelectionSpec.from_raw(tool_categories=[])

    tools = await ToolRegistry.create_registered_tools(_FakeConfig(spec))

    assert "resolve_datetime" not in [getattr(t, "name", None) for t in tools]


def test_tool_declares_read_only_other_identity() -> None:
    metadata = ResolveDatetimeTool().metadata

    assert metadata.name == "resolve_datetime"
    assert metadata.read_only is True
    assert metadata.concurrency_safe is True
    assert metadata.category is ToolCategory.OTHER


def test_time_tools_cross_reference() -> None:
    description = ResolveDatetimeTool().description

    assert "get_current_time" in description
    assert "validate_local_time" in description
    assert "resolve_datetime" in CurrentTimeTool().description
    assert "resolve_datetime" in ValidateLocalTimeTool().description
    # The description promises only what the engine does at this point: no
    # verbatim-span refusal and no instruction to quote the result as a source.
    assert "refused when that text" not in description
    assert "Quote the result" not in description


def test_tool_runs_through_the_json_surface() -> None:
    tool = ResolveDatetimeTool()

    assert tool.run_json_sync({"phrase": "tomorrow at 3pm", "timezone": SYDNEY}) == {
        "resolved": "2026-09-16T15:00:00+10:00",
        "has_time": True,
        "timezone": SYDNEY,
    }

    refused = asyncio.run(tool.run_json_async({"phrase": "soon", "timezone": "UTC"}))
    assert refused["resolution"] == "unsupported_expression"

    with pytest.raises(ValidationError):
        tool.run_json_sync({"phrase": "tomorrow"})
