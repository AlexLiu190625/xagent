"""The tenant cap of fair-share dispatch: which tenant is held back, and where."""

import pytest

from xagent.web.services.task_dispatch_cap import DispatchCapSnapshot, DispatchPolicy

BAND = DispatchPolicy(limit=4, interactive_reserve=2)


def band_cap(occupancy, *, active=2, held=False, policy=BAND):
    return DispatchCapSnapshot(
        policy=policy,
        active_ids=frozenset(range(900, 900 + active)),
        held=held,
        occupancy=occupancy,
    )


# --- which tenant is held back ---------------------------------------------------


@pytest.mark.parametrize(
    ("occupancy", "queued", "expected"),
    [
        ({"x": 2, "y": 2}, {}, None),
        ({"x": 2, "y": 2}, {"a": 1}, None),
        ({}, {"a": 1, "b": 2}, None),
        # The top holder has nothing queued, so nobody is held back.
        ({"x": 3, "y": 1}, {"y": 5, "z": 6}, None),
        # A tie: only the tied tenant with the oldest queued command is held back.
        ({"x": 2, "y": 2}, {"x": 7, "y": 3}, "y"),
        ({"x": 2, "y": 2}, {"x": 3, "y": 7}, "x"),
        ({"x": 2, "y": 2}, {"x": 9, "z": 1}, "x"),
        ({"a": 2, "b": 1}, {"a": 9, "b": 1}, "a"),
        # The empty key of a row from before tenant keys is an ordinary tenant.
        ({"": 3, "b": 1}, {"": 4, "b": 5}, ""),
        ({"a": 2, "b": 1}, None, None),
    ],
    ids=[
        "empty-listing",
        "one-tenant",
        "nothing-held",
        "top-holder-not-queued",
        "tie-held-back-is-the-earlier-queue",
        "tie-held-back-is-the-earlier-queue-other-side",
        "tie-only-one-queued",
        "unique-top",
        "empty-key",
        "no-listing",
    ],
)
def test_the_capped_tenant_rule(occupancy, queued, expected):
    assert band_cap(occupancy).capped_tenant(queued) == expected


@pytest.mark.parametrize(
    ("active", "held", "band", "expected"),
    [(2, False, True, "a"), (3, False, True, "a"), (2, True, False, None)],
    ids=["lower-edge", "upper-edge", "held"],
)
def test_the_cap_applies_only_between_the_batch_line_and_the_limit(
    active, held, band, expected
):
    cap = band_cap({"a": 2, "b": 1}, active=active, held=held)

    assert cap.in_reserve_band is band
    assert cap.capped_tenant({"a": 1, "b": 2}) == expected
