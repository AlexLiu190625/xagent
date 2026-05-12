"""Unit tests for ``xagent.core.utils.api_key``.

Covers the three key contracts callers rely on:

  - Generation produces a syntactically correct key (brand + alphabet
    + lengths) and a bcrypt hash that verifies against it.
  - Parse is strict -- any deviation from the format returns None
    rather than partial / lenient parses, so a bad header never
    reaches the bcrypt step.
  - verify_api_key matches bcrypt's contract (true on match, false
    on miss, false on garbage input rather than raising).

Plus a couple of robustness checks:

  - verify_dummy spends roughly the same time as a real verify_api_key
    call, so an attacker can't enumerate prefixes by timing.
  - generate_api_key retries on prefix collision and gives up cleanly
    if a mock keeps colliding.
"""

import re
import time
from unittest.mock import MagicMock

import pytest

from xagent.core.utils.api_key import (
    BCRYPT_COST,
    KEY_ALPHABET,
    KEY_BRAND,
    KEY_PREFIX_LENGTH,
    KEY_SECRET_LENGTH,
    PREFIX_COLLISION_RETRIES,
    generate_api_key,
    parse_api_key,
    verify_api_key,
    verify_dummy,
)

# ===== generate_api_key =====


def test_generate_format() -> None:
    """Generated key matches xag_<6 alnum>_<32 alnum>; halves stay within alphabet."""
    full, prefix, key_hash = generate_api_key(db=None)

    # Brand + segment lengths
    assert full.startswith(f"{KEY_BRAND}_")
    parts = full.split("_")
    assert len(parts) == 3
    assert parts[0] == KEY_BRAND
    assert len(parts[1]) == KEY_PREFIX_LENGTH
    assert len(parts[2]) == KEY_SECRET_LENGTH

    # Returned prefix matches the embedded prefix segment
    assert parts[1] == prefix
    assert len(prefix) == KEY_PREFIX_LENGTH

    # Alphabet constraint -- no underscores, dashes, or other glyphs slip in
    alphabet_re = re.compile(f"^[{re.escape(KEY_ALPHABET)}]+$")
    assert alphabet_re.fullmatch(parts[1])
    assert alphabet_re.fullmatch(parts[2])

    # bcrypt hash is the standard $2b$ prefix with our cost factor
    assert key_hash.startswith(f"$2b${BCRYPT_COST:02d}$")


def test_generate_persists_only_hash() -> None:
    """The returned hash verifies the full key; the secret is never returned twice."""
    full, _prefix, key_hash = generate_api_key(db=None)
    # Round-trip: hash verifies its source
    assert verify_api_key(full, key_hash) is True
    # Sanity: hash does NOT verify a different key
    assert verify_api_key(full + "X", key_hash) is False


def test_generate_prefix_collision_retry() -> None:
    """First call collides (mock returns existing row), second succeeds."""
    mock_db = MagicMock()
    # Sequence of .filter(...).first() results:
    #   1st call -> simulated collision (returns a truthy "row" object)
    #   2nd call -> no collision (returns None)
    mock_db.query.return_value.filter.return_value.first.side_effect = [
        object(),  # collision on first prefix
        None,  # second prefix is free
    ]

    full, prefix, key_hash = generate_api_key(db=mock_db)
    assert len(prefix) == KEY_PREFIX_LENGTH
    assert verify_api_key(full, key_hash) is True
    # We called the DB exactly twice (one collision + one success)
    assert mock_db.query.return_value.filter.return_value.first.call_count == 2


def test_generate_gives_up_after_retry_cap() -> None:
    """All PREFIX_COLLISION_RETRIES draws colliding -> RuntimeError, not infinite loop."""
    mock_db = MagicMock()
    # Always return a truthy "existing row" to force perpetual collision
    mock_db.query.return_value.filter.return_value.first.return_value = object()

    with pytest.raises(RuntimeError, match="unique key prefix"):
        generate_api_key(db=mock_db)

    # Exhausted the retry budget exactly
    assert (
        mock_db.query.return_value.filter.return_value.first.call_count
        == PREFIX_COLLISION_RETRIES
    )


# ===== parse_api_key =====


def test_parse_valid() -> None:
    """Well-formed key splits cleanly into (prefix, secret)."""
    full, prefix, _hash = generate_api_key(db=None)
    parsed = parse_api_key(full)
    assert parsed is not None
    parsed_prefix, parsed_secret = parsed
    assert parsed_prefix == prefix
    assert len(parsed_secret) == KEY_SECRET_LENGTH
    # Reassembly round-trip
    assert f"{KEY_BRAND}_{parsed_prefix}_{parsed_secret}" == full


@pytest.mark.parametrize(
    "bad",
    [
        "",  # empty
        "no_brand_prefix",  # wrong brand
        "xag_only_two_parts",  # 3 underscores -> 4 parts? actually wrong split
        "xag_short_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",  # prefix len != 6
        "xag_ABCDEF_short",  # secret len != 32
        "xag_ABCDEF_" + "!" * KEY_SECRET_LENGTH,  # secret contains forbidden char
    ],
)
def test_parse_invalid(bad: str) -> None:
    """Any deviation from xag_<6>_<32 alnum> returns None, not a partial parse."""
    assert parse_api_key(bad) is None


def test_parse_non_string_input() -> None:
    """Non-string / None input returns None rather than raising."""
    assert parse_api_key(None) is None  # type: ignore[arg-type]
    assert parse_api_key(123) is None  # type: ignore[arg-type]


# ===== verify_api_key =====


def test_verify_correct() -> None:
    """A freshly generated key verifies against its own hash."""
    full, _prefix, key_hash = generate_api_key(db=None)
    assert verify_api_key(full, key_hash) is True


def test_verify_wrong_secret() -> None:
    """Tampering with even a single secret char fails verification."""
    full, _prefix, key_hash = generate_api_key(db=None)
    # Flip the last char of the secret to something else still in the alphabet
    flipped = full[:-1] + ("A" if full[-1] != "A" else "B")
    assert verify_api_key(flipped, key_hash) is False


def test_verify_empty_inputs_return_false() -> None:
    """Empty / malformed inputs return False rather than raising."""
    assert verify_api_key("", "") is False
    assert verify_api_key("xag_ABCDEF_anything", "") is False
    assert verify_api_key("", "$2b$12$notarealhash") is False


def test_verify_garbage_hash_returns_false() -> None:
    """Malformed bcrypt hash strings produce False, not ValueError leaking out."""
    full, _prefix, _hash = generate_api_key(db=None)
    assert verify_api_key(full, "not-a-bcrypt-hash") is False


# ===== verify_dummy =====


def test_verify_dummy_runs_without_raising() -> None:
    """Dummy verification is callable and returns None."""
    assert verify_dummy() is None


def test_verify_dummy_timing_similar_to_verify() -> None:
    """verify_dummy() runs roughly as long as a real verify_api_key() call.

    We don't need tight bounds; the threat model is "an attacker can tell
    fast (index miss) from slow (bcrypt run) responses". A ratio inside
    [0.3, 3.0] is good enough -- bcrypt timing is dominated by the cost
    factor, not by what's being verified. Generous bounds keep CI happy
    on overcommitted runners.
    """
    full, _prefix, key_hash = generate_api_key(db=None)

    # Warm any lazy bcrypt init so we don't measure first-call overhead
    verify_api_key(full, key_hash)
    verify_dummy()

    t0 = time.perf_counter()
    verify_api_key(full, key_hash)
    real_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    verify_dummy()
    dummy_elapsed = time.perf_counter() - t0

    ratio = dummy_elapsed / real_elapsed
    assert 0.3 < ratio < 3.0, (
        f"verify_dummy timing diverged from verify_api_key: "
        f"real={real_elapsed * 1000:.1f}ms, dummy={dummy_elapsed * 1000:.1f}ms, "
        f"ratio={ratio:.2f}"
    )
