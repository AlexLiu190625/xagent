"""SDK API key generation, parsing, and bcrypt verification utilities.

The SDK key format is ``xag_<6-char prefix>_<32-char secret>``.

  - The **prefix** is a public-safe lookup handle. It is what we index in
    ``agent_api_keys.key_prefix`` and what we allow callers (and logs) to
    see in cleartext. It does NOT confer any auth power on its own.

  - The **secret** is the unguessable half. The server only ever stores
    ``bcrypt(full_key, cost=12)`` in ``agent_api_keys.key_hash``; the
    plaintext secret leaves the response of ``POST /api/agents/{id}/api-key``
    exactly once and is never persisted server-side.

Why the alphabet excludes ``_`` (underscore):
    Parse logic splits on ``_`` to recover (prefix, secret). If either half
    contained ``_`` we'd get ambiguous splits. Restricting both halves to
    ``[A-Za-z0-9]`` makes the format unambiguous and copy-paste safe.

Why bcrypt and not a faster hash:
    A leaked database row exposes ``key_hash``. With a fast hash (SHA-256,
    HMAC) an attacker can brute-force the 32-char secret in tractable time.
    bcrypt cost=12 puts a single attempt at ~100ms on commodity hardware
    so the 62^32 keyspace becomes computationally infeasible.

Why ``verify_dummy`` exists:
    To prevent a timing oracle: an attacker who can measure response
    latency must not be able to distinguish "prefix not found" from
    "prefix found, secret wrong". Both paths must spend the same ~100ms
    bcrypt work. ``verify_dummy`` is called on the prefix-miss branch to
    keep timings symmetric.
"""

import logging
import secrets
import string
from typing import Optional, Tuple

import bcrypt
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ===== Key format constants =====

# Brand prefix that identifies an xagent SDK key. Matches Stripe's ``sk_*``,
# OpenAI's ``sk-*``, Anthropic's ``sk-ant-*`` pattern -- a stable token in
# logs and CI scanners that says "this is an xagent credential".
KEY_BRAND = "xag"

# Length of the public-safe lookup handle. 6 chars over a 62-symbol alphabet
# gives ~57 billion combinations; the partial unique index on
# ``agent_api_keys.key_prefix`` will catch any collision and we retry.
KEY_PREFIX_LENGTH = 6

# Length of the secret half. 32 chars over 62 symbols gives ~190 bits of
# entropy, well beyond what bcrypt cost=12 can amplify and well beyond
# brute-force range.
KEY_SECRET_LENGTH = 32

# Alphabet for both halves. ASCII letters + digits, no separators, no
# punctuation, no homoglyphs to worry about for copy-paste.
KEY_ALPHABET = string.ascii_letters + string.digits

# bcrypt work factor. Each +1 doubles the cost. cost=12 yields ~100ms per
# checkpw on typical hardware in 2026, the same band Stripe / Auth0 use.
BCRYPT_COST = 12

# How many times we re-roll the prefix on the (extremely unlikely) chance
# that the random prefix is already taken. The keyspace makes real
# collisions vanishing; this cap exists only to guarantee termination if
# somebody mis-mocks the DB in tests.
PREFIX_COLLISION_RETRIES = 5


def _generate_random_string(length: int) -> str:
    """Return a cryptographically random string of *length* chars from KEY_ALPHABET.

    Uses ``secrets`` (CSPRNG-backed) rather than ``random``. ``secrets.choice``
    in a tight loop is the canonical way to draw a fixed-length token from a
    custom alphabet -- ``secrets.token_urlsafe`` would let through ``-`` and
    ``_`` which we explicitly forbid.
    """
    return "".join(secrets.choice(KEY_ALPHABET) for _ in range(length))


# Pre-computed bcrypt hash used by ``verify_dummy``. Computed at module
# load so we only pay the bcrypt cost once at startup; subsequent dummy
# verifications just re-run ``checkpw`` against this constant hash. The
# value being "dummy" is irrelevant -- checkpw against a known-bad input
# always returns False and burns the right amount of CPU.
_DUMMY_HASH = bcrypt.hashpw(b"dummy", bcrypt.gensalt(rounds=BCRYPT_COST))


def generate_api_key(db: Optional[Session]) -> Tuple[str, str, str]:
    """Generate a new SDK API key and its bcrypt hash.

    The caller is responsible for INSERTing a row into ``agent_api_keys``
    using the returned ``key_prefix`` and ``key_hash``; this function is
    stateless and does not write anything to the database itself. The
    ``db`` parameter is only used to probe for prefix collisions so we
    can re-roll before letting the INSERT hit the unique-index trap.

    Args:
        db: SQLAlchemy session, used to ``SELECT`` against
            ``agent_api_keys.key_prefix``. Pass ``None`` to skip the
            collision check (useful in unit tests where you don't have a
            session and the keyspace makes a real collision impossible).

    Returns:
        Tuple ``(full_key, key_prefix, key_hash)`` where:
          - ``full_key`` is the plaintext ``xag_<prefix>_<secret>``; show
            this to the user once and never persist it.
          - ``key_prefix`` is the 6-char lookup handle to store in
            ``agent_api_keys.key_prefix``.
          - ``key_hash`` is the bcrypt-hashed full key (utf-8 str) to
            store in ``agent_api_keys.key_hash``.

    Raises:
        RuntimeError: if we hit ``PREFIX_COLLISION_RETRIES`` consecutive
            prefix collisions. In production this is effectively
            impossible (62^6 keyspace, sparse population). In tests this
            usually means a mock is returning the same row every time.
    """
    # Import locally to avoid a circular dependency: core/utils/ shouldn't
    # depend on web/models/ at module-import time.
    from xagent.web.models.agent_api_key import AgentApiKey

    for attempt in range(PREFIX_COLLISION_RETRIES):
        prefix = _generate_random_string(KEY_PREFIX_LENGTH)

        # Collision check is best-effort. The DB unique index is the real
        # authority; this just avoids the ugly path of catching
        # IntegrityError on the caller's INSERT.
        if db is not None:
            existing = (
                db.query(AgentApiKey).filter(AgentApiKey.key_prefix == prefix).first()
            )
            if existing is not None:
                logger.warning(
                    f"API key prefix collision on attempt {attempt + 1}, "
                    f"re-rolling (prefix={prefix})"
                )
                continue

        secret = _generate_random_string(KEY_SECRET_LENGTH)
        full_key = f"{KEY_BRAND}_{prefix}_{secret}"

        # bcrypt.hashpw expects bytes in, bytes out. We store utf-8 str
        # in the DB so callers don't have to remember to decode every
        # time they read the column.
        key_hash = bcrypt.hashpw(
            full_key.encode("utf-8"), bcrypt.gensalt(rounds=BCRYPT_COST)
        ).decode("utf-8")

        return full_key, prefix, key_hash

    # Reaching this point means PREFIX_COLLISION_RETRIES consecutive draws
    # all collided. Real-world prob is essentially 0; treat as a hard error
    # so test mocks don't silently loop forever.
    raise RuntimeError(
        f"Failed to generate a unique key prefix after "
        f"{PREFIX_COLLISION_RETRIES} attempts"
    )


def parse_api_key(raw: str) -> Optional[Tuple[str, str]]:
    """Split a raw API key string into ``(prefix, secret)`` if well-formed.

    A well-formed key has shape ``xag_<6 chars>_<32 chars>`` where both
    char-class halves draw from ``KEY_ALPHABET``. Anything else returns
    ``None``; the caller treats that as ``invalid_api_key`` (same response
    we give for a wrong secret -- never tell the attacker which check
    failed).

    Args:
        raw: The full key string as received from the ``Authorization``
            header (already stripped of any ``Bearer `` prefix by the
            HTTP layer).

    Returns:
        ``(prefix, secret)`` on success; ``None`` on any format mismatch.

    Notes:
        Never include ``raw`` in log lines. If you must log the failure,
        log only the brand or the prefix segment after a ``parse`` has
        already separated it -- never the full string and never the secret.
    """
    if not isinstance(raw, str) or not raw:
        return None

    # ``rsplit`` with maxsplit=2 guards against any future tweak where
    # the brand or prefix segments grow ``_`` characters; today the
    # alphabet excludes ``_`` so an exact split into 3 parts is what we
    # require.
    parts = raw.split("_")
    if len(parts) != 3:
        return None

    brand, prefix, secret = parts
    if brand != KEY_BRAND:
        return None
    if len(prefix) != KEY_PREFIX_LENGTH or len(secret) != KEY_SECRET_LENGTH:
        return None

    # Final guard: each half is strictly in KEY_ALPHABET. A future change
    # that allows broader chars in storage but not in parsing would fail
    # closed here, which is the right direction (refuse weird keys rather
    # than accept and hash them).
    alphabet_set = set(KEY_ALPHABET)
    if any(c not in alphabet_set for c in prefix):
        return None
    if any(c not in alphabet_set for c in secret):
        return None

    return prefix, secret


def verify_api_key(raw: str, stored_hash: str) -> bool:
    """Verify a raw plaintext API key against a stored bcrypt hash.

    ``bcrypt.checkpw`` is constant-time within its work factor; passing
    a wrong secret takes the same ~100ms as a correct one, so an
    attacker cannot use timing to distinguish near-misses.

    Args:
        raw: The full plaintext key. Caller must NOT log this.
        stored_hash: The bcrypt-hashed key read from
            ``agent_api_keys.key_hash``.

    Returns:
        True if and only if bcrypt confirms the secret matches.

    Notes:
        Any malformed input (empty raw, malformed stored_hash) returns
        False rather than raising; callers treat both as auth failure
        and bcrypt's exception cases are hard to distinguish from "wrong
        password" semantically anyway.
    """
    if not raw or not stored_hash:
        return False
    try:
        return bcrypt.checkpw(raw.encode("utf-8"), stored_hash.encode("utf-8"))
    except (ValueError, TypeError):
        # ValueError: bcrypt rejects mangled hash strings.
        # TypeError: defensive -- shouldn't happen given the encode() above.
        return False


def verify_dummy() -> None:
    """Burn the same ~100ms of bcrypt work that ``verify_api_key`` would.

    Call this on the auth path where the prefix lookup missed the index --
    i.e. the row doesn't exist or is revoked. Without this, the response
    time for "no such prefix" (fast: index miss returns immediately) would
    differ visibly from the time for "prefix found, bcrypt rejected"
    (slow: ~100ms), letting an attacker enumerate which prefixes exist
    just by timing.

    The return value is intentionally ignored; we only care about the
    side effect of CPU time spent.
    """
    bcrypt.checkpw(b"dummy", _DUMMY_HASH)
