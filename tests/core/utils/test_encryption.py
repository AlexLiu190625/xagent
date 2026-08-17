import pytest
from cryptography.fernet import Fernet

from xagent.core.utils.encryption import (
    EncryptionDecodeError,
    _get_encryption_key,
    decrypt_env_dict,
    decrypt_env_dict_strict,
    decrypt_value,
    decrypt_value_strict,
    encrypt_env_dict,
    encrypt_value,
    get_cipher,
)


def test_encrypt_decrypt_roundtrip():
    original_value = "my_super_secret_value"
    encrypted = encrypt_value(original_value)

    assert encrypted != original_value
    assert isinstance(encrypted, str)

    decrypted = decrypt_value(encrypted)
    assert decrypted == original_value


def test_encrypt_empty_value():
    assert encrypt_value("") == ""
    assert encrypt_value(None) is None


def test_encrypt_value_idempotent():
    """Re-encrypting an already-encrypted value is a no-op (no double-encryption)."""
    once = encrypt_value("secret")
    assert encrypt_value(once) == once
    assert decrypt_value(encrypt_value(once)) == "secret"


def test_encrypt_value_encrypts_fake_ciphertext_prefix():
    """A plaintext that merely looks like a Fernet token is still encrypted.

    Guards against the old prefix-only heuristic that would store such a value
    in plaintext at rest.
    """
    looks_like_token = "gAAAAABnot-a-real-token"
    encrypted = encrypt_value(looks_like_token)
    assert encrypted != looks_like_token
    assert decrypt_value(encrypted) == looks_like_token


def test_decrypt_empty_value():
    assert decrypt_value("") == ""
    assert decrypt_value(None) is None


def test_decrypt_invalid_token():
    # Provide an invalid token, should catch InvalidToken and return the original string
    invalid_encrypted = "invalid_token_value"
    result = decrypt_value(invalid_encrypted)
    assert result == "invalid_token_value"


def test_get_encryption_key_no_env(monkeypatch):
    monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "development")
    key = _get_encryption_key()
    assert key == "RQMpe38gK3m0szjpSmTNw_sP3Y54r6hDc6JewBoPKXc="


def test_get_encryption_key_production_missing_key(monkeypatch):
    monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(
        ValueError, match="ENCRYPTION_KEY environment variable is not set"
    ):
        _get_encryption_key()


def test_get_encryption_key_with_env(monkeypatch):
    test_key = "some_test_key_base64_encoded="
    monkeypatch.setenv("ENCRYPTION_KEY", test_key)
    key = _get_encryption_key()
    assert key == test_key


STRICT_KEY_A = "RQMpe38gK3m0szjpSmTNw_sP3Y54r6hDc6JewBoPKXc="
STRICT_KEY_B = Fernet.generate_key().decode()


@pytest.fixture
def use_key(monkeypatch):
    """Switch the module to a given encryption key for one test.

    Setting ENCRYPTION_KEY alone has no effect once get_cipher() has run:
    it is lru_cached. Clear on every switch, and again on teardown so no
    test key survives into another test in the same process.
    """

    def _use(key):
        monkeypatch.setenv("ENCRYPTION_KEY", key)
        get_cipher.cache_clear()

    yield _use
    get_cipher.cache_clear()


def test_decrypt_value_strict_raises_on_foreign_token(use_key):
    """A token from another key raises, where the lenient helper stays silent."""
    use_key(STRICT_KEY_A)
    token = encrypt_value("secret")

    use_key(STRICT_KEY_B)
    assert decrypt_value(token) == token
    with pytest.raises(EncryptionDecodeError):
        decrypt_value_strict(token)


def test_decrypt_value_strict_roundtrip(use_key):
    use_key(STRICT_KEY_A)
    assert decrypt_value_strict(encrypt_value("secret")) == "secret"


@pytest.mark.parametrize(
    "plaintext",
    ["invalid_token_value", "gAAAAABnot-a-real-token", "sk-abc123", "plain text"],
)
def test_decrypt_value_strict_passes_plaintext_through(use_key, plaintext):
    """Values that are not token-shaped are returned unchanged, as before.

    No key can open them, so classifying them as plaintext loses nothing.
    """
    use_key(STRICT_KEY_A)
    assert decrypt_value_strict(plaintext) == plaintext


def test_decrypt_value_strict_empty_value(use_key):
    use_key(STRICT_KEY_A)
    assert decrypt_value_strict("") == ""
    assert decrypt_value_strict(None) is None


def test_decrypt_value_strict_error_is_value_error(use_key):
    """Callers already catching ValueError keep working."""
    use_key(STRICT_KEY_A)
    token = encrypt_value("secret")

    use_key(STRICT_KEY_B)
    with pytest.raises(ValueError):
        decrypt_value_strict(token)


def test_decrypt_value_strict_error_omits_the_value(use_key):
    use_key(STRICT_KEY_A)
    token = encrypt_value("secret")

    use_key(STRICT_KEY_B)
    with pytest.raises(EncryptionDecodeError) as excinfo:
        decrypt_value_strict(token)
    assert token not in str(excinfo.value)


def test_decrypt_env_dict_strict(use_key):
    use_key(STRICT_KEY_A)
    assert decrypt_env_dict_strict(encrypt_env_dict({"A": "1", "B": "2"})) == {
        "A": "1",
        "B": "2",
    }
    assert decrypt_env_dict_strict(None) is None
    assert decrypt_env_dict_strict({}) == {}

    mixed = {"A": encrypt_value("1"), "B": "plain", "C": 5, "D": None}
    assert decrypt_env_dict_strict(mixed) == {
        "A": "1",
        "B": "plain",
        "C": 5,
        "D": None,
    }


def test_decrypt_env_dict_strict_raises_on_foreign_token(use_key):
    """One unreadable entry fails the whole map, instead of leaking ciphertext."""
    use_key(STRICT_KEY_A)
    foreign = encrypt_value("1")

    use_key(STRICT_KEY_B)
    env = {"A": encrypt_value("ok"), "B": foreign}
    assert decrypt_env_dict(env)["B"] == foreign
    with pytest.raises(EncryptionDecodeError):
        decrypt_env_dict_strict(env)
