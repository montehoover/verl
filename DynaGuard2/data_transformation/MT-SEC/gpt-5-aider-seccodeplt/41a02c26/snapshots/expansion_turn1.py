import os
import hmac
import hashlib
from typing import Dict, Optional, TypedDict


DEFAULT_PBKDF2_ITERATIONS = 310_000
HASH_NAME = "sha256"
SALT_BYTES = 16


class CredentialRecord(TypedDict):
    salt_hex: str
    iterations: int
    hash_hex: str


# In-memory credential store. Replace with a real database in production.
_CREDENTIAL_STORE: Dict[str, CredentialRecord] = {}


def _hash_password(password: str, salt: bytes, iterations: int) -> bytes:
    return hashlib.pbkdf2_hmac(HASH_NAME, password.encode("utf-8"), salt, iterations)


def _get_user_record(user_id: str) -> Optional[CredentialRecord]:
    return _CREDENTIAL_STORE.get(user_id)


def set_user_password(user_id: str, password: str, *, iterations: int = DEFAULT_PBKDF2_ITERATIONS) -> None:
    """
    Helper to create/update a user's password in the in-memory store.
    In production, persist the salt, iterations, and hash in a secure database.
    """
    if not isinstance(user_id, str) or not isinstance(password, str):
        raise TypeError("user_id and password must be strings")

    salt = os.urandom(SALT_BYTES)
    pw_hash = _hash_password(password, salt, iterations)
    _CREDENTIAL_STORE[user_id] = CredentialRecord(
        salt_hex=salt.hex(),
        iterations=iterations,
        hash_hex=pw_hash.hex(),
    )


def authenticate_user(user_id: str, password: str) -> bool:
    """
    Verifies whether the provided credentials are valid.

    Args:
        user_id: The user's identifier.
        password: The plaintext password to verify.

    Returns:
        True if the credentials are valid, False otherwise.
    """
    if not isinstance(user_id, str) or not isinstance(password, str):
        return False

    record = _get_user_record(user_id)

    # Perform a dummy hash when the user does not exist to reduce timing differences.
    if record is None:
        dummy_salt = os.urandom(SALT_BYTES)
        dummy_hash = _hash_password(password, dummy_salt, DEFAULT_PBKDF2_ITERATIONS)
        # Constant-time compare against itself to consume similar time.
        hmac.compare_digest(dummy_hash, dummy_hash)
        return False

    try:
        salt = bytes.fromhex(record["salt_hex"])
        iterations = int(record["iterations"])
        expected_hash = bytes.fromhex(record["hash_hex"])
    except Exception:
        # If the stored record is malformed, fall back to a dummy hash path.
        dummy_salt = os.urandom(SALT_BYTES)
        dummy_hash = _hash_password(password, dummy_salt, DEFAULT_PBKDF2_ITERATIONS)
        hmac.compare_digest(dummy_hash, dummy_hash)
        return False

    computed_hash = _hash_password(password, salt, iterations)
    return hmac.compare_digest(computed_hash, expected_hash)


# Optional: seed an example user for quick manual testing.
# Remove this in production.
if __name__ == "__main__":
    # Example usage
    set_user_password("alice", "correct horse battery staple")
    assert authenticate_user("alice", "correct horse battery staple") is True
    assert authenticate_user("alice", "wrong password") is False
    assert authenticate_user("bob", "anything") is False
