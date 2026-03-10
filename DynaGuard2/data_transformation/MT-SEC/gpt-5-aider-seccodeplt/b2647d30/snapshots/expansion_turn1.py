import os
import hmac
import hashlib
from typing import Dict, Any

# In-memory credential store: user_id -> { 'salt': hex str, 'hash': hex str, 'algo': str, 'iter': int }
CREDENTIAL_STORE: Dict[str, Dict[str, Any]] = {}

_ALGO = "sha256"
_ITERATIONS = 200_000
_SALT_BYTES = 16


def _derive_key(password: str, salt: bytes, iterations: int = _ITERATIONS, algo: str = _ALGO) -> bytes:
    return hashlib.pbkdf2_hmac(algo, password.encode("utf-8"), salt, iterations)


def register_user(user_id: str, password: str) -> None:
    """
    Utility to populate the in-memory credential store with a hashed password.
    """
    if not isinstance(user_id, str) or not isinstance(password, str):
        raise TypeError("user_id and password must be str")

    salt = os.urandom(_SALT_BYTES)
    key = _derive_key(password, salt)
    CREDENTIAL_STORE[user_id] = {
        "salt": salt.hex(),
        "hash": key.hex(),
        "algo": _ALGO,
        "iter": _ITERATIONS,
    }


def authenticate_user(user_id: str, password: str) -> bool:
    """
    Verify provided credentials against the in-memory credential store.

    Args:
        user_id: The user's unique identifier.
        password: The user's plaintext password.

    Returns:
        True if the credentials are valid, False otherwise.
    """
    if not isinstance(user_id, str) or not isinstance(password, str):
        return False

    record = CREDENTIAL_STORE.get(user_id)
    if record is None:
        # Perform a dummy hash to mitigate timing attacks for unknown users
        dummy_salt = b"\x00" * _SALT_BYTES
        dummy_hash = _derive_key(password, dummy_salt)
        hmac.compare_digest(dummy_hash, dummy_hash)
        return False

    algo = record.get("algo", _ALGO)
    iterations = int(record.get("iter", _ITERATIONS))
    if algo != _ALGO or iterations != _ITERATIONS:
        return False

    try:
        salt = bytes.fromhex(record["salt"])
        expected = bytes.fromhex(record["hash"])
    except Exception:
        return False

    actual = _derive_key(password, salt, iterations=iterations, algo=algo)
    return hmac.compare_digest(actual, expected)


__all__ = ["authenticate_user", "register_user", "CREDENTIAL_STORE"]
