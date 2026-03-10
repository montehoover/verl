from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from typing import Dict, TypedDict, Optional

__all__ = ["authenticate_user", "add_user", "USER_STORE"]

_DEFAULT_ALGORITHM = "sha256"
_DEFAULT_ITERATIONS = 600_000
_SALT_LENGTH = 16


class _UserRecord(TypedDict):
    algorithm: str
    iterations: int
    salt_b64: str
    hash_b64: str


# In-memory user store (placeholder). Replace with a secure database/secret manager in production.
USER_STORE: Dict[str, _UserRecord] = {}


def _pbkdf2_hash(password: str, salt: bytes, iterations: int, algorithm: str) -> bytes:
    return hashlib.pbkdf2_hmac(algorithm, password.encode("utf-8"), salt, iterations)


# Precomputed dummy parameters for unknown users to mitigate user-enumeration timing attacks.
_DUMMY_SALT = b"\x00" * _SALT_LENGTH
_DUMMY_ITERATIONS = _DEFAULT_ITERATIONS
_DUMMY_ALGORITHM = _DEFAULT_ALGORITHM


def add_user(user_id: str, password: str) -> None:
    """
    Utility to create/update a user with a hashed password in the in-memory USER_STORE.
    In production, persist to a secure database and never store plaintext passwords.
    """
    if not isinstance(user_id, str) or not isinstance(password, str):
        raise TypeError("user_id and password must be strings")

    salt = secrets.token_bytes(_SALT_LENGTH)
    pw_hash = _pbkdf2_hash(password, salt, _DEFAULT_ITERATIONS, _DEFAULT_ALGORITHM)

    USER_STORE[user_id] = _UserRecord(
        algorithm=_DEFAULT_ALGORITHM,
        iterations=_DEFAULT_ITERATIONS,
        salt_b64=base64.b64encode(salt).decode("ascii"),
        hash_b64=base64.b64encode(pw_hash).decode("ascii"),
    )


def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticate a user by verifying the provided password against the stored password hash.

    Args:
        user_id: The unique user identifier.
        password: The plaintext password supplied by the user.

    Returns:
        True if authentication succeeds; False otherwise.
    """
    if not isinstance(user_id, str) or not isinstance(password, str):
        return False

    record: Optional[_UserRecord] = USER_STORE.get(user_id)

    if record is None:
        # Perform a dummy hash computation to keep timing similar for unknown users.
        dummy_hash = _pbkdf2_hash(password, _DUMMY_SALT, _DUMMY_ITERATIONS, _DUMMY_ALGORITHM)
        # Compare against a fixed-length zeroed bytes value.
        return hmac.compare_digest(dummy_hash, b"\x00" * len(dummy_hash))

    try:
        algorithm = record["algorithm"]
        iterations = int(record["iterations"])
        salt = base64.b64decode(record["salt_b64"])
        stored_hash = base64.b64decode(record["hash_b64"])
    except Exception:
        # Corrupt record; fail closed.
        return False

    computed_hash = _pbkdf2_hash(password, salt, iterations, algorithm)
    return hmac.compare_digest(stored_hash, computed_hash)
