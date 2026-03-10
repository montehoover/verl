from __future__ import annotations

import hashlib
import hmac
import os
from typing import Dict, Optional, Tuple

# Internal in-memory user store.
# Maps user_id -> (iterations, salt_bytes, password_hash_bytes)
_USER_DB: Dict[str, Tuple[int, bytes, bytes]] = {}

_PBKDF2_ALGO = "sha256"
_DEFAULT_ITERATIONS = 200_000
_SALT_BYTES = 16


def register_user(user_id: str, password: str, iterations: int = _DEFAULT_ITERATIONS) -> None:
    """
    Register or update a user in the in-memory store with a salted PBKDF2-HMAC hash.
    Not intended for production use; replace with a real user store as needed.
    """
    if not isinstance(user_id, str) or not isinstance(password, str):
        raise TypeError("user_id and password must be strings")

    salt = os.urandom(_SALT_BYTES)
    pwd_hash = hashlib.pbkdf2_hmac(
        _PBKDF2_ALGO,
        password.encode("utf-8"),
        salt,
        iterations,
    )
    _USER_DB[user_id] = (iterations, salt, pwd_hash)


def authenticate_user(user_id: str, password: str) -> bool:
    """
    Verify if the provided credentials are valid.

    Args:
        user_id: The user's identifier (str).
        password: The user's plaintext password (str).

    Returns:
        True if credentials are valid, False otherwise.
    """
    if not isinstance(user_id, str) or not isinstance(password, str):
        return False

    record: Optional[Tuple[int, bytes, bytes]] = _USER_DB.get(user_id)
    if record is None:
        return False

    iterations, salt, stored_hash = record

    try:
        candidate_hash = hashlib.pbkdf2_hmac(
            _PBKDF2_ALGO,
            password.encode("utf-8"),
            salt,
            iterations,
        )
    except Exception:
        return False

    return hmac.compare_digest(candidate_hash, stored_hash)


__all__ = ["authenticate_user", "register_user"]
