from __future__ import annotations

import hmac
import os
import hashlib
from typing import Dict

# In-memory user store: maps user_id -> encoded password hash string
# The encoded format is: pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>
_USER_STORE: Dict[str, str] = {}


def _hash_password(password: str, salt: bytes, iterations: int) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)


def _encode_password(password: str, iterations: int = 260000) -> str:
    salt = os.urandom(16)
    dk = _hash_password(password, salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${dk.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algo, iter_str, salt_hex, hash_hex = encoded.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        iterations = int(iter_str)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except Exception:
        return False

    candidate = _hash_password(password, salt, iterations)
    return hmac.compare_digest(candidate, expected)


def register_user(user_id: str, password: str) -> None:
    """
    Registers or updates a user's credentials in the in-memory store.
    Stores a salted PBKDF2-SHA256 hash, not the plaintext password.
    """
    if not isinstance(user_id, str) or not isinstance(password, str) or not user_id:
        raise ValueError("user_id and password must be non-empty strings")
    _USER_STORE[user_id] = _encode_password(password)


def set_user_store(store: Dict[str, str]) -> None:
    """
    Replaces the internal user store with the provided mapping of
    user_id -> encoded password hash (pbkdf2_sha256 format).
    """
    global _USER_STORE
    if not isinstance(store, dict):
        raise ValueError("store must be a dict mapping user_id to encoded password")
    _USER_STORE = dict(store)


def authenticate_user(user_id: str, password: str) -> bool:
    """
    Returns True if the provided user_id/password combination is valid
    against the current in-memory user store, False otherwise.
    """
    if not isinstance(user_id, str) or not isinstance(password, str) or not user_id:
        return False
    encoded = _USER_STORE.get(user_id)
    if not encoded:
        return False
    return _verify_password(password, encoded)
