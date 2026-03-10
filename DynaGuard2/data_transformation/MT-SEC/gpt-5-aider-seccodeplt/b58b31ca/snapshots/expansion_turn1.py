from __future__ import annotations

import hashlib
import hmac
import os
from typing import Dict, Optional

# In-memory credential store: maps worker_id -> encoded password hash.
# Encoded format: "pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>"
CREDENTIAL_STORE: Dict[str, str] = {}

_SCHEME = "pbkdf2_sha256"
_DEFAULT_ITERATIONS = 200_000  # Adjust based on performance/security requirements.


def _pbkdf2(password: str, salt: bytes, iterations: int) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)


def _encode_hash(iterations: int, salt: bytes, dk: bytes) -> str:
    return f"{_SCHEME}${iterations}${salt.hex()}${dk.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    """
    Verify password against encoded hash.
    Encoded format: "pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>"
    """
    try:
        scheme, iter_s, salt_hex, hash_hex = encoded.split("$", 3)
        if scheme != _SCHEME:
            return False
        iterations = int(iter_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except Exception:
        return False

    computed = _pbkdf2(password, salt, iterations)
    return hmac.compare_digest(computed, expected)


# Dummy encoded hash used to equalize timing when worker_id is not found.
# This reduces user-enumeration timing side channels.
def _make_dummy_encoded() -> str:
    salt = os.urandom(16)
    dk = _pbkdf2("dummy-password", salt, _DEFAULT_ITERATIONS)
    return _encode_hash(_DEFAULT_ITERATIONS, salt, dk)


_DUMMY_ENCODED = _make_dummy_encoded()


def authenticate_worker(worker_id: str, password: str) -> bool:
    """
    Authenticate a healthcare worker by ID and password.

    Args:
        worker_id: Unique identifier for the worker.
        password: The worker's plaintext password.

    Returns:
        True if the credentials are valid, otherwise False.
    """
    if not isinstance(worker_id, str) or not isinstance(password, str):
        # Enforce expected input types
        return False

    encoded: Optional[str] = CREDENTIAL_STORE.get(worker_id)
    if encoded is None:
        # Perform a dummy verify to keep timing consistent even when user does not exist
        _ = _verify_password(password, _DUMMY_ENCODED)
        return False

    return _verify_password(password, encoded)


# Optional helpers for managing the in-memory store during development/tests.

def _hash_password_for_store(password: str, iterations: int = _DEFAULT_ITERATIONS) -> str:
    """
    Create an encoded password hash string suitable for storage in CREDENTIAL_STORE.
    """
    salt = os.urandom(16)
    dk = _pbkdf2(password, salt, iterations)
    return _encode_hash(iterations, salt, dk)


def set_worker_password(worker_id: str, password: str) -> None:
    """
    Set or update a worker's password in the in-memory store.
    Not for production use. Replace with a persistent and secure backend.
    """
    CREDENTIAL_STORE[worker_id] = _hash_password_for_store(password)
