import os
import hmac
import hashlib
from typing import Dict, Tuple

# Internal in-memory credential store:
# Maps staff_id -> (salt: bytes, password_hash: bytes, iterations: int)
_CREDENTIALS: Dict[str, Tuple[bytes, bytes, int]] = {}

# Constants used to mitigate timing attacks when staff_id is not found
_DUMMY_SALT = b'\x00' * 16
_DUMMY_ITERS = 100_000


def _normalize_staff_id(staff_id: str) -> str:
    return staff_id.strip()


def _hash_password(password: str, salt: bytes, iterations: int = 200_000) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
        dklen=32,
    )


def authenticate_staff(staff_id: str, password: str) -> bool:
    """
    Verify staff credentials.

    Args:
        staff_id: The unique staff identifier.
        password: The plaintext password to verify.

    Returns:
        True if credentials are valid; False otherwise.
    """
    if not isinstance(staff_id, str) or not isinstance(password, str):
        return False

    staff_id_norm = _normalize_staff_id(staff_id)
    if not staff_id_norm:
        # Avoid early returns without comparable work
        _ = _hash_password(password, _DUMMY_SALT, _DUMMY_ITERS)
        return False

    record = _CREDENTIALS.__dict__.get(staff_id_norm) if hasattr(_CREDENTIALS, "__dict__") else _CREDENTIALS.get(staff_id_norm)
    if record is None:
        # Do comparable work to mitigate timing differences
        _ = _hash_password(password, _DUMMY_SALT, _DUMMY_ITERS)
        return False

    salt, stored_hash, iterations = record
    computed = _hash_password(password, salt, iterations)
    return hmac.compare_digest(stored_hash, computed)


def register_staff(staff_id: str, password: str, iterations: int = 200_000) -> None:
    """
    Register or update staff credentials in the in-memory store.
    This is provided for integration/testing; production should use a persistent, secure store.

    Args:
        staff_id: The unique staff identifier.
        password: The plaintext password to store (will be salted and hashed).
        iterations: PBKDF2 iterations (default 200,000).
    """
    if not isinstance(staff_id, str) or not isinstance(password, str):
        raise TypeError("staff_id and password must be strings")

    staff_id_norm = _normalize_staff_id(staff_id)
    if not staff_id_norm:
        raise ValueError("staff_id cannot be empty")

    if iterations < 100_000:
        raise ValueError("iterations should be >= 100,000 for security")

    salt = os.urandom(16)
    pwd_hash = _hash_password(password, salt, iterations)
    _CREDENTIALS[staff_id_norm] = (salt, pwd_hash, iterations)


def clear_credentials() -> None:
    """
    Clear all credentials from the in-memory store.
    Useful for tests.
    """
    _CREDENTIALS.clear()


__all__ = ["authenticate_staff", "register_staff", "clear_credentials"]
