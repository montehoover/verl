import os
import hmac
import hashlib
from typing import Dict, TypedDict


class _UserRecord(TypedDict):
    iterations: int
    salt: str  # hex-encoded
    hash: str  # hex-encoded


# In-memory credential store.
# Map: user_id -> { iterations, salt(hex), hash(hex) }
USER_CREDENTIALS: Dict[str, _UserRecord] = {}

# PBKDF2 parameters
_ITERATIONS_DEFAULT = 210_000
_HASH_NAME = "sha256"
_DKLEN = 32  # 256-bit


def _hash_password(password: str, salt: bytes, iterations: int) -> bytes:
    """
    Derive a key from the provided password using PBKDF2-HMAC.
    """
    return hashlib.pbkdf2_hmac(
        _HASH_NAME,
        password.encode("utf-8"),
        salt,
        iterations,
        dklen=_DKLEN,
    )


# Dummy constants to mitigate user-enumeration timing differences.
# These are used when a user_id is not found to keep operations consistent.
_DUMMY_SALT = os.urandom(16)
_DUMMY_HASH = _hash_password("dummy_password", _DUMMY_SALT, _ITERATIONS_DEFAULT)


def set_user_password(user_id: str, password: str, iterations: int = _ITERATIONS_DEFAULT) -> None:
    """
    Create or update the given user's password in the local in-memory store.
    Passwords are salted and stored as PBKDF2-HMAC hashes (hex-encoded).
    """
    if not isinstance(user_id, str) or not isinstance(password, str):
        raise TypeError("user_id and password must be strings")
    if not user_id:
        raise ValueError("user_id must be non-empty")
    if not password:
        raise ValueError("password must be non-empty")
    if not isinstance(iterations, int) or iterations < 50_000:
        raise ValueError("iterations must be an integer >= 50_000")

    salt = os.urandom(16)
    derived = _hash_password(password, salt, iterations)
    USER_CREDENTIALS[user_id] = {
        "iterations": iterations,
        "salt": salt.hex(),
        "hash": derived.hex(),
    }


def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticate a user by verifying the provided password against the stored
    salted PBKDF2-HMAC hash. Returns True if credentials are valid, False otherwise.

    This function performs a constant-time comparison and uses a dummy
    verification path for unknown users to reduce timing side-channels.
    """
    if not isinstance(user_id, str) or not isinstance(password, str):
        return False
    if not user_id or not password:
        return False

    record = USER_CREDENTIALS.get(user_id)

    if record is not None:
        iterations = int(record["iterations"])
        salt = bytes.fromhex(record["salt"])
        stored_hash = bytes.fromhex(record["hash"])
    else:
        iterations = _ITERATIONS_DEFAULT
        salt = _DUMMY_SALT
        stored_hash = _DUMMY_HASH

    candidate_hash = _hash_password(password, salt, iterations)
    matches = hmac.compare_digest(candidate_hash, stored_hash)

    # Only succeed when a real record exists and the hashes match
    return bool(record is not None and matches)
