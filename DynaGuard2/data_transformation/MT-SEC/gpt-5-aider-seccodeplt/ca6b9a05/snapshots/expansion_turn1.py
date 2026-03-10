import os
import hmac
import hashlib
from typing import Dict

# Configuration for password hashing
_HASH_NAME = "sha256"
_PBKDF2_ITERATIONS = 200_000
_SALT_LENGTH = 16  # bytes

# In-memory user credential store.
# Each entry: { "salt": hex_string, "password_hash": hex_string }
_USER_DATABASE: Dict[str, Dict[str, str]] = {}


def _pbkdf2_hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(_HASH_NAME, password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)


def _verify_password(password: str, salt: bytes, expected_hash: bytes) -> bool:
    test_hash = _pbkdf2_hash(password, salt)
    return hmac.compare_digest(test_hash, expected_hash)


def _secure_dummy_verify(password: str) -> None:
    # Run a dummy verification to help mitigate timing differences for unknown users
    dummy_salt = os.urandom(_SALT_LENGTH)
    dummy_hash = _pbkdf2_hash(password if isinstance(password, str) else "", dummy_salt)
    hmac.compare_digest(dummy_hash, b"\x00" * len(dummy_hash))


def register_user(user_id: str, password: str) -> None:
    if not isinstance(user_id, str) or not isinstance(password, str):
        raise TypeError("user_id and password must be strings")
    if not user_id:
        raise ValueError("user_id must not be empty")
    if not password:
        raise ValueError("password must not be empty")

    salt = os.urandom(_SALT_LENGTH)
    pwd_hash = _pbkdf2_hash(password, salt)
    _USER_DATABASE[user_id] = {
        "salt": salt.hex(),
        "password_hash": pwd_hash.hex(),
    }


def authenticate_user(user_id: str, password: str) -> bool:
    if not isinstance(user_id, str) or not isinstance(password, str):
        return False
    if not user_id or not password:
        return False

    record = _USER_DATABASE.get(user_id)
    if record is None:
        _secure_dummy_verify(password)
        return False

    try:
        salt = bytes.fromhex(record["salt"])
        expected_hash = bytes.fromhex(record["password_hash"])
    except (KeyError, ValueError):
        _secure_dummy_verify(password)
        return False

    return _verify_password(password, salt, expected_hash)
