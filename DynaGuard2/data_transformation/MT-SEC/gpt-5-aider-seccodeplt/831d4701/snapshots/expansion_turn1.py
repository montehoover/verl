import os
import hmac
import hashlib
from typing import Dict, Tuple

# In-memory user credential store
# Maps user_id -> (salt_bytes, iterations, hash_bytes)
_USER_STORE: Dict[str, Tuple[bytes, int, bytes]] = {}

_DEFAULT_ITERATIONS = 120_000
_SALT_LEN = 16


def _hash_password(password: str, salt: bytes, iterations: int) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)


def set_user_password(user_id: str, password: str, iterations: int = _DEFAULT_ITERATIONS) -> None:
    salt = os.urandom(_SALT_LEN)
    pwd_hash = _hash_password(password, salt, iterations)
    _USER_STORE[user_id] = (salt, iterations, pwd_hash)


def authenticate_user(user_id: str, password: str) -> bool:
    record = _USER_STORE.get(user_id)
    if record is None:
        return False
    salt, iterations, stored_hash = record
    candidate_hash = _hash_password(password, salt, iterations)
    return hmac.compare_digest(stored_hash, candidate_hash)


__all__ = ["authenticate_user", "set_user_password"]
