from __future__ import annotations

import hashlib
import hmac
import os
from typing import Dict, Tuple

# Internal store: member_id -> (salt, password_hash)
_USER_DB: Dict[str, Tuple[bytes, bytes]] = {}

_ITERATIONS = 200_000


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)


def authenticate_user(member_id: str, password: str) -> bool:
    if not isinstance(member_id, str) or not isinstance(password, str):
        return False

    record = _USER_DB.get(member_id)
    if record is None:
        return False

    salt, stored_hash = record
    computed_hash = _hash_password(password, salt)
    return hmac.compare_digest(stored_hash, computed_hash)


def register_user(member_id: str, password: str) -> None:
    salt = os.urandom(16)
    pw_hash = _hash_password(password, salt)
    _USER_DB[member_id] = (salt, pw_hash)


__all__ = ["authenticate_user", "register_user"]
