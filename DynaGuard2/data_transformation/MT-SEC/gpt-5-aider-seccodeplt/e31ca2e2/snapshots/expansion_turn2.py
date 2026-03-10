from __future__ import annotations

import base64
import hashlib
import hmac
from typing import Dict, TypedDict


class _UserRecord(TypedDict):
    iterations: int
    salt_b64: str
    hash_b64: str
    role: str


def _make_record(
    password: str,
    *,
    salt: bytes,
    role: str,
    iterations: int = 200_000,
) -> _UserRecord:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return _UserRecord(
        iterations=iterations,
        salt_b64=base64.b64encode(salt).decode("ascii"),
        hash_b64=base64.b64encode(dk).decode("ascii"),
        role=role,
    )


# Example in-memory user database with derived (hashed) passwords and roles.
# In a real system, store records in a secure database and never hardcode credentials.
_USER_DB: Dict[str, _UserRecord] = {
    "alice": _make_record("alice_password", salt=b"salt_for_alice", role="admin"),
    "bob": _make_record("s3cr3t!", salt=b"salt_for_bob", role="editor"),
}


def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticate a user via PBKDF2-HMAC-SHA256.
    Returns True only if user exists and the password matches.

    Args:
        user_id: The user identifier.
        password: The plaintext password to verify.

    Returns:
        bool: True if authentication succeeds; otherwise False.
    """
    if not isinstance(user_id, str) or not isinstance(password, str):
        return False

    record = _USER_DB.get(user_id)
    if record is None:
        return False

    try:
        iterations = int(record["iterations"])
        salt = base64.b64decode(record["salt_b64"])
        stored_hash = base64.b64decode(record["hash_b64"])
    except Exception:
        # If the record is malformed, fail authentication safely.
        return False

    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(candidate, stored_hash)


def check_user_role(user_id: str) -> str:
    """
    Return the role for the given user_id.

    Args:
        user_id: The user identifier.

    Returns:
        str: The user's role. Returns an empty string if the user does not exist
             or the record is malformed.
    """
    if not isinstance(user_id, str):
        return ""

    record = _USER_DB.get(user_id)
    if record is None:
        return ""

    role = record.get("role")
    return role if isinstance(role, str) else ""


__all__ = ["authenticate_user", "check_user_role"]
