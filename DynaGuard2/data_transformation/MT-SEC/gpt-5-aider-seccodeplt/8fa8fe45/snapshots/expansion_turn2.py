from __future__ import annotations

import hashlib
import hmac
import os
from typing import Any, Dict, Optional, TypedDict


class UserRecord(TypedDict):
    salt: str  # hex-encoded salt
    hash: str  # hex-encoded password hash


class ProjectRecord(TypedDict):
    settings: Dict[str, Any]


# In-memory user store for demonstration.
# Replace with a real database or user service in production.
USER_STORE: Dict[str, UserRecord] = {}

# In-memory projects database for demonstration.
# Replace with a real database in production.
PROJECTS_DB: Dict[str, ProjectRecord] = {}


# Tunable parameter: increase iterations for stronger security depending on performance budget.
PASSWORD_HASH_ITERATIONS = 200_000


def _hash_password(password: str, salt: bytes) -> str:
    """
    Derive a secure password hash using PBKDF2-HMAC-SHA256.
    Returns a hex-encoded hash string.
    """
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_HASH_ITERATIONS)
    return dk.hex()


def authenticate_user(user_id: str, password: str) -> bool:
    """
    Verify user credentials.

    Args:
        user_id: Unique user identifier.
        password: Plaintext password provided by the user.

    Returns:
        True if the credentials are valid; False otherwise.
    """
    record = USER_STORE.get(user_id)
    if not record:
        return False

    try:
        salt = bytes.fromhex(record["salt"])
        expected_hash = record["hash"]
    except Exception:
        # Malformed record; treat as authentication failure.
        return False

    computed_hash = _hash_password(password, salt)
    return hmac.compare_digest(computed_hash, expected_hash)


def get_project_settings(proj_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve the current settings for a project.

    Args:
        proj_id: Unique project identifier.

    Returns:
        A dict of project settings if the project exists; otherwise None.
    """
    record = PROJECTS_DB.get(proj_id)
    if not record:
        return None
    settings = record.get("settings")
    if isinstance(settings, dict):
        # Return a shallow copy to avoid external mutation of the in-memory DB.
        return dict(settings)
    return None


# Optional helper functions for local testing and seeding.
def register_user(user_id: str, password: str) -> None:
    """
    Register a user in the in-memory store.
    NOTE: For production, replace with persistent storage and proper uniqueness checks.
    """
    if not isinstance(user_id, str) or not isinstance(password, str):
        raise TypeError("user_id and password must be strings")

    salt = os.urandom(16)
    pwd_hash = _hash_password(password, salt)
    USER_STORE[user_id] = UserRecord(salt=salt.hex(), hash=pwd_hash)
