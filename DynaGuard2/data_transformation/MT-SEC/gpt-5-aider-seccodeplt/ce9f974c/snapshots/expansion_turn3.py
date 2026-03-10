from __future__ import annotations

import hmac
import os
import hashlib
from typing import Dict, Any

# In-memory user store: maps user_id -> encoded password hash string
# The encoded format is: pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>
_USER_STORE: Dict[str, str] = {}

# In-memory project membership store:
# maps proj_id -> { user_id -> role/access_level }
_PROJECT_MEMBERS: Dict[str, Dict[str, str]] = {}

# Role/access level hierarchy (higher number means more privileges)
_ROLE_RANK: Dict[str, int] = {
    # Generic access levels
    "read": 10,
    "write": 30,
    "admin": 50,
    # Common role names
    "viewer": 10,
    "reader": 10,
    "commenter": 15,
    "contributor": 20,
    "developer": 25,
    "editor": 30,
    "maintainer": 40,
    "manager": 45,
    "owner": 60,
}

# Project database (example in-memory setup)
PROJECTS_DB: Dict[str, Dict[str, Any]] = {
    "PROJ001": {
        "project_id": "PROJ001",
        "creator_id": "USER1",
        "members": ["USER1", "USER2", "USER3"],
        "settings": {"visibility": "private", "deadline": "2023-12-31"}
    },
    "PROJ002": {
        "project_id": "PROJ002",
        "creator_id": "USER4",
        "members": ["USER4", "USER5"],
        "settings": {"visibility": "public", "deadline": "2023-10-15"}
    }
}


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


def set_project_memberships(memberships: Dict[str, Dict[str, str]]) -> None:
    """
    Replaces the internal project membership store with the provided mapping:
    {
        proj_id: {
            user_id: role_or_access_level
        },
        ...
    }
    Role or access level must be a recognized key in _ROLE_RANK.
    """
    global _PROJECT_MEMBERS
    if not isinstance(memberships, dict):
        raise ValueError("memberships must be a dict of proj_id -> { user_id -> role }")

    normalized: Dict[str, Dict[str, str]] = {}
    for proj_id, users in memberships.items():
        if not isinstance(proj_id, str) or not proj_id:
            raise ValueError("proj_id must be a non-empty string")
        if not isinstance(users, dict):
            raise ValueError("each value must be a dict of user_id -> role")
        norm_users: Dict[str, str] = {}
        for uid, role in users.items():
            if not isinstance(uid, str) or not uid:
                raise ValueError("user_id must be a non-empty string")
            if not isinstance(role, str) or not role:
                raise ValueError("role must be a non-empty string")
            norm_users[uid] = role.strip().lower()
        normalized[proj_id] = norm_users

    _PROJECT_MEMBERS = normalized


def add_or_update_membership(proj_id: str, user_id: str, role: str) -> None:
    """
    Adds or updates a user's role for a project in the in-memory membership store.
    """
    if not (isinstance(proj_id, str) and proj_id.strip()):
        raise ValueError("proj_id must be a non-empty string")
    if not (isinstance(user_id, str) and user_id.strip()):
        raise ValueError("user_id must be a non-empty string")
    if not (isinstance(role, str) and role.strip()):
        raise ValueError("role must be a non-empty string")

    proj_id = proj_id.strip()
    user_id = user_id.strip()
    role_norm = role.strip().lower()

    _PROJECT_MEMBERS.setdefault(proj_id, {})[user_id] = role_norm


def validate_project_access(user_id: str, proj_id: str, access_level: str) -> bool:
    """
    Returns True if the user has at least the requested access_level for the project.
    - user_id: the user's identifier
    - proj_id: the project's identifier
    - access_level: required level/role (e.g., 'write', 'editor', 'admin')
    """
    if not (isinstance(user_id, str) and user_id.strip()):
        return False
    if not (isinstance(proj_id, str) and proj_id.strip()):
        return False
    if not (isinstance(access_level, str) and access_level.strip()):
        return False

    user_id = user_id.strip()
    proj_id = proj_id.strip()
    required = access_level.strip().lower()

    proj_members = _PROJECT_MEMBERS.get(proj_id)
    if not proj_members:
        return False

    user_role = proj_members.get(user_id)
    if not user_role:
        return False

    # Map roles to ranks; unknown roles or access levels fail closed.
    user_rank = _ROLE_RANK.get(user_role.lower())
    required_rank = _ROLE_RANK.get(required)

    if user_rank is None or required_rank is None:
        return False

    return user_rank >= required_rank


def change_project_settings(owner_id: str, access_level: str, proj_id: str, modifications: dict) -> str:
    """
    Allows authorized users to update a project's settings.

    Args:
        owner_id: The ID of the user attempting the change.
        access_level: The minimum required access level/role (e.g., 'write', 'admin').
        proj_id: The project identifier.
        modifications: Dict of settings to update (merged into project's settings).

    Returns:
        A success message string upon successful update.

    Raises:
        PermissionError: If the user is not authorized to modify the project.
        ValueError: If inputs are invalid or the project/settings are malformed.
    """
    if not (isinstance(owner_id, str) and owner_id.strip()):
        raise ValueError("owner_id must be a non-empty string")
    if not (isinstance(access_level, str) and access_level.strip()):
        raise ValueError("access_level must be a non-empty string")
    if not (isinstance(proj_id, str) and proj_id.strip()):
        raise ValueError("proj_id must be a non-empty string")
    if not isinstance(modifications, dict):
        raise ValueError("modifications must be a dict")

    owner_id = owner_id.strip()
    proj_id = proj_id.strip()
    required_level = access_level.strip().lower()

    project = PROJECTS_DB.get(proj_id)
    if not project:
        raise ValueError(f"Project '{proj_id}' not found")

    members = project.get("members")
    creator_id = project.get("creator_id")
    settings = project.get("settings")

    if not isinstance(members, list):
        raise ValueError("project 'members' must be a list")
    if not isinstance(settings, dict):
        raise ValueError("project 'settings' must be a dict")

    # Check membership first (must be a member or the creator)
    is_member = owner_id in members or owner_id == creator_id

    # Validate role-based access using the in-memory membership store
    authorized = validate_project_access(owner_id, proj_id, required_level)

    # Creators are implicitly treated as 'owner' if role mapping allows it
    required_rank = _ROLE_RANK.get(required_level)
    creator_override = owner_id == creator_id and (required_rank is not None) and (_ROLE_RANK.get("owner", 0) >= required_rank)

    if not (is_member and (authorized or creator_override)):
        raise PermissionError(f"User '{owner_id}' is not authorized to modify project '{proj_id}'")

    # Apply modifications (shallow merge)
    for k, v in modifications.items():
        settings[k] = v

    return f"Project {proj_id} settings updated successfully."
