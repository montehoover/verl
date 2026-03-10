from __future__ import annotations

import hashlib
import hmac
import os
import copy
from typing import Dict, Tuple, Set, Optional, Any

# Internal store: member_id -> (salt, password_hash)
_USER_DB: Dict[str, Tuple[bytes, bytes]] = {}

# Internal project stores
# project_key -> settings dict
_PROJECTS: Dict[str, Dict[str, Any]] = {}
# project_key -> set of member_ids
_PROJECT_MEMBERS: Dict[str, Set[str]] = {}

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


def create_project(project_key: str, settings: Optional[Dict[str, Any]] = None) -> None:
    if not isinstance(project_key, str):
        raise TypeError("project_key must be a string")
    if settings is None:
        settings = {}
    elif not isinstance(settings, dict):
        raise TypeError("settings must be a dict or None")
    _PROJECTS[project_key] = dict(settings)
    _PROJECT_MEMBERS.setdefault(project_key, set())


def add_member_to_project(project_key: str, member_id: str) -> None:
    if not isinstance(project_key, str):
        raise TypeError("project_key must be a string")
    if not isinstance(member_id, str):
        raise TypeError("member_id must be a string")
    _PROJECT_MEMBERS.setdefault(project_key, set()).add(member_id)


def get_project_settings(project_key: str, member_id: str) -> Optional[Dict[str, Any]]:
    if not isinstance(project_key, str) or not isinstance(member_id, str):
        return None
    settings = _PROJECTS.get(project_key)
    if settings is None:
        return None
    members = _PROJECT_MEMBERS.get(project_key, set())
    if member_id not in members:
        return None
    return copy.deepcopy(settings)


__all__ = [
    "authenticate_user",
    "register_user",
    "create_project",
    "add_member_to_project",
    "get_project_settings",
]
