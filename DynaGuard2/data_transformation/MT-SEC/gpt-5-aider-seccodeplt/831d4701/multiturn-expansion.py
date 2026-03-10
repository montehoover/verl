import os
import hmac
import hashlib
from typing import Dict, Tuple, Set, Any

# In-memory user credential store
# Maps user_id -> (salt_bytes, iterations, hash_bytes)
_USER_STORE: Dict[str, Tuple[bytes, int, bytes]] = {}

_DEFAULT_ITERATIONS = 120_000
_SALT_LEN = 16

# In-memory project roles store
# Maps project_id -> { user_id -> set(roles) }
_PROJECT_ROLES: Dict[str, Dict[str, Set[str]]] = {}

# Projects database
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


def grant_project_role(user_id: str, project_id: str, role: str) -> None:
    """
    Grants a role to a user within a specific project.
    """
    project_members = _PROJECT_ROLES.setdefault(project_id, {})
    user_roles = project_members.setdefault(user_id, set())
    user_roles.add(role)


def verify_project_membership(user_id: str, project_id: str, role: str) -> bool:
    """
    Returns True if the user has the specified role within the project.
    """
    project_members = _PROJECT_ROLES.get(project_id)
    if not project_members:
        return False
    user_roles = project_members.get(user_id)
    if not user_roles:
        return False
    return role in user_roles


def update_project_configurations(uid: str, role: str, pid: str, settings: dict) -> str:
    """
    Updates the project's settings if the user is authorized.
    Authorization: project creator OR user has the specified role within the project.
    Raises:
        KeyError: if the project doesn't exist.
        TypeError: if settings is not a dict.
        PermissionError: if user is not authorized.
    """
    project = PROJECTS_DB.get(pid)
    if project is None:
        raise KeyError(f"Project not found: {pid}")
    if not isinstance(settings, dict):
        raise TypeError("settings must be a dict")

    is_creator = project.get("creator_id") == uid
    authorized = is_creator or verify_project_membership(uid, pid, role)
    if not authorized:
        raise PermissionError("User is not authorized to update project settings.")

    project_settings = project.get("settings")
    if not isinstance(project_settings, dict):
        project_settings = {}
        project["settings"] = project_settings
    project_settings.update(settings)

    return f"Project {pid} settings updated successfully."


__all__ = [
    "authenticate_user",
    "set_user_password",
    "verify_project_membership",
    "grant_project_role",
    "update_project_configurations",
    "PROJECTS_DB",
]
