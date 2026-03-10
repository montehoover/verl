import os
import hmac
import hashlib
from typing import Dict, Tuple, Set

# In-memory user credential store
# Maps user_id -> (salt_bytes, iterations, hash_bytes)
_USER_STORE: Dict[str, Tuple[bytes, int, bytes]] = {}

_DEFAULT_ITERATIONS = 120_000
_SALT_LEN = 16

# In-memory project roles store
# Maps project_id -> { user_id -> set(roles) }
_PROJECT_ROLES: Dict[str, Dict[str, Set[str]]] = {}


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


__all__ = ["authenticate_user", "set_user_password", "verify_project_membership", "grant_project_role"]
