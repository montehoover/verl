from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class CredentialRecord:
    salt: bytes
    password_hash: bytes
    iterations: int = 210000


# In-memory credential store for demonstration/testing purposes.
# Replace with a persistent store (e.g., database) in production.
_CREDENTIAL_STORE: Dict[str, CredentialRecord] = {}


def _hash_password(password: str, salt: bytes, iterations: int) -> bytes:
    """Derive a password hash using PBKDF2-HMAC-SHA256."""
    if not isinstance(password, str):
        raise TypeError("password must be a str")
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)


def set_employee_password(employee_id: str, password: str, iterations: int = 210000) -> None:
    """
    Create or update the stored password for an employee in the in-memory store.
    This is provided to allow populating the store for testing/integration.
    """
    if not isinstance(employee_id, str):
        raise TypeError("employee_id must be a str")
    if not isinstance(password, str):
        raise TypeError("password must be a str")
    if not employee_id:
        raise ValueError("employee_id must not be empty")
    if not password:
        raise ValueError("password must not be empty")

    salt = secrets.token_bytes(16)
    pwd_hash = _hash_password(password, salt, iterations)
    _CREDENTIAL_STORE[employee_id] = CredentialRecord(salt=salt, password_hash=pwd_hash, iterations=iterations)


def verify_employee_credentials(employee_id: str, password: str) -> bool:
    """
    Verify an employee's credentials.

    Args:
        employee_id: Unique identifier for the employee.
        password: Plaintext password to verify.

    Returns:
        True if the provided credentials are valid, False otherwise.
    """
    if not isinstance(employee_id, str) or not isinstance(password, str):
        return False
    rec = _CREDENTIAL_STORE.get(employee_id)
    if rec is None:
        return False
    derived = _hash_password(password, rec.salt, rec.iterations)
    return hmac.compare_digest(derived, rec.password_hash)


def _clear_in_memory_store() -> None:
    """Clear the in-memory credential store (intended for testing)."""
    _CREDENTIAL_STORE.clear()
