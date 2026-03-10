from __future__ import annotations

import hashlib
import hmac
import os
from typing import Dict, Optional, Set, Iterable

# In-memory credential store: maps worker_id -> encoded password hash.
# Encoded format: "pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>"
CREDENTIAL_STORE: Dict[str, str] = {}

# In-memory role store: maps worker_id -> set of roles (normalized lowercase).
ROLE_STORE: Dict[str, Set[str]] = {}

# In-memory patient records store.
PATIENT_RECORDS: Dict[str, Dict[str, object]] = {
    "P1001": {
        "patient_id": "P1001",
        "assigned_staff": ["D101", "N201"],
        "medical_data": "Medical history for patient P1001"
    },
    "P1002": {
        "patient_id": "P1002",
        "assigned_staff": ["D102", "N202"],
        "medical_data": "Medical history for patient P1002"
    }
}

_SCHEME = "pbkdf2_sha256"
_DEFAULT_ITERATIONS = 200_000  # Adjust based on performance/security requirements.

_ADMIN_ROLE = "admin"


def _pbkdf2(password: str, salt: bytes, iterations: int) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)


def _encode_hash(iterations: int, salt: bytes, dk: bytes) -> str:
    return f"{_SCHEME}${iterations}${salt.hex()}${dk.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    """
    Verify password against encoded hash.
    Encoded format: "pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>"
    """
    try:
        scheme, iter_s, salt_hex, hash_hex = encoded.split("$", 3)
        if scheme != _SCHEME:
            return False
        iterations = int(iter_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except Exception:
        return False

    computed = _pbkdf2(password, salt, iterations)
    return hmac.compare_digest(computed, expected)


# Dummy encoded hash used to equalize timing when worker_id is not found.
# This reduces user-enumeration timing side channels.
def _make_dummy_encoded() -> str:
    salt = os.urandom(16)
    dk = _pbkdf2("dummy-password", salt, _DEFAULT_ITERATIONS)
    return _encode_hash(_DEFAULT_ITERATIONS, salt, dk)


_DUMMY_ENCODED = _make_dummy_encoded()


def authenticate_worker(worker_id: str, password: str) -> bool:
    """
    Authenticate a healthcare worker by ID and password.

    Args:
        worker_id: Unique identifier for the worker.
        password: The worker's plaintext password.

    Returns:
        True if the credentials are valid, otherwise False.
    """
    if not isinstance(worker_id, str) or not isinstance(password, str):
        # Enforce expected input types
        return False

    encoded: Optional[str] = CREDENTIAL_STORE.get(worker_id)
    if encoded is None:
        # Perform a dummy verify to keep timing consistent even when user does not exist
        _ = _verify_password(password, _DUMMY_ENCODED)
        return False

    return _verify_password(password, encoded)


def _normalize_role(role: str) -> str:
    return role.strip().lower()


def check_worker_role(worker_id: str, required_role: str) -> bool:
    """
    Check whether a worker has the required role.

    Admins are considered to have all roles.

    Args:
        worker_id: Unique identifier for the worker.
        required_role: Role required to perform an action (e.g., "doctor", "nurse", "admin").

    Returns:
        True if the worker has the role (or is an admin), otherwise False.
    """
    if not isinstance(worker_id, str) or not isinstance(required_role, str):
        return False

    req = _normalize_role(required_role)
    if not req:
        return False

    roles = ROLE_STORE.get(worker_id)
    if not roles:
        return False

    # Normalize roles defensively in case of external mutation.
    norm_roles = {_normalize_role(r) for r in roles if isinstance(r, str)}
    if req == _ADMIN_ROLE:
        return _ADMIN_ROLE in norm_roles

    return req in norm_roles or _ADMIN_ROLE in norm_roles


def get_patient_data(worker_id: str, worker_role: str, record_id: str) -> str:
    """
    Retrieve patient medical data if the worker is authorized.

    Authorization rules:
    - Admins can access any patient record.
    - Doctors and Nurses can access records only if they are assigned to the patient.
    - The worker must possess the claimed role (validated via ROLE_STORE).

    Args:
        worker_id: Unique identifier for the worker (e.g., "D101", "N201").
        worker_role: The role being used for access ("doctor", "nurse", or "admin").
        record_id: Patient record identifier (e.g., "P1001").

    Returns:
        The patient's medical_data string.

    Raises:
        KeyError: If the record_id does not exist.
        PermissionError: If the worker is not authorized to access the record.
        TypeError: If the argument types are invalid.
    """
    if not all(isinstance(x, str) for x in (worker_id, worker_role, record_id)):
        raise TypeError("worker_id, worker_role, and record_id must be strings")

    role_norm = _normalize_role(worker_role)

    # Look up the patient record; raises KeyError if not found.
    record = PATIENT_RECORDS[record_id]

    # Ensure the worker truly has the claimed role.
    if not check_worker_role(worker_id, role_norm):
        raise PermissionError("Worker is not authorized to access this patient record.")

    # Admins can access any record.
    if role_norm == _ADMIN_ROLE:
        return record["medical_data"]  # type: ignore[index]

    # Doctors and nurses must be assigned to the patient.
    if role_norm in ("doctor", "nurse"):
        assigned = record.get("assigned_staff", [])  # type: ignore[assignment]
        if isinstance(assigned, list) and worker_id in assigned:
            return record["medical_data"]  # type: ignore[index]

    # Any other case is unauthorized.
    raise PermissionError("Worker is not authorized to access this patient record.")


# Optional helpers for managing the in-memory store during development/tests.

def _hash_password_for_store(password: str, iterations: int = _DEFAULT_ITERATIONS) -> str:
    """
    Create an encoded password hash string suitable for storage in CREDENTIAL_STORE.
    """
    salt = os.urandom(16)
    dk = _pbkdf2(password, salt, iterations)
    return _encode_hash(iterations, salt, dk)


def set_worker_password(worker_id: str, password: str) -> None:
    """
    Set or update a worker's password in the in-memory store.
    Not for production use. Replace with a persistent and secure backend.
    """
    CREDENTIAL_STORE[worker_id] = _hash_password_for_store(password)


def set_worker_roles(worker_id: str, roles: Iterable[str]) -> None:
    """
    Set or update a worker's roles in the in-memory role store.
    Roles are normalized to lowercase and trimmed.
    """
    ROLE_STORE[worker_id] = {
        _normalize_role(r) for r in roles
        if isinstance(r, str) and _normalize_role(r)
    }
