from __future__ import annotations

import base64
import hashlib
import hmac
import os
from typing import Dict, TypedDict, Set

_HASH_NAME = "sha256"
_DEFAULT_ITERATIONS = 210_000
_SALT_LEN = 16


class _CredentialRecord(TypedDict):
    salt: str  # base64 encoded
    hash: str  # base64 encoded
    iterations: int


# In-memory credential store. Populate using store_staff_password().
_STAFF_CREDENTIALS_DB: Dict[str, _CredentialRecord] = {}

# Dummy values to mitigate timing attacks for unknown users.
_DUMMY_SALT = b"\x00" * _SALT_LEN
_DUMMY_HASH = hashlib.pbkdf2_hmac(_HASH_NAME, b"invalid", _DUMMY_SALT, _DEFAULT_ITERATIONS)


def _derive_key(password: str, salt: bytes, iterations: int) -> bytes:
    return hashlib.pbkdf2_hmac(_HASH_NAME, password.encode("utf-8"), salt, iterations)


def _hash_password(password: str, iterations: int = _DEFAULT_ITERATIONS) -> _CredentialRecord:
    salt = os.urandom(_SALT_LEN)
    dk = _derive_key(password, salt, iterations)
    return {
        "salt": base64.b64encode(salt).decode("ascii"),
        "hash": base64.b64encode(dk).decode("ascii"),
        "iterations": iterations,
    }


def store_staff_password(staff_id: str, password: str, *, iterations: int = _DEFAULT_ITERATIONS) -> None:
    record = _hash_password(password, iterations)
    _STAFF_CREDENTIALS_DB[staff_id] = record


def authenticate_staff(staff_id: str, password: str) -> bool:
    """
    Authenticate a staff member by verifying the provided password against the stored hash.

    Args:
        staff_id: The staff identifier.
        password: The plaintext password to verify.

    Returns:
        True if credentials are valid; False otherwise.
    """
    record = _STAFF_CREDENTIALS_DB.get(staff_id)
    if record is None:
        # Do dummy work to mitigate timing attacks for unknown users.
        candidate = _derive_key(password, _DUMMY_SALT, _DEFAULT_ITERATIONS)
        hmac.compare_digest(candidate, _DUMMY_HASH)
        return False

    try:
        salt = base64.b64decode(record["salt"])
        stored_hash = base64.b64decode(record["hash"])
        iterations = int(record.get("iterations", _DEFAULT_ITERATIONS))
    except Exception:
        # Malformed record, treat as authentication failure with dummy work.
        candidate = _derive_key(password, _DUMMY_SALT, _DEFAULT_ITERATIONS)
        hmac.compare_digest(candidate, _DUMMY_HASH)
        return False

    candidate = _derive_key(password, salt, iterations)
    return hmac.compare_digest(candidate, stored_hash)


# ------------------------
# Authorization primitives
# ------------------------

# In-memory access control list (ACL): patient_id -> set of staff_ids with access.
_STAFF_PATIENT_ACCESS: Dict[str, Set[str]] = {}

# Roles with full access to all patient records.
_ADMIN_ROLES: Set[str] = {"admin", "privacy_officer", "security_officer"}


def grant_access_to_patient(staff_id: str, patient_id: str) -> None:
    """
    Grant explicit access for a staff member to a patient's data.
    """
    _STAFF_PATIENT_ACCESS.setdefault(patient_id, set()).add(staff_id)


def revoke_access_to_patient(staff_id: str, patient_id: str) -> None:
    """
    Revoke explicit access for a staff member to a patient's data.
    """
    s = _STAFF_PATIENT_ACCESS.get(patient_id)
    if not s:
        return
    s.discard(staff_id)
    if not s:
        # Clean up empty entry
        _STAFF_PATIENT_ACCESS.pop(patient_id, None)


def _consttime_contains(values: Set[str], candidate: str) -> bool:
    """
    Membership test that avoids early returns to reduce timing signal.
    """
    found = False
    for v in values:
        if hmac.compare_digest(v, candidate):
            found = True
    return found


def check_staff_permission(staff_id: str, staff_role: str, patient_id: str) -> bool:
    """
    Determine if a staff member has permission to access a patient's data.

    Policy:
    - Admin-like roles have access to all patient records.
    - Otherwise, access is allowed only if the staff member has been explicitly granted
      access to the specified patient via grant_access_to_patient().

    Args:
        staff_id: The staff member's identifier.
        staff_role: The staff member's role (e.g., "admin", "doctor", "nurse").
        patient_id: The patient identifier.

    Returns:
        True if access is permitted; False otherwise.
    """
    role = staff_role.strip().lower()
    if role in _ADMIN_ROLES:
        return True

    allowed_staff = _STAFF_PATIENT_ACCESS.get(patient_id)
    if not allowed_staff:
        # No explicit grants for this patient.
        return False

    return _consttime_contains(allowed_staff, staff_id)


# ------------------------
# Patient data store and retrieval
# ------------------------

PATIENT_RECORDS = {
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


def retrieve_patient_data(staff_id: str, staff_role: str, patient_id: str) -> str:
    """
    Retrieve a patient's medical data if the requesting staff member is authorized.

    Args:
        staff_id: The staff identifier.
        staff_role: The staff role (e.g., "doctor", "nurse", "admin").
        patient_id: The patient identifier.

    Returns:
        The patient's medical data as a string.

    Raises:
        PermissionError: If the staff member is not authorized to access the patient's data.
        KeyError: If the patient record does not exist.
    """
    record = PATIENT_RECORDS.get(patient_id)
    if record is None:
        raise KeyError(f"Patient record not found: {patient_id}")

    if not check_staff_permission(staff_id, staff_role, patient_id):
        raise PermissionError(f"Staff '{staff_id}' is not authorized to access patient '{patient_id}'")

    return record["medical_data"]
