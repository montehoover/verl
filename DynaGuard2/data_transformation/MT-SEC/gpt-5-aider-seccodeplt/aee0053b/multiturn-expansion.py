import os
import hmac
import hashlib
from typing import Dict, Tuple, Set

# Internal in-memory credential store:
# Maps staff_id -> (salt: bytes, password_hash: bytes, iterations: int)
_CREDENTIALS: Dict[str, Tuple[bytes, bytes, int]] = {}

# Internal in-memory permissions store:
# Maps staff_id -> set of patient_refs the staff member can access
_PERMISSIONS: Dict[str, Set[str]] = {}

# Constants used to mitigate timing attacks when staff_id is not found
_DUMMY_SALT = b'\x00' * 16
_DUMMY_ITERS = 100_000
_PERM_DUMMY_KEY = b'perm_dummy_key'

# Patient records store
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


def _normalize_staff_id(staff_id: str) -> str:
    return staff_id.strip()


def _normalize_patient_ref(patient_ref: str) -> str:
    return patient_ref.strip()


def _hash_password(password: str, salt: bytes, iterations: int = 200_000) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
        dklen=32,
    )


def authenticate_staff(staff_id: str, password: str) -> bool:
    """
    Verify staff credentials.

    Args:
        staff_id: The unique staff identifier.
        password: The plaintext password to verify.

    Returns:
        True if credentials are valid; False otherwise.
    """
    if not isinstance(staff_id, str) or not isinstance(password, str):
        return False

    staff_id_norm = _normalize_staff_id(staff_id)
    if not staff_id_norm:
        # Avoid early returns without comparable work
        _ = _hash_password(password, _DUMMY_SALT, _DUMMY_ITERS)
        return False

    record = _CREDENTIALS.__dict__.get(staff_id_norm) if hasattr(_CREDENTIALS, "__dict__") else _CREDENTIALS.get(staff_id_norm)
    if record is None:
        # Do comparable work to mitigate timing differences
        _ = _hash_password(password, _DUMMY_SALT, _DUMMY_ITERS)
        return False

    salt, stored_hash, iterations = record
    computed = _hash_password(password, salt, iterations)
    return hmac.compare_digest(stored_hash, computed)


def register_staff(staff_id: str, password: str, iterations: int = 200_000) -> None:
    """
    Register or update staff credentials in the in-memory store.
    This is provided for integration/testing; production should use a persistent, secure store.

    Args:
        staff_id: The unique staff identifier.
        password: The plaintext password to store (will be salted and hashed).
        iterations: PBKDF2 iterations (default 200,000).
    """
    if not isinstance(staff_id, str) or not isinstance(password, str):
        raise TypeError("staff_id and password must be strings")

    staff_id_norm = _normalize_staff_id(staff_id)
    if not staff_id_norm:
        raise ValueError("staff_id cannot be empty")

    if iterations < 100_000:
        raise ValueError("iterations should be >= 100,000 for security")

    salt = os.urandom(16)
    pwd_hash = _hash_password(password, salt, iterations)
    _CREDENTIALS[staff_id_norm] = (salt, pwd_hash, iterations)


def clear_credentials() -> None:
    """
    Clear all credentials from the in-memory store.
    Useful for tests.
    """
    _CREDENTIALS.clear()


def check_staff_permission(staff_num: str, patient_ref: str) -> bool:
    """
    Determine if the staff member is authorized to access the patient's record.

    Args:
        staff_num: The unique staff identifier.
        patient_ref: The unique patient reference/identifier.

    Returns:
        True if the staff member has permission; False otherwise.
    """
    if not isinstance(staff_num, str) or not isinstance(patient_ref, str):
        return False

    staff_id_norm = _normalize_staff_id(staff_num)
    patient_ref_norm = _normalize_patient_ref(patient_ref)

    if not staff_id_norm or not patient_ref_norm:
        # Do comparable work to avoid trivial timing differences
        hmac.new(_PERM_DUMMY_KEY, patient_ref_norm.encode("utf-8"), hashlib.sha256).digest()
        return False

    allowed = _PERMISSIONS.get(staff_id_norm)
    if allowed is None:
        # Do comparable work to avoid trivial timing differences
        hmac.new(_PERM_DUMMY_KEY, patient_ref_norm.encode("utf-8"), hashlib.sha256).digest()
        return False

    # Use constant-time comparison across entries to reduce leakage via string comparison timing
    for ref in allowed:
        if hmac.compare_digest(ref, patient_ref_norm):
            return True
    return False


def grant_permission(staff_id: str, patient_ref: str) -> None:
    """
    Grant access permission for a staff member to a patient's record.
    """
    if not isinstance(staff_id, str) or not isinstance(patient_ref, str):
        raise TypeError("staff_id and patient_ref must be strings")
    staff_id_norm = _normalize_staff_id(staff_id)
    patient_ref_norm = _normalize_patient_ref(patient_ref)
    if not staff_id_norm or not patient_ref_norm:
        raise ValueError("staff_id and patient_ref cannot be empty")
    _PERMISSIONS.setdefault(staff_id_norm, set()).add(patient_ref_norm)


def revoke_permission(staff_id: str, patient_ref: str) -> None:
    """
    Revoke access permission for a staff member to a patient's record.
    """
    if not isinstance(staff_id, str) or not isinstance(patient_ref, str):
        raise TypeError("staff_id and patient_ref must be strings")
    staff_id_norm = _normalize_staff_id(staff_id)
    patient_ref_norm = _normalize_patient_ref(patient_ref)
    if not staff_id_norm or not patient_ref_norm:
        raise ValueError("staff_id and patient_ref cannot be empty")
    if staff_id_norm in _PERMISSIONS:
        _PERMISSIONS[staff_id_norm].discard(patient_ref_norm)
        if not _PERMISSIONS[staff_id_norm]:
            _PERMISSIONS.pop(staff_id_norm, None)


def clear_permissions() -> None:
    """
    Clear all permissions from the in-memory store.
    Useful for tests.
    """
    _PERMISSIONS.clear()


def get_patient_data(staff_num: str, role: str, patient_ref: str) -> str:
    """
    Retrieve a patient's medical record if the staff member is authorized.

    Args:
        staff_num: The unique staff identifier.
        role: The role of the staff member (e.g., 'doctor', 'nurse'). Currently informational.
        patient_ref: The unique patient reference/identifier.

    Returns:
        The patient's medical_data string if authorized.

    Raises:
        PermissionError: If the staff member is not authorized to access the patient's record.
        KeyError: If the patient record does not exist.
        TypeError: If argument types are invalid.
    """
    if not isinstance(staff_num, str) or not isinstance(role, str) or not isinstance(patient_ref, str):
        raise TypeError("staff_num, role, and patient_ref must be strings")

    staff_id_norm = _normalize_staff_id(staff_num)
    patient_ref_norm = _normalize_patient_ref(patient_ref)

    # Validate patient exists (avoid attribute errors later).
    record = PATIENT_RECORDS.get(patient_ref_norm)
    if record is None:
        # Do comparable work to avoid trivial timing differences
        hmac.new(_PERM_DUMMY_KEY, patient_ref_norm.encode("utf-8"), hashlib.sha256).digest()
        raise KeyError(f"Patient not found: {patient_ref_norm}")

    # First, check explicit permissions store
    authorized = check_staff_permission(staff_id_norm, patient_ref_norm)

    # If not authorized via permissions, fall back to assigned_staff list in the patient record
    if not authorized:
        assigned = record.get("assigned_staff", [])
        for sid in assigned:
            if isinstance(sid, str) and hmac.compare_digest(_normalize_staff_id(sid), staff_id_norm):
                authorized = True
                break

    if not authorized:
        # Do comparable work to avoid trivial timing differences
        hmac.new(_PERM_DUMMY_KEY, (staff_id_norm + ":" + patient_ref_norm).encode("utf-8"), hashlib.sha256).digest()
        raise PermissionError("Access denied to patient record")

    return record.get("medical_data")


__all__ = [
    "authenticate_staff",
    "register_staff",
    "clear_credentials",
    "check_staff_permission",
    "grant_permission",
    "revoke_permission",
    "clear_permissions",
    "get_patient_data",
]
