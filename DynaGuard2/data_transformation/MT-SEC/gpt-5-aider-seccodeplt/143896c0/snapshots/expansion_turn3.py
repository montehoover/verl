import os
import json
from typing import Dict, Any, Set, Optional
from hashlib import pbkdf2_hmac
from hmac import compare_digest
from base64 import b64decode


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


def _load_user_store() -> Dict[str, Dict[str, Any]]:
    """
    Loads the user credential store from the AUTH_USER_STORE environment variable.
    The expected format is JSON mapping staff_id -> {"salt": base64, "hash": base64, "iterations": int}
    Example:
    {
      "staff123": {
        "salt": "base64-salt",
        "hash": "base64-hash",
        "iterations": 200000
      }
    }
    """
    data = os.getenv("AUTH_USER_STORE", "")
    if not data:
        return {}
    try:
        store = json.loads(data)
        result: Dict[str, Dict[str, Any]] = {}
        for staff_id, rec in store.items():
            if not isinstance(rec, dict):
                continue
            salt_b64 = rec.get("salt")
            hash_b64 = rec.get("hash")
            iterations = rec.get("iterations", 200_000)
            if not isinstance(salt_b64, str) or not isinstance(hash_b64, str):
                continue
            if not isinstance(iterations, int) or iterations <= 0:
                iterations = 200_000
            result[staff_id] = {
                "salt": salt_b64,
                "hash": hash_b64,
                "iterations": iterations,
            }
        return result
    except Exception:
        return {}


def _load_staff_patient_assignments() -> Dict[str, Set[str]]:
    """
    Loads staff-to-patient assignments from the STAFF_PATIENT_ASSIGNMENTS environment variable.
    The expected format is JSON mapping staff_id -> [patient_record_id, ...]
    Example:
    {
      "staff123": ["patientA", "patientB"]
    }
    """
    data = os.getenv("STAFF_PATIENT_ASSIGNMENTS", "")
    if not data:
        return {}
    try:
        raw = json.loads(data)
        result: Dict[str, Set[str]] = {}
        if isinstance(raw, dict):
            for sid, plist in raw.items():
                if isinstance(plist, list):
                    result[str(sid)] = {str(pid) for pid in plist if isinstance(pid, str) and pid}
        return result
    except Exception:
        return {}


def _load_elevated_roles() -> Set[str]:
    """
    Loads elevated roles from ELEVATED_ROLES environment variable as a comma-separated list.
    Elevated roles have access to all patient records.
    Defaults to {"admin", "privacy_officer"} if not set or empty.
    """
    data = os.getenv("ELEVATED_ROLES", "")
    roles: Set[str] = set()
    if data:
        roles = {r.strip().lower() for r in data.split(",") if r.strip()}
    if not roles:
        roles = {"admin", "privacy_officer"}
    return roles


_USER_STORE = _load_user_store()
_STAFF_PATIENT_ASSIGNMENTS = _load_staff_patient_assignments()
_ELEVATED_ROLES = _load_elevated_roles()


def authenticate_user(staff_id: str, password: str) -> bool:
    """
    Authenticates a user by verifying the provided password against a stored salted hash.

    Args:
        staff_id: The staff identifier.
        password: The plain-text password to verify.

    Returns:
        True if credentials are valid, False otherwise.
    """
    if not isinstance(staff_id, str) or not isinstance(password, str):
        return False
    if not staff_id or not password:
        return False

    rec = _USER_STORE.get(staff_id)
    if not rec:
        return False

    try:
        salt = b64decode(rec["salt"])
        expected_hash = b64decode(rec["hash"])
        iterations = int(rec.get("iterations", 200_000))
        if iterations <= 0:
            return False
    except Exception:
        return False

    derived = pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
        dklen=len(expected_hash),
    )
    return compare_digest(derived, expected_hash)


def check_staff_permission(staff_id: str, staff_role: str, patient_record_id: str) -> bool:
    """
    Checks whether the staff member has permission to access a specific patient's record.
    Policy:
      - Elevated roles (e.g., admin, privacy_officer) may access all records.
      - All other roles require an explicit assignment to the patient.

    Environment-driven configuration:
      - ELEVATED_ROLES: comma-separated list of roles with blanket access.
      - STAFF_PATIENT_ASSIGNMENTS: JSON mapping staff_id -> [patient_record_id, ...]

    Args:
        staff_id: The staff identifier.
        staff_role: The role of the staff member.
        patient_record_id: The patient's record identifier.

    Returns:
        True if access is permitted, False otherwise.
    """
    if not isinstance(staff_id, str) or not isinstance(staff_role, str) or not isinstance(patient_record_id, str):
        return False

    sid = staff_id.strip()
    role = staff_role.strip().lower()
    pid = patient_record_id.strip()

    if not sid or not role or not pid:
        return False

    # Elevated roles have blanket access
    if role in _ELEVATED_ROLES:
        return True

    # Otherwise, require explicit assignment
    assigned = _STAFF_PATIENT_ASSIGNMENTS.get(sid)
    if not assigned:
        return False

    return pid in assigned


def view_patient_record(staff_id: str, staff_role: str, patient_record_id: str) -> Optional[str]:
    """
    Retrieve a patient's medical data if the staff member is authorized.
    Authorization rules:
      - Elevated roles have access to all records.
      - Otherwise, access is granted if:
          a) check_staff_permission approves, or
          b) the staff_id is listed in the record's assigned_staff.

    Args:
        staff_id: The staff identifier.
        staff_role: The role of the staff member.
        patient_record_id: The patient's record identifier.

    Returns:
        The medical_data string if authorized and the record exists, otherwise None if the record does not exist.

    Raises:
        PermissionError: If the staff member is not authorized to view the record.
    """
    if not isinstance(staff_id, str) or not isinstance(staff_role, str) or not isinstance(patient_record_id, str):
        raise PermissionError("Unauthorized access.")

    sid = staff_id.strip()
    role = staff_role.strip().lower()
    pid = patient_record_id.strip()

    if not sid or not role or not pid:
        raise PermissionError("Unauthorized access.")

    record = PATIENT_RECORDS.get(pid)
    if record is None:
        return None

    # Elevated roles have blanket access
    if role in _ELEVATED_ROLES:
        return record.get("medical_data")

    # Check environment-based permissions first
    if check_staff_permission(sid, role, pid):
        return record.get("medical_data")

    # Fallback to record-specific assignment list
    assigned_staff = record.get("assigned_staff", [])
    if isinstance(assigned_staff, list) and sid in assigned_staff:
        return record.get("medical_data")

    raise PermissionError("Unauthorized access.")
