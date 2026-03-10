import os
import json
import hmac
import base64
import hashlib
from typing import Dict, Any


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


# Path to credentials store; set PERSONNEL_CREDENTIALS_FILE to override.
CREDENTIALS_PATH = os.environ.get(
    "PERSONNEL_CREDENTIALS_FILE",
    os.path.join(os.path.dirname(__file__), "credentials.json"),
)


def _load_credentials() -> Dict[str, Any]:
    """
    Loads personnel credential records from a JSON file.

    Expected JSON structure:
    {
        "personnel_id_1": {
            "salt": "<base64-encoded-bytes>",
            "hash": "<base64-encoded-bytes>",
            "iterations": 600000,
            "role": "physician",  # optional
            "assigned_patients": ["patient_id_1", "patient_id_2"]  # optional
        },
        ...
    }
    """
    try:
        with open(CREDENTIALS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except FileNotFoundError:
        pass
    except Exception:
        # Intentionally swallow to avoid leaking details in authentication paths.
        pass
    return {}


def _verify_password(provided_password: str, salt_b64: str, hash_b64: str, iterations: int) -> bool:
    """
    Verifies a password using PBKDF2-HMAC-SHA256 and constant-time comparison.
    """
    try:
        salt = base64.b64decode(salt_b64, validate=True)
        expected = base64.b64decode(hash_b64, validate=True)
    except Exception:
        return False

    if not salt or not expected or iterations <= 0:
        return False

    try:
        derived = hashlib.pbkdf2_hmac(
            "sha256",
            provided_password.encode("utf-8"),
            salt,
            iterations,
            dklen=len(expected),
        )
    except Exception:
        return False

    return hmac.compare_digest(derived, expected)


# Cache credentials at import to avoid repeated I/O on each call.
_CREDENTIALS = _load_credentials()


def authenticate_personnel(personnel_id: str, password: str) -> bool:
    """
    Authenticate personnel by verifying the provided credentials.

    Args:
        personnel_id: The personnel identifier (string).
        password: The plaintext password to verify.

    Returns:
        True if credentials are valid, False otherwise.
    """
    if not isinstance(personnel_id, str) or not isinstance(password, str):
        return False
    if not personnel_id or not password:
        return False

    record = _CREDENTIALS.get(personnel_id)
    if not isinstance(record, dict):
        # Unknown ID
        return False

    salt_b64 = record.get("salt")
    hash_b64 = record.get("hash")
    iterations = record.get("iterations", 600000)

    try:
        iterations = int(iterations)
    except Exception:
        iterations = 600000

    if not salt_b64 or not hash_b64:
        return False

    return _verify_password(password, salt_b64, hash_b64, iterations)


def verify_access(personnel_id: str, personnel_role: str, patient_identifier: str) -> bool:
    """
    Verify whether a personnel member has access to a patient's records.

    Access rules (conservative defaults):
    - The personnel_id must exist in the credentials store.
    - Elevated roles ("admin", "privacy_officer") have access to all patients.
    - Other clinical roles must be explicitly assigned to the patient via
      the "assigned_patients" list in the personnel record.
    - If a role is stored in the credentials record, it takes precedence over
      the provided personnel_role argument.

    Args:
        personnel_id: Unique personnel identifier.
        personnel_role: Role of the personnel (e.g., "physician", "nurse", "admin").
        patient_identifier: Unique patient identifier.

    Returns:
        True if access is permitted; False otherwise.
    """
    # Basic type and value validation
    if not isinstance(personnel_id, str) or not isinstance(personnel_role, str) or not isinstance(patient_identifier, str):
        return False
    if not personnel_id or not personnel_role or not patient_identifier:
        return False

    record = _CREDENTIALS.get(personnel_id)
    if not isinstance(record, dict):
        # Unknown personnel
        return False

    # If role is stored with the record, prefer it over the provided role to prevent spoofing.
    stored_role = record.get("role")
    role = (stored_role if isinstance(stored_role, str) and stored_role.strip() else personnel_role).strip().lower()

    # Elevated roles with blanket access (must still be a known personnel_id)
    elevated_roles = {"admin", "privacy_officer"}
    if role in elevated_roles:
        return True

    # Default: require explicit assignment
    assigned = record.get("assigned_patients")
    if isinstance(assigned, list):
        # Normalize identifiers to strings for comparison
        assigned_set = {str(pid) for pid in assigned}
        return str(patient_identifier) in assigned_set

    # No assignment info: deny by default
    return False


def get_patient_record(personnel_id: str, personnel_role: str, patient_identifier: str) -> str:
    """
    Retrieve a patient's medical data if the requester is authorized.

    Args:
        personnel_id: Unique identifier for the staff member.
        personnel_role: Role of the staff member (e.g., "physician", "nurse", "admin").
        patient_identifier: Unique patient identifier.

    Returns:
        The medical data string for the specified patient.

    Raises:
        PermissionError: If the requester is not authorized to access the record.
        KeyError: If the patient record does not exist.
    """
    # Treat invalid inputs as unauthorized.
    if not isinstance(personnel_id, str) or not isinstance(personnel_role, str) or not isinstance(patient_identifier, str):
        raise PermissionError("Access denied")
    if not personnel_id or not personnel_role or not patient_identifier:
        raise PermissionError("Access denied")

    record = PATIENT_RECORDS.get(patient_identifier)
    if record is None:
        raise KeyError(f"Patient record not found: {patient_identifier}")

    # Primary authorization check using existing access control.
    if verify_access(personnel_id, personnel_role, patient_identifier):
        return record.get("medical_data")

    # Secondary check against patient-assigned staff list.
    assigned_staff = record.get("assigned_staff")
    if isinstance(assigned_staff, list) and str(personnel_id) in {str(s) for s in assigned_staff}:
        return record.get("medical_data")

    # Unauthorized
    raise PermissionError("Access denied")
