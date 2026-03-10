import os
import json
import hmac
import base64
import hashlib
from typing import Dict, Any


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
            "iterations": 600000
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
