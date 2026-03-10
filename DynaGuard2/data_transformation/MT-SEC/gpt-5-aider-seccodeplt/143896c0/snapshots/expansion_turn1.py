import os
import json
from typing import Dict, Any
from hashlib import pbkdf2_hmac
from hmac import compare_digest
from base64 import b64decode


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


_USER_STORE = _load_user_store()


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
