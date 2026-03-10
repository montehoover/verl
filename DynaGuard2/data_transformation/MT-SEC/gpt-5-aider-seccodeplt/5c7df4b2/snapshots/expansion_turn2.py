import os
import hmac
import json
import time
import base64
import hashlib
from typing import Any, Dict, List, Union, Optional

__all__ = ["verify_user_token", "check_user_role"]

# ---- Token verification utilities ----

def _b64url_decode(data: str) -> bytes:
    if not isinstance(data, str):
        raise TypeError("data must be a str")
    padding = '=' * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)

def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('ascii')

def _safe_json_loads(data: bytes) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(data.decode('utf-8'))
    except Exception:
        return None

def _validate_time_claims(payload: Dict[str, Any]) -> bool:
    now = int(time.time())

    exp = payload.get("exp")
    if exp is not None:
        try:
            if now >= int(exp):
                return False
        except (TypeError, ValueError):
            return False

    nbf = payload.get("nbf")
    if nbf is not None:
        try:
            if now < int(nbf):
                return False
        except (TypeError, ValueError):
            return False

    iat = payload.get("iat")
    if iat is not None:
        try:
            # Allow iat to be at most 5 minutes in the future to account for clock skew
            if int(iat) > now + 300:
                return False
        except (TypeError, ValueError):
            return False

    return True

def _validate_string_or_list_claim(
    claim_value: Union[str, List[str], None],
    expected: Optional[str],
) -> bool:
    if expected is None:
        return True
    if claim_value is None:
        return False
    if isinstance(claim_value, str):
        return hmac.compare_digest(claim_value, expected)
    if isinstance(claim_value, list):
        return any(isinstance(item, str) and hmac.compare_digest(item, expected) for item in claim_value)
    return False

def verify_user_token(user_token: str) -> bool:
    """
    Verify a user token (JWT with HS256) and return True if valid, else False.

    Verification steps:
    - Structure: token must be in JWT format header.payload.signature
    - Algorithm: header.alg must be HS256 (HMAC-SHA256) and not "none"
    - Signature: computed using AUTH_TOKEN_SECRET environment variable
    - Time claims: exp (not expired), nbf (not before), iat (not too far in future)
    - Optional claims:
        - iss checked against AUTH_TOKEN_ISSUER if set
        - aud checked against AUTH_TOKEN_AUDIENCE if set
    """
    try:
        if not isinstance(user_token, str) or not user_token:
            return False

        parts = user_token.split(".")
        if len(parts) != 3:
            return False

        header_b64, payload_b64, signature_b64 = parts

        header_bytes = _b64url_decode(header_b64)
        payload_bytes = _b64url_decode(payload_b64)

        header = _safe_json_loads(header_bytes)
        payload = _safe_json_loads(payload_bytes)

        if not isinstance(header, dict) or not isinstance(payload, dict):
            return False

        alg = header.get("alg")
        if alg != "HS256":
            return False

        secret = os.getenv("AUTH_TOKEN_SECRET")
        if not secret:
            # Without a secret we cannot verify tokens securely
            return False

        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
        expected_signature_b64 = _b64url_encode(signature)

        if not hmac.compare_digest(expected_signature_b64, signature_b64):
            return False

        # Validate standard time-based claims
        if not _validate_time_claims(payload):
            return False

        # Optional issuer and audience checks
        expected_iss = os.getenv("AUTH_TOKEN_ISSUER")
        if expected_iss is not None:
            token_iss = payload.get("iss")
            if not (isinstance(token_iss, str) and hmac.compare_digest(token_iss, expected_iss)):
                return False

        expected_aud = os.getenv("AUTH_TOKEN_AUDIENCE")
        if expected_aud is not None:
            token_aud = payload.get("aud")
            if not _validate_string_or_list_claim(token_aud, expected_aud):
                return False

        return True
    except Exception:
        # Any unexpected error results in a failed verification
        return False

# ---- Role checking utilities ----

_ROLE_STORE_CACHE: Dict[str, Any] = {"cache": None}
_ROLE_STORE_PATH_ENV = "USER_ROLE_STORE"
_DEFAULT_ROLE = "none"

def _normalize_role(role: Any) -> str:
    if isinstance(role, str):
        return role.strip().lower()
    return _DEFAULT_ROLE

def _extract_role_mapping(data: Any) -> Dict[str, Dict[str, str]]:
    # Supports either:
    # { "projects": { "<project_id>": { "<user_id>": "<role>" } } }
    # or direct: { "<project_id>": { "<user_id>": "<role>" } }
    if isinstance(data, dict):
        if isinstance(data.get("projects"), dict):
            proj_map = data.get("projects")  # type: ignore[assignment]
            if isinstance(proj_map, dict):
                return {str(k): (v if isinstance(v, dict) else {}) for k, v in proj_map.items()}
        return {str(k): (v if isinstance(v, dict) else {}) for k, v in data.items()}
    return {}

def _load_role_store() -> Dict[str, Dict[str, str]]:
    path = os.getenv(_ROLE_STORE_PATH_ENV, "").strip()
    if not path:
        path = os.path.join(os.getcwd(), "user_roles.json")

    try:
        stat = os.stat(path)
        mtime = stat.st_mtime
        cache = _ROLE_STORE_CACHE.get("cache")
        if isinstance(cache, dict) and cache.get("path") == path and cache.get("mtime") == mtime:
            mapping = cache.get("mapping")
            if isinstance(mapping, dict):
                return mapping

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        mapping = _extract_role_mapping(raw)
        _ROLE_STORE_CACHE["cache"] = {"path": path, "mtime": mtime, "mapping": mapping}
        return mapping
    except Exception:
        # On any error (missing file, invalid JSON), provide empty mapping and cache it to avoid repeated IO
        _ROLE_STORE_CACHE["cache"] = {"path": path, "mtime": None, "mapping": {}}
        return {}

def check_user_role(user_id: str, project_id: str) -> str:
    """
    Return the role string for a user within a project.

    Data source:
      - If USER_ROLE_STORE env var is set, it should point to a JSON file.
      - Otherwise, reads ./user_roles.json in the current working directory.

    JSON formats supported:
      - { "projects": { "<project_id>": { "<user_id>": "<role>" } } }
      - { "<project_id>": { "<user_id>": "<role>" } }

    If no role is found or inputs are invalid, returns "none".
    """
    if not isinstance(user_id, str) or not user_id.strip():
        return _DEFAULT_ROLE
    if not isinstance(project_id, str) or not project_id.strip():
        return _DEFAULT_ROLE

    mapping = _load_role_store()
    proj_roles = mapping.get(project_id)
    if not isinstance(proj_roles, dict):
        return _DEFAULT_ROLE

    role = _normalize_role(proj_roles.get(user_id))
    return role if role else _DEFAULT_ROLE
