from typing import Mapping, Any, Iterable, Optional, Union
import hmac

EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def _lookup_header(headers: Mapping[str, Any], key: str) -> Any:
    key_lower = key.lower()
    for k, v in headers.items():
        if isinstance(k, str) and k.lower() == key_lower:
            return v
    return None


def _normalize_header_value(value: Any) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    return stripped
    return None


def _extract_token_from_headers(headers: Mapping[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        raw = _lookup_header(headers, key)
        token = _normalize_header_value(raw)
        if token is not None:
            return token
    return None


def validate_csrf_token(headers: Mapping[str, Any]) -> bool:
    """
    Validate the CSRF token from request headers.

    Looks for a token in common CSRF header names and compares it to the expected token
    using a constant-time comparison.

    Returns True if valid, False otherwise.
    """
    if not isinstance(headers, Mapping):
        return False

    candidate_header_names = (
        "x-csrf-token",
        "x-csrftoken",
        "csrf-token",
        "x-xsrf-token",
    )

    token = _extract_token_from_headers(headers, candidate_header_names)
    if token is None or not isinstance(token, str):
        return False

    return hmac.compare_digest(token, EXPECTED_CSRF_TOKEN)


__all__ = ["validate_csrf_token"]
