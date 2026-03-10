import hmac
from typing import Any, Dict

def validate_csrf_protection(headers: Dict[str, Any]) -> bool:
    if not isinstance(headers, dict):
        return False

    lowered = {}
    for k, v in headers.items():
        if isinstance(k, str):
            lowered[k.lower()] = v

    header_candidates = [
        "x-csrf-token",
        "x-csrftoken",
        "csrf-token",
        "x-xsrf-token",
        "x-xsrftoken",
    ]

    token = None
    for key in header_candidates:
        if key in lowered:
            token = lowered[key]
            break

    if token is None:
        return False

    if isinstance(token, (list, tuple)):
        token = token[0] if token else ""

    if not isinstance(token, str):
        token = str(token)

    token = token.strip()
    if not token:
        return False

    expected = "secure_csrf_token_value"
    return hmac.compare_digest(token, expected)
