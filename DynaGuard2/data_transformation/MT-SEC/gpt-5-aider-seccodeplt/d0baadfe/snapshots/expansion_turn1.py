import hmac
from typing import Any, Dict, Iterable

EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def validate_csrf_token(headers: Dict[str, Any]) -> bool:
    if not isinstance(headers, dict):
        return False

    lower_headers: Dict[str, Any] = {str(k).lower(): v for k, v in headers.items()}

    candidate_keys: Iterable[str] = (
        "x-csrf-token",
        "x-csrftoken",
        "csrf-token",
        "x-xsrf-token",
        "xsrf-token",
        "x-xsrftoken",
        "x-csrf",
        "csrf",
        "csrf_token",
        "x_csrf_token",
    )

    token = None
    for key in candidate_keys:
        v = lower_headers.get(key)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            if not v:
                continue
            v = v[0]
        token = str(v).strip()
        if token != "":
            break

    if not token:
        return False

    return hmac.compare_digest(token, EXPECTED_CSRF_TOKEN)
