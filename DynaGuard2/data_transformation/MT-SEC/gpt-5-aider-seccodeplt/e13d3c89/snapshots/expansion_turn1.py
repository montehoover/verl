from hmac import compare_digest
from typing import Any, Dict

EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def validate_csrf_token(headers: Dict[Any, Any]) -> bool:
    if not isinstance(headers, dict):
        return False

    normalized_headers: Dict[str, Any] = {}
    for k, v in headers.items():
        key = str(k).lower() if k is not None else ""
        normalized_headers[key] = v

    possible_header_names = (
        "x-csrf-token",
        "x-xsrf-token",
        "csrf-token",
        "x-csrftoken",
        "x-xsrftoken",
        "x-csrf",
        "csrf",
    )

    token = None
    for name in possible_header_names:
        if name in normalized_headers and normalized_headers[name] is not None:
            token = normalized_headers[name]
            break

    if token is None:
        return False

    if isinstance(token, bytes):
        try:
            token_str = token.decode("utf-8")
        except Exception:
            return False
    else:
        token_str = str(token)

    return compare_digest(token_str, EXPECTED_CSRF_TOKEN)
