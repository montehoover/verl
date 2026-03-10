from typing import Mapping, Any

EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def verify_csrf_token(headers: Mapping[str, Any]) -> bool:
    """
    Verify that the CSRF token provided in the headers matches the expected token.

    Args:
        headers: A mapping of request header names to values.

    Returns:
        True if a recognized CSRF header is present and equals the expected token; False otherwise.
    """
    if not headers:
        return False

    # Normalize header keys to lowercase for case-insensitive lookup
    normalized = {str(k).lower(): v for k, v in headers.items()}

    # Common CSRF header names across frameworks and clients
    candidate_header_keys = [
        "x-csrf-token",
        "x-csrftoken",
        "x-xsrf-token",
        "x-xsrftoken",
        "csrf-token",
        "csrf_token",
    ]

    for key in candidate_header_keys:
        if key in normalized:
            value = normalized[key]
            # Support list-like values some frameworks use
            if isinstance(value, (list, tuple)):
                if not value:
                    return False
                value = value[0]
            token = str(value)
            return token == EXPECTED_CSRF_TOKEN

    # No CSRF header found
    return False
