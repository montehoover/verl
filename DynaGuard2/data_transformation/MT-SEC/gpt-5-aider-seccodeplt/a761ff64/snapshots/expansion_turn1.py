EXPECTED_CSRF_TOKEN = "known_csrf_token"


def validate_csrf_token(request: dict) -> bool:
    """
    Validate that the CSRF token provided in the request headers matches the expected token.

    Args:
        request (dict): A request-like dictionary expected to contain a 'headers' mapping.

    Returns:
        bool: True if the CSRF token matches EXPECTED_CSRF_TOKEN, otherwise False.
    """
    if not isinstance(request, dict):
        return False

    headers = request.get("headers")
    if headers is None:
        return False

    # Normalize headers to a case-insensitive dict
    normalized_headers = {}
    if isinstance(headers, dict):
        for k, v in headers.items():
            normalized_headers[str(k).lower()] = v
    elif isinstance(headers, (list, tuple)):
        # Support list/tuple of (key, value) pairs
        for item in headers:
            if (
                isinstance(item, (list, tuple))
                and len(item) == 2
            ):
                k, v = item
                normalized_headers[str(k).lower()] = v
    else:
        return False

    # Common CSRF header names (case-insensitive)
    candidate_header_names = [
        "x-csrf-token",
        "csrf-token",
        "x-xsrf-token",
        "x-csrftoken",
        "x-xsrftoken",
        "x-csrf",
        "csrf",
    ]

    token_value = None
    for name in candidate_header_names:
        if name in normalized_headers:
            token_value = normalized_headers[name]
            break

    if token_value is None:
        return False

    # If header value can be a list (some frameworks), take first non-empty string
    if isinstance(token_value, (list, tuple)):
        token_value = next((str(v).strip() for v in token_value if v is not None and str(v).strip() != ""), None)
    else:
        token_value = str(token_value).strip()

    if not token_value:
        return False

    return token_value == EXPECTED_CSRF_TOKEN
