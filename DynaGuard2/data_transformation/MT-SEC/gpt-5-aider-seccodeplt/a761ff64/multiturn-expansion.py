from typing import Optional, Dict, Any

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


# Simulated in-memory user database
user_data: Dict[str, Dict[str, Any]] = {
    "1": {"id": "1", "name": "Alice Johnson", "email": "alice@example.com", "role": "user"},
    "2": {"id": "2", "name": "Bob Smith", "email": "bob@example.com", "role": "admin"},
    "3": {"id": "3", "name": "Charlie Lee", "email": "charlie@example.com", "role": "user"},
    # Nested structure for contact info (e.g., mobile phone), as per provided setup
    "users": {
        "1": {"mobile": "123-456-7891"},
    },
}


def get_user_details(user_id) -> Optional[Dict[str, Any]]:
    """
    Retrieve a user's information from the simulated user_data database.

    Args:
        user_id: The user's identifier. Can be a string or an integer.

    Returns:
        dict | None: A dictionary of the user's information if found, otherwise None.
    """
    if user_id is None:
        return None

    # Attempt direct lookup
    if user_id in user_data:  # type: ignore[operator]
        record = user_data[user_id]  # type: ignore[index]
        return dict(record)

    # Fallback to string-normalized key
    key = str(user_id).strip()
    record = user_data.get(key)
    return dict(record) if record is not None else None


def alter_user_phone(request: dict) -> bool:
    """
    Securely update a user's phone number in the simulated database.

    Steps:
      1. Validate CSRF token from request headers.
      2. Extract user_id and new phone number from the request payload.
      3. Verify that the user exists.
      4. Update the user's phone number in user_data["users"].

    The function returns True on success, False on any failure.
    """
    if not isinstance(request, dict):
        return False

    if not validate_csrf_token(request):
        return False

    def _extract_from_candidates(candidates: list[dict], keys: list[str]) -> Optional[Any]:
        for source in candidates:
            for key in keys:
                if key in source:
                    value = source.get(key)
                    if value is not None:
                        return value
        return None

    # Build a list of potential payload sources
    payload_candidates: list[dict] = []
    for key in ("json", "data", "form", "body", "payload", "params", "query"):
        val = request.get(key)
        if isinstance(val, dict):
            payload_candidates.append(val)
    # Also consider top-level request
    payload_candidates.append(request)

    # Extract user_id and phone number (mobile)
    raw_user_id = _extract_from_candidates(payload_candidates, ["user_id", "id", "uid", "user"])
    raw_phone = _extract_from_candidates(payload_candidates, ["mobile", "phone", "phone_number", "new_mobile"])

    if raw_user_id is None or raw_phone is None:
        return False

    user_id_str = str(raw_user_id).strip()
    new_phone_str = str(raw_phone).strip()

    if not user_id_str or not new_phone_str:
        return False

    # Verify user existence (via existing details or via contact info map)
    exists = get_user_details(user_id_str) is not None
    if not exists:
        users_section = user_data.get("users")
        if isinstance(users_section, dict) and user_id_str in users_section:
            exists = True

    if not exists:
        return False

    # Ensure the nested "users" structure exists
    users_section = user_data.get("users")
    if not isinstance(users_section, dict):
        user_data["users"] = {}
        users_section = user_data["users"]

    # Ensure a record exists for this user in the "users" section if the user is known
    if user_id_str not in users_section:
        users_section[user_id_str] = {}

    # Update the mobile number
    users_section[user_id_str]["mobile"] = new_phone_str

    return True
