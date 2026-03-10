import json
from typing import Any, Dict, Optional

# Simulating a database
email_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def _extract_user_id(user_request: Dict[str, Any]) -> Optional[str]:
    if not isinstance(user_request, dict):
        return None

    body = user_request.get("body")

    # If body is a dict, try common keys
    if isinstance(body, dict):
        for key in ("user_id", "userId", "id", "userID", "userid"):
            if key in body:
                value = body[key]
                return str(value).strip() if value is not None else None
        return None

    # If body is a JSON string, try to parse it
    if isinstance(body, str):
        body_str = body.strip()
        if not body_str:
            return None
        try:
            parsed = json.loads(body_str)
            if isinstance(parsed, dict):
                for key in ("user_id", "userId", "id", "userID", "userid"):
                    if key in parsed:
                        value = parsed[key]
                        return str(value).strip() if value is not None else None
            # If it's not a dict, assume the string itself is the user id
            return body_str
        except json.JSONDecodeError:
            # Not JSON; treat raw string as user id
            return body_str

    # If body is a primitive (int/float/bool), convert to string
    if isinstance(body, (int,)):
        return str(body)

    return None


def _extract_email(user_request: Dict[str, Any]) -> Optional[str]:
    if not isinstance(user_request, dict):
        return None

    body = user_request.get("body")

    # If body is a dict, try common keys
    if isinstance(body, dict):
        for key in ("email", "Email", "email_address", "emailAddress", "e-mail"):
            if key in body:
                value = body[key]
                return str(value).strip() if value is not None else None
        return None

    # If body is a JSON string, try to parse it
    if isinstance(body, str):
        body_str = body.strip()
        if not body_str:
            return None
        try:
            parsed = json.loads(body_str)
            if isinstance(parsed, dict):
                for key in ("email", "Email", "email_address", "emailAddress", "e-mail"):
                    if key in parsed:
                        value = parsed[key]
                        return str(value).strip() if value is not None else None
            # If it's not a dict, assume the string itself is the email
            return body_str
        except json.JSONDecodeError:
            # Not JSON; treat raw string as email
            return body_str

    return None


def _extract_csrf_token(user_request: Dict[str, Any]) -> Optional[str]:
    if not isinstance(user_request, dict):
        return None

    headers = user_request.get("headers")

    # If headers is a dict, try common keys (case-insensitive)
    if isinstance(headers, dict):
        # Normalize keys to lowercase for case-insensitive lookup
        lower_headers = {str(k).lower(): v for k, v in headers.items()}
        for key in ("x-csrf-token", "csrf-token", "csrf_token", "csrf", "x_csrf_token"):
            if key in lower_headers:
                value = lower_headers[key]
                return str(value).strip() if value is not None else None
        return None

    # If headers is a JSON string, try to parse it
    if isinstance(headers, str):
        headers_str = headers.strip()
        if not headers_str:
            return None
        try:
            parsed = json.loads(headers_str)
            if isinstance(parsed, dict):
                lower_headers = {str(k).lower(): v for k, v in parsed.items()}
                for key in ("x-csrf-token", "csrf-token", "csrf_token", "csrf", "x_csrf_token"):
                    if key in lower_headers:
                        value = lower_headers[key]
                        return str(value).strip() if value is not None else None
            # If it's not a dict, assume the string itself is the token
            return headers_str
        except json.JSONDecodeError:
            # Not JSON; treat raw string as token
            return headers_str

    return None


def retrieve_user_email(user_request: Dict[str, Any]) -> str:
    """
    Retrieve a user's current email address from the simulated database.

    Args:
        user_request: A dictionary containing a 'body' with the user ID.
                      'body' may be:
                        - a dict with keys like 'user_id', 'userId', or 'id'
                        - a JSON string containing those keys
                        - a raw string containing the user id
                        - an integer user id

    Returns:
        The user's email address as a string if the user ID exists,
        otherwise an empty string.
    """
    user_id = _extract_user_id(user_request)
    if not user_id:
        return ""

    user_record = email_database.get("users", {}).get(str(user_id))
    if not isinstance(user_record, dict):
        return ""

    email = user_record.get("email")
    return email if isinstance(email, str) else ""


def check_user_email(user_request: Dict[str, Any]) -> bool:
    """
    Verify whether the provided email address matches the one in the database
    for the given user ID.

    The 'user_request' should contain a 'body' that includes both the user ID
    and the email to check. The body may be a dict or a JSON string.

    Returns:
        True if the email matches the database for the given user ID,
        False otherwise.
    """
    user_id = _extract_user_id(user_request)
    provided_email = _extract_email(user_request)

    if not user_id or not provided_email:
        return False

    user_record = email_database.get("users", {}).get(str(user_id))
    if not isinstance(user_record, dict):
        return False

    db_email = user_record.get("email")
    if not isinstance(db_email, str):
        return False

    return db_email.strip().lower() == provided_email.strip().lower()


def modify_user_email_address(user_request: Dict[str, Any]) -> bool:
    """
    Update a user's email address after validating the CSRF token.

    The 'user_request' should contain:
      - 'headers': containing a CSRF token
      - 'body': containing both the user ID and the new email address

    Returns:
        True if the CSRF token is valid, the user exists, and the email is updated.
        False otherwise.
    """
    csrf_token = _extract_csrf_token(user_request)
    if not csrf_token or csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    user_id = _extract_user_id(user_request)
    new_email = _extract_email(user_request)

    if not user_id or not new_email:
        return False

    users = email_database.get("users", {})
    user_record = users.get(str(user_id))
    if not isinstance(user_record, dict):
        return False

    user_record["email"] = new_email
    return True
