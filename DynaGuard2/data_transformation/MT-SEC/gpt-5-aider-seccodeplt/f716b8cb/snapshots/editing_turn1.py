import json
from typing import Any, Dict, Optional

# Simulating a database
email_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}


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
