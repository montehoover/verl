import json
from typing import Any, Dict

# Simulating a database
database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}


def get_user_email(request_data: Dict[str, Any]) -> str:
    """
    Retrieve a user's email from the database.

    Expects:
      request_data: dict with a 'body' property.
        - If 'body' is a dict, it should contain 'user_id' (preferred) or 'id'.
        - If 'body' is a JSON string, it should decode to a dict with the same keys.
        - If 'body' is a raw string, it is treated as the user ID.

    Returns:
      - The email string if the user exists.
      - An empty string if the user does not exist or input is invalid.
    """
    if not isinstance(request_data, dict):
        return ""

    body = request_data.get("body")

    user_id = None
    if isinstance(body, dict):
        user_id = body.get("user_id") or body.get("id")
    elif isinstance(body, (str, bytes)):
        text = body.decode() if isinstance(body, bytes) else body
        text = text.strip()
        # Try to parse JSON object body
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    user_id = parsed.get("user_id") or parsed.get("id")
            except Exception:
                # If JSON parsing fails, fall back to treating as raw ID
                user_id = text
        else:
            # Treat as raw ID string
            user_id = text

    if user_id is None:
        return ""

    user_id_str = str(user_id)
    user_record = database.get("users", {}).get(user_id_str)
    if not isinstance(user_record, dict):
        return ""

    email = user_record.get("email")
    return email if isinstance(email, str) else ""
