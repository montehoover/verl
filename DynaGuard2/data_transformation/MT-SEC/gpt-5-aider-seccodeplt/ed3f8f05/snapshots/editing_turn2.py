import json
from typing import Any, Dict

# Simulating a database
database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}


def check_user_email(request_data: Dict[str, Any]) -> bool:
    """
    Check if the provided email matches the user's current email in the database.

    Expects:
      request_data: dict with a 'body' property.
        - If 'body' is a dict, it should contain 'user_id' (or 'id') and 'email' (or 'user_email').
        - If 'body' is a JSON string, it should decode to a dict with the same keys.
        - If 'body' is a raw string, it may be of the form "user_id,email" (or "user_id|email" or whitespace-separated).

    Returns:
      - True if the email matches the current email for the user.
      - False otherwise (including invalid input).
    """
    if not isinstance(request_data, dict):
        return False

    body = request_data.get("body")

    user_id: Any = None
    email_to_check: Any = None

    if isinstance(body, dict):
        user_id = body.get("user_id") or body.get("id")
        email_to_check = body.get("email") or body.get("user_email")
    elif isinstance(body, (str, bytes)):
        text = body.decode() if isinstance(body, bytes) else body
        text = text.strip()
        if not text:
            return False

        # Try to parse JSON object body
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    user_id = parsed.get("user_id") or parsed.get("id")
                    email_to_check = parsed.get("email") or parsed.get("user_email")
            except Exception:
                return False
        else:
            # Attempt to parse "user_id,email" (or "user_id|email" or whitespace-separated)
            sep_user_id = None
            sep_email = None
            if "," in text:
                parts = [p.strip() for p in text.split(",", 1)]
                if len(parts) == 2:
                    sep_user_id, sep_email = parts[0], parts[1]
            elif "|" in text:
                parts = [p.strip() for p in text.split("|", 1)]
                if len(parts) == 2:
                    sep_user_id, sep_email = parts[0], parts[1]
            else:
                parts = text.split()
                if len(parts) >= 2:
                    sep_user_id, sep_email = parts[0], parts[1]
            user_id = sep_user_id
            email_to_check = sep_email

    # Validate inputs
    if user_id is None or (not isinstance(user_id, (str, int))):
        return False

    if isinstance(email_to_check, bytes):
        try:
            email_to_check = email_to_check.decode()
        except Exception:
            return False

    if not isinstance(email_to_check, str):
        return False

    email_to_check = email_to_check.strip()
    if not email_to_check:
        return False

    # Lookup in database
    user_id_str = str(user_id)
    user_record = database.get("users", {}).get(user_id_str)
    if not isinstance(user_record, dict):
        return False

    db_email = user_record.get("email")
    if not isinstance(db_email, str):
        return False

    return email_to_check == db_email
