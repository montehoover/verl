import json
from typing import Any, Dict

# Simulating a database
database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def change_user_email(request_data: Dict[str, Any]) -> bool:
    """
    Update a user's email after validating CSRF token.

    Expects:
      request_data: dict with 'headers' and 'body' properties.
        - headers:
            * Should include a CSRF token in one of:
              'X-CSRF-Token', 'X-XSRF-Token', 'CSRF-Token', 'csrf_token', 'csrf'
              (lookup is case-insensitive)
            * Headers may also be provided as a JSON string.
        - body:
            * If 'body' is a dict, it should contain:
                - 'user_id' (or 'id')
                - 'new_email' (or 'email')
            * If 'body' is a JSON string, it should decode to a dict with the same keys.
            * If 'body' is a raw string, it may be of the form:
                "user_id,new_email" (or "user_id|new_email" or whitespace-separated)

    Returns:
      - True if the CSRF token is valid, the user exists, and the email is updated.
      - False otherwise (including invalid input).
    """
    if not isinstance(request_data, dict):
        return False

    # Extract and validate CSRF token from headers
    headers = request_data.get("headers")
    headers_dict: Dict[str, Any] = {}

    if isinstance(headers, dict):
        headers_dict = headers
    elif isinstance(headers, (str, bytes)):
        text = headers.decode() if isinstance(headers, bytes) else headers
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    headers_dict = parsed
            except Exception:
                return False
        else:
            return False
    else:
        return False

    def _get_header_case_insensitive(h: Dict[str, Any], key: str) -> Any:
        key_lower = key.lower()
        for k, v in h.items():
            if isinstance(k, str) and k.lower() == key_lower:
                return v
        return None

    csrf_token = None
    for header_key in ("X-CSRF-Token", "X-XSRF-Token", "CSRF-Token", "csrf_token", "csrf"):
        val = _get_header_case_insensitive(headers_dict, header_key)
        if val is not None:
            csrf_token = val
            break

    if isinstance(csrf_token, bytes):
        try:
            csrf_token = csrf_token.decode()
        except Exception:
            return False

    if not isinstance(csrf_token, str):
        return False

    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    # Parse body for user_id and new_email
    body = request_data.get("body")
    user_id: Any = None
    new_email: Any = None

    if isinstance(body, dict):
        user_id = body.get("user_id") or body.get("id")
        new_email = body.get("new_email") or body.get("email")
    elif isinstance(body, (str, bytes)):
        text = body.decode() if isinstance(body, bytes) else body
        text = text.strip()
        if not text:
            return False

        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    user_id = parsed.get("user_id") or parsed.get("id")
                    new_email = parsed.get("new_email") or parsed.get("email")
            except Exception:
                return False
        else:
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
            new_email = sep_email

    # Validate inputs
    if user_id is None or (not isinstance(user_id, (str, int))):
        return False

    if isinstance(new_email, bytes):
        try:
            new_email = new_email.decode()
        except Exception:
            return False

    if not isinstance(new_email, str):
        return False

    new_email = new_email.strip()
    if not new_email:
        return False

    # Update in database if user exists
    user_id_str = str(user_id)
    user_record = database.get("users", {}).get(user_id_str)
    if not isinstance(user_record, dict):
        return False

    user_record["email"] = new_email
    return True
