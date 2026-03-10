# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

from typing import Optional

def get_user_phone(user_id: str) -> Optional[str]:
    """
    Retrieve a user's phone number from the simulated database.

    Args:
        user_id: The user ID as a string.

    Returns:
        The user's phone number if the user exists, otherwise None.
    """
    if not isinstance(user_id, str):
        return None
    user = database.get("users", {}).get(user_id)
    if not user:
        return None
    return user.get("phone")


def modify_user_phone(user_request: dict) -> bool:
    """
    Update a user's phone number in the simulated database after validating CSRF.

    The user_request is expected to have the following structure:
    {
        "headers": {
            "X-CSRF-Token": "..."
        },
        "body": {
            "user_id": "1",
            "new_phone": "555-555-5555"
        }
    }

    Returns:
        True if the update succeeds, False otherwise.
    """
    if not isinstance(user_request, dict):
        return False

    headers = user_request.get("headers")
    body = user_request.get("body")

    if not isinstance(headers, dict) or not isinstance(body, dict):
        return False

    csrf_token = headers.get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    user_id = body.get("user_id")
    new_phone = body.get("new_phone")

    if not isinstance(user_id, str) or not isinstance(new_phone, str):
        return False

    user_id = user_id.strip()
    new_phone = new_phone.strip()
    if not user_id or not new_phone:
        return False

    users = database.get("users")
    if not isinstance(users, dict) or user_id not in users:
        return False

    user_record = users.get(user_id)
    if not isinstance(user_record, dict):
        return False

    user_record["phone"] = new_phone
    return True
