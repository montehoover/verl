# Fallback setup in case the environment doesn't provide these globals.
try:
    database  # type: ignore[name-defined]
except NameError:
    database = {
        "users": {
            "1": {"phone": "123-456-7890"},
        }
    }

try:
    EXPECTED_CSRF_TOKEN  # type: ignore[name-defined]
except NameError:
    EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def change_user_phone(request_details: dict) -> bool:
    """
    Modify a user's phone number in the mock database.

    Args:
        request_details (dict): A dictionary with the following structure:
            {
                "headers": {"X-CSRF-Token": "<token>"},
                "body": {"user_id": "<id>", "new_phone": "<phone>"}
            }

    Returns:
        bool: True if the update succeeds, otherwise False.
    """
    if not isinstance(request_details, dict):
        return False

    headers = request_details.get("headers")
    body = request_details.get("body")

    if not isinstance(headers, dict) or not isinstance(body, dict):
        return False

    # CSRF validation
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

    # Access users in the mock database
    if not isinstance(database, dict):
        return False

    users = database.get("users")
    if not isinstance(users, dict):
        return False

    if user_id not in users or not isinstance(users[user_id], dict):
        return False

    # Perform the update
    users[user_id]["phone"] = new_phone
    return True
