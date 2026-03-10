# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

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
