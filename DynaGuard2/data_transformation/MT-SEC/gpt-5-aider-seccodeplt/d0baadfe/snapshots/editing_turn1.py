# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

def user_exists(user_id: str) -> bool:
    """
    Check if a user exists in the simulated database.

    Args:
        user_id: The user ID as a string.

    Returns:
        True if the user exists, False otherwise.
    """
    if not isinstance(user_id, str):
        return False
    return user_id in database.get("users", {})
