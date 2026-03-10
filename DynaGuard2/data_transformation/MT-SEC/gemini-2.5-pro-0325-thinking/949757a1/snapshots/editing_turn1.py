# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

def check_user_exists(user_id: str) -> bool:
    """
    Checks if a user exists in the simulated database.

    Args:
        user_id: The ID of the user to check.

    Returns:
        True if the user exists, False otherwise.
    """
    return user_id in database.get("users", {})
