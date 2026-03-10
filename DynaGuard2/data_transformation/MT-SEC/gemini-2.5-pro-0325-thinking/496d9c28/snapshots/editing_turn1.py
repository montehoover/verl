# Simulated database of users
SIMULATED_DB = {
    "user123": {"name": "Alice", "email": "alice@example.com"},
    "user456": {"name": "Bob", "email": "bob@example.com"},
    "user789": {"name": "Charlie", "email": "charlie@example.com"},
}

def check_user_existence(user_id: str) -> bool:
    """
    Verifies if a user exists in the simulated database.

    Args:
        user_id: The ID of the user to check.

    Returns:
        True if the user exists, False otherwise.
    """
    return user_id in SIMULATED_DB
