# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

def user_exists(user_id: str) -> bool:
    """
    Checks if a user exists in the simulated database.

    Args:
        user_id: The ID of the user to check.

    Returns:
        True if the user exists, False otherwise.
    """
    return user_id in database["users"]

if __name__ == '__main__':
    # Example usage:
    print(f"User '1' exists: {user_exists('1')}")
    print(f"User '2' exists: {user_exists('2')}")
