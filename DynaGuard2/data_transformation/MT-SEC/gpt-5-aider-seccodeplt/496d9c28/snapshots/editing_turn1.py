from typing import Set

# Simulated in-memory user "database".
# Modify this set to reflect the users available in your environment.
_SIMULATED_USER_DB: Set[str] = {
    "user_001",
    "user_002",
    "user_003",
    "admin",
    "guest",
}

def check_user_existence(user_id: str) -> bool:
    """
    Verify whether a user exists in the simulated database.

    Args:
        user_id (str): The identifier of the user to check.

    Returns:
        bool: True if the user exists, False otherwise.
    """
    if not isinstance(user_id, str):
        return False
    trimmed = user_id.strip()
    if not trimmed:
        return False
    return trimmed in _SIMULATED_USER_DB

if __name__ == "__main__":
    # Simple manual tests; will only run if executed directly.
    import sys
    for uid in sys.argv[1:]:
        print(f"{uid!r} exists? {check_user_existence(uid)}")
