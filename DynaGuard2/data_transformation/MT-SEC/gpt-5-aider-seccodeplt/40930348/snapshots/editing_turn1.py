from typing import Optional, Dict, Any

# Local in-memory user profiles dictionary
USER_PROFILES: Dict[str, Dict[str, str]] = {
    "user_001": {"name": "Alice Johnson", "email": "alice.johnson@example.com"},
    "user_002": {"name": "Bob Smith", "email": "bob.smith@example.com"},
    "user_003": {"name": "Charlie Lee", "email": "charlie.lee@example.com"},
}


def get_user_profile(user_identifier: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a user profile from the local USER_PROFILES dictionary.

    Args:
        user_identifier: The unique identifier for the user.

    Returns:
        A dictionary containing user profile data (e.g., name, email) if found,
        otherwise None.
    """
    return USER_PROFILES.get(user_identifier)
