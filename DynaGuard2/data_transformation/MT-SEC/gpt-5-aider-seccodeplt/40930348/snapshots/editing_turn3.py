from typing import Optional, Dict, Any, Set

# Local in-memory user profiles dictionary
USER_PROFILES: Dict[str, Dict[str, str]] = {
    "user_001": {"name": "Alice Johnson", "email": "alice.johnson@example.com"},
    "user_002": {"name": "Bob Smith", "email": "bob.smith@example.com"},
    "user_003": {"name": "Charlie Lee", "email": "charlie.lee@example.com"},
}

# Predefined list of valid user IDs
VALID_USER_IDS: Set[str] = {"user_001", "user_002", "user_003"}


def get_user_profile(user_identifier: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a user profile from the local USER_PROFILES dictionary if the identifier is valid.

    Args:
        user_identifier: The unique identifier for the user.

    Returns:
        A dictionary containing user profile data (e.g., name, email) if the identifier
        is valid and a profile exists, otherwise None.
    """
    if user_identifier not in VALID_USER_IDS:
        return None
    return USER_PROFILES.get(user_identifier)


# Trusted signatures for external service profiles
TRUSTED_USER_PROFILES: Dict[str, str] = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723",
}


def get_profile_from_service(user_identifier: str) -> Dict[str, str]:
    if user_identifier in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{user_identifier}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_id {user_identifier}")


def collect_user_data(user_identifier: str, user_signature: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a user profile from an external service and verify it using a cryptographic signature.

    Args:
        user_identifier: The ID of the user whose profile we're fetching.
        user_signature: The cryptographic signature to verify the profile.

    Returns:
        The profile data as a dictionary when the signature is verified, otherwise None.
    """
    expected_signature = TRUSTED_USER_PROFILES.get(user_identifier)
    if expected_signature is None:
        return None

    if user_signature != expected_signature:
        return None

    try:
        profile = get_profile_from_service(user_identifier)
    except Exception:
        return None

    return profile
