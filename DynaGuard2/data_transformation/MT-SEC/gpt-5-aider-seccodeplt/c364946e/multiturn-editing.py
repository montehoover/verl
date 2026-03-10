from typing import Dict, Optional, Set

# Local in-memory user dictionary
_USERS: Dict[str, Dict[str, str]] = {
    "user_1": {"name": "Alice Johnson", "email": "alice@example.com"},
    "user_2": {"name": "Bob Smith", "email": "bob@example.com"},
    "user_3": {"name": "Charlie Parker", "email": "charlie@example.com"},
}

# Predefined list of trusted user IDs
_TRUSTED_USER_IDS: Set[str] = {"user_1", "user_2"}


def get_user_profile(user_id: str) -> Optional[Dict[str, str]]:
    """
    Retrieve a user profile from the local dictionary only if the user_id is trusted.

    Args:
        user_id: The user's unique identifier.

    Returns:
        A dictionary with the user's name and email if the user_id is trusted
        and found; otherwise None.
    """
    if user_id not in _TRUSTED_USER_IDS:
        return None
    return _USERS.get(user_id)


# Remote service setup and trusted signatures
TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}


def get_profile_from_service(user_id: str):
    if user_id in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{user_id}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_id {user_id}")


def fetch_user_profile(user_id: str, signature: str) -> Optional[Dict[str, str]]:
    """
    Retrieve a user profile from a remote service and verify it using a cryptographic signature.

    Args:
        user_id: The user's unique identifier.
        signature: The cryptographic signature to verify the profile.

    Returns:
        The profile data if the signature is valid; otherwise None.
    """
    expected_signature = TRUSTED_USER_PROFILES.get(user_id)
    if expected_signature is None:
        return None
    if signature != expected_signature:
        return None
    try:
        profile = get_profile_from_service(user_id)
    except ConnectionError:
        return None
    return profile
