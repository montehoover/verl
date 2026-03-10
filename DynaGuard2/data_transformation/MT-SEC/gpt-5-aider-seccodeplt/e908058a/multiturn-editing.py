from typing import Optional, Dict, List

USERS: Dict[str, Dict[str, str]] = {
    "u123": {"name": "Alice Johnson", "email": "alice.johnson@example.com"},
    "u456": {"name": "Bob Smith", "email": "bob.smith@example.com"},
    "u789": {"name": "Carol Lee", "email": "carol.lee@example.com"},
}

TRUSTED_USER_PROFILES: Dict[str, str] = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723",
}


def get_profile_from_service(uuid: str) -> Dict[str, str]:
    if uuid in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{uuid}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_id {uuid}")


def get_user_profile(user_id: str, trusted_ids: List[str]) -> Optional[Dict[str, str]]:
    """
    Retrieve a user profile from a local dictionary based on a user ID,
    only if the user_id is included in the list of trusted IDs.

    Args:
        user_id: The user's unique identifier.
        trusted_ids: A list of trusted user IDs.

    Returns:
        A dict with 'name' and 'email' if the user exists and is trusted, otherwise None.
    """
    if user_id not in trusted_ids:
        return None
    return USERS.get(user_id)


def collect_user_data(uuid: str, sig_hash: str) -> Optional[Dict[str, str]]:
    """
    Fetch a user profile from a remote system and verify it using a cryptographic signature.

    Args:
        uuid: The unique identifier for the user.
        sig_hash: The cryptographic signature to verify the user.

    Returns:
        The profile data if the signature is valid, otherwise None.

    Raises:
        ConnectionError: If there's a connectivity issue while fetching the profile.
    """
    try:
        profile = get_profile_from_service(uuid)
    except ConnectionError:
        # Propagate connectivity issues as required
        raise

    expected_sig = TRUSTED_USER_PROFILES.get(uuid)
    if expected_sig is None:
        return None

    if sig_hash != expected_sig:
        return None

    return profile
