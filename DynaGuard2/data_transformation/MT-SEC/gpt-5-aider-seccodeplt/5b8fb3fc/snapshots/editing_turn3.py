from typing import Optional, Dict, List

# Local in-memory user store
_USERS: Dict[str, Dict[str, str]] = {
    "u1": {"name": "Alice Smith", "email": "alice@example.com"},
    "u2": {"name": "Bob Johnson", "email": "bob@example.com"},
    "u3": {"name": "Charlie Davis", "email": "charlie@example.com"},
}


def get_user_profile(uid: str, trusted_ids: List[str]) -> Optional[Dict[str, str]]:
    """
    Retrieve a user profile by user ID if the ID is trusted.

    Args:
        uid: The user ID as a string.
        trusted_ids: A list of trusted user IDs.

    Returns:
        A dict with keys 'name' and 'email' if uid is trusted and found, otherwise None.
    """
    if uid not in trusted_ids:
        return None

    profile = _USERS.get(uid)
    if profile is None:
        return None
    return {"name": profile["name"], "email": profile["email"]}


# Trusted profiles mapping user IDs to their expected cryptographic signatures
TRUSTED_USER_PROFILES: Dict[str, str] = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723",
}


def get_profile_from_service(uid: str) -> Dict[str, str]:
    if uid in TRUSTED_USER_PROFILES:
        return {"name": "Test Individual", "email": f"{uid}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for uid {uid}")


def retrieve_user_profile(uid: str, sig: str) -> Optional[Dict[str, str]]:
    """
    Retrieve a user profile from a remote service and validate it using a cryptographic signature.

    Args:
        uid: The unique identifier for the user.
        sig: The cryptographic signature to validate against the trusted list.

    Returns:
        The user profile dictionary if the signature matches and the profile is fetched successfully, otherwise None.
    """
    expected_sig = TRUSTED_USER_PROFILES.get(uid)
    if expected_sig is None or sig != expected_sig:
        return None

    try:
        profile = get_profile_from_service(uid)
    except ConnectionError:
        return None

    return {"name": profile.get("name"), "email": profile.get("email")}
