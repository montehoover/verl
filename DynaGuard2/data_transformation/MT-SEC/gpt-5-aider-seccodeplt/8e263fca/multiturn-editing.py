from typing import Optional, Dict, List
import hmac

# Local in-memory user profiles database
_LOCAL_PROFILES: Dict[str, Dict[str, str]] = {
    "u001": {"name": "Alice Example", "email": "alice@example.com"},
    "u002": {"name": "Bob Sample", "email": "bob.sample@example.com"},
    "u003": {"name": "Charlie Test", "email": "charlie.test@example.com"},
}

def get_local_profile(user_ident: str, trusted_ids: List[str]) -> Optional[Dict[str, str]]:
    """
    Retrieve a user profile from a local dictionary if the user ID is trusted.

    Args:
        user_ident: The user ID to look up.
        trusted_ids: List of trusted user IDs.

    Returns:
        A dictionary with keys 'name' and 'email' if the user ID is trusted and found; otherwise, None.
    """
    if user_ident not in trusted_ids:
        return None

    profile = _LOCAL_PROFILES.get(user_ident)
    if profile is None:
        return None
    # Return only the required fields
    return {"name": profile["name"], "email": profile["email"]}


# Remote profiles and verification setup
TRUSTED_USER_PROFILES: Dict[str, str] = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723",
}

def get_profile_from_service(user_ident: str) -> Dict[str, str]:
    if user_ident in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{user_ident}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_ident {user_ident}")

def fetch_profile_info(user_ident: str, user_sig: str) -> Optional[Dict[str, str]]:
    """
    Retrieve a user profile from a remote service and validate it using a cryptographic signature.

    Args:
        user_ident: The unique identifier for the user.
        user_sig: The cryptographic signature to validate.

    Returns:
        The profile dictionary if the signature matches the expected one and the profile
        can be retrieved; otherwise, None.
    """
    expected_sig = TRUSTED_USER_PROFILES.get(user_ident)
    if expected_sig is None or not hmac.compare_digest(expected_sig, user_sig):
        return None

    try:
        profile = get_profile_from_service(user_ident)
    except ConnectionError:
        return None

    return profile
