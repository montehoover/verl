TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}

def get_profile_from_service(user_ident: str):
    if user_ident in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{user_ident}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_ident {user_ident}")

def fetch_profile_info(user_ident: str, user_sig: str):
    """
    Retrieves a user profile from a remote service and validates it
    using a cryptographic signature.

    Args:
        user_ident: The unique identifier for the user (str).
        user_sig: The cryptographic signature (str).

    Returns:
        The profile dictionary if the signature matches the expected one,
        otherwise None.
    """
    expected_sig = TRUSTED_USER_PROFILES.get(user_ident)
    if expected_sig is None or user_sig != expected_sig:
        return None
    
    try:
        profile = get_profile_from_service(user_ident)
        return profile
    except ConnectionError:
        # Optionally log the error or handle it differently
        return None
