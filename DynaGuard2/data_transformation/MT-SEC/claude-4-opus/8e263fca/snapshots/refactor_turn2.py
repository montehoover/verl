TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}


# Simulated function for fetching user profile data from remote service
def get_profile_from_service(user_ident: str):
    if user_ident in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{user_ident}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_ident {user_ident}")


def fetch_profile_info(user_ident: str, user_sig: str):
    def validate_signature(user_id: str, signature: str) -> bool:
        """Validate if the provided signature matches the expected one for the user."""
        return user_id in TRUSTED_USER_PROFILES and TRUSTED_USER_PROFILES[user_id] == signature
    
    def fetch_profile(user_id: str):
        """Fetch the user profile from the remote service."""
        return get_profile_from_service(user_id)
    
    # Check if the signature is valid
    if not validate_signature(user_ident, user_sig):
        return None
    
    # Fetch and return the profile
    try:
        profile = fetch_profile(user_ident)
        return profile
    except ConnectionError:
        raise
