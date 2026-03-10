TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}

def get_profile_from_service(user_ident: str):
    if user_ident in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{user_ident}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_ident {user_ident}")

def fetch_profile_info(user_ident, user_sig):
    # Check if the user_ident exists in trusted profiles
    if user_ident not in TRUSTED_USER_PROFILES:
        return None
    
    # Validate the cryptographic signature
    if TRUSTED_USER_PROFILES[user_ident] != user_sig:
        return None
    
    # Fetch the profile from the service
    try:
        profile = get_profile_from_service(user_ident)
        return profile
    except ConnectionError:
        return None
