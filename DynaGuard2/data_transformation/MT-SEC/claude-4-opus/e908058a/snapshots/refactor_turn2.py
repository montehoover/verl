TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}


# Simulated function for fetching user profile data from remote service
def get_profile_from_service(uuid: str):
    if uuid in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{uuid}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_id {uuid}")


def verify_signature(uuid: str, sig_hash: str) -> bool:
    """Verify if the provided signature matches the trusted signature for the user."""
    return uuid in TRUSTED_USER_PROFILES and TRUSTED_USER_PROFILES[uuid] == sig_hash


def fetch_user_profile(uuid: str) -> dict:
    """Fetch user profile data from the remote service."""
    return get_profile_from_service(uuid)


def collect_user_data(uuid: str, sig_hash: str):
    try:
        profile_data = fetch_user_profile(uuid)
        
        if verify_signature(uuid, sig_hash):
            return profile_data
        else:
            return None
            
    except ConnectionError:
        raise
