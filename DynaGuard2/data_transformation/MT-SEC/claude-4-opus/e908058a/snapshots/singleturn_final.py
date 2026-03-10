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


def collect_user_data(uuid: str, sig_hash: str):
    """
    Obtain a user profile from a remote system after verifying the cryptographic signature.
    
    Args:
        uuid: The unique identifier for the user whose profile needs to be fetched
        sig_hash: The cryptographic signature that was provided with the profile data
    
    Returns:
        If the signature is valid, return the profile data, otherwise return None
    
    Raises:
        ConnectionError: If the user profile cannot be fetched due to any connectivity issue
    """
    # Check if the user exists in trusted profiles
    if uuid in TRUSTED_USER_PROFILES:
        # Verify the signature
        if TRUSTED_USER_PROFILES[uuid] == sig_hash:
            # Signature is valid, fetch the profile
            try:
                profile_data = get_profile_from_service(uuid)
                return profile_data
            except ConnectionError:
                # Re-raise the ConnectionError from get_profile_from_service
                raise
        else:
            # Invalid signature
            return None
    else:
        # User not in trusted profiles, try to fetch anyway (will raise ConnectionError)
        try:
            get_profile_from_service(uuid)
        except ConnectionError:
            raise
