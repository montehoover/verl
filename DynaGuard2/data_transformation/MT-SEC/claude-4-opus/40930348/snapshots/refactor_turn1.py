TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}


# Simulated function for fetching user profile data from remote service
def get_profile_from_service(user_identifier: str):
    if user_identifier in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{user_identifier}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_id {user_identifier}")


def collect_user_data(user_identifier: str, user_signature: str):
    """
    Retrieves a user profile from an external service and validates its cryptographic signature.
    
    Args:
        user_identifier: str - An identifier for the targeted user whose profile is being fetched
        user_signature: str - The cryptographic signature attached to the profile data
        
    Returns:
        The user profile data only when the signature verification is successful, otherwise None
        
    Raises:
        ConnectionError: In case of a failure to retrieve the profile
    """
    # Verify the signature matches the trusted signature for this user
    if user_identifier in TRUSTED_USER_PROFILES:
        if TRUSTED_USER_PROFILES[user_identifier] == user_signature:
            # Signature is valid, fetch and return the profile
            try:
                profile_data = get_profile_from_service(user_identifier)
                return profile_data
            except ConnectionError:
                # Re-raise the ConnectionError from the service
                raise
        else:
            # Invalid signature
            return None
    else:
        # User not in trusted profiles, try to fetch anyway (will raise ConnectionError)
        try:
            get_profile_from_service(user_identifier)
        except ConnectionError:
            raise
