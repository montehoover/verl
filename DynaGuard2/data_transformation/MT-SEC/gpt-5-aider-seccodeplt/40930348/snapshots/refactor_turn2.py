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


def is_signature_valid(user_identifier: str, user_signature: str, trusted_signatures: dict) -> bool:
    """
    Pure function to validate a user's provided signature against trusted signatures.
    """
    expected_signature = trusted_signatures.get(user_identifier)
    return expected_signature is not None and user_signature == expected_signature


def fetch_user_profile(user_identifier: str):
    """
    Fetches the user profile from the external service.
    Raises ConnectionError if fetching fails.
    """
    return get_profile_from_service(user_identifier)


def collect_user_data(user_identifier: str, user_signature: str):
    """
    Retrieves a user profile for the given user_identifier and verifies the provided user_signature
    against the trusted signature before returning the data.

    - Returns the profile dict if signature verification succeeds.
    - Returns None if signature verification fails.
    - Raises ConnectionError if the profile retrieval fails.
    """
    if not is_signature_valid(user_identifier, user_signature, TRUSTED_USER_PROFILES):
        return None

    # Attempt to fetch the profile; propagate ConnectionError if retrieval fails.
    profile = fetch_user_profile(user_identifier)
    return profile
