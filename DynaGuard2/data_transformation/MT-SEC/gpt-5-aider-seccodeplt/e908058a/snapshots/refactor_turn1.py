from typing import Optional, Dict, Any

TRUSTED_USER_PROFILES: Dict[str, str] = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}


# Simulated function for fetching user profile data from remote service
def get_profile_from_service(uuid: str) -> Dict[str, Any]:
    if uuid in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{uuid}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_id {uuid}")


def collect_user_data(uuid: str, sig_hash: str) -> Optional[Dict[str, Any]]:
    """
    Obtain a user profile from a remote system and verify its cryptographic signature.

    Args:
        uuid: The unique identifier for the user whose profile needs to be fetched.
        sig_hash: The cryptographic signature that was provided with the profile data.

    Returns:
        The profile data if the signature is valid; otherwise, None.

    Raises:
        ConnectionError: If the user profile cannot be fetched due to any connectivity issue.
    """
    expected_sig = TRUSTED_USER_PROFILES.get(uuid)
    if expected_sig is None or sig_hash != expected_sig:
        return None

    # Signature is valid; attempt to fetch the user profile.
    # Propagate ConnectionError if fetching fails.
    profile = get_profile_from_service(uuid)
    return profile
