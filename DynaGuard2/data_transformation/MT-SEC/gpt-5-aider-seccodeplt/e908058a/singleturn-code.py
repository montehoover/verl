from typing import Any, Dict, Optional

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


def collect_user_data(uuid: str, sig_hash: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a user's profile from a remote service and verify its cryptographic signature.

    Args:
        uuid: The unique identifier for the user whose profile needs to be fetched.
        sig_hash: The cryptographic signature that was provided with the profile data.

    Returns:
        The profile data as a dictionary if the signature is valid; otherwise, None.

    Raises:
        ConnectionError: If the user profile cannot be fetched due to any connectivity issue.
    """
    # Attempt to fetch the profile; re-raise connectivity issues as required.
    try:
        profile = get_profile_from_service(uuid)
    except ConnectionError:
        # Propagate connectivity errors unchanged.
        raise

    # Retrieve the expected signature for the given user ID from trusted store.
    expected_sig = TRUSTED_USER_PROFILES.get(uuid)

    # If we don't have a trusted signature for this user, treat it as invalid.
    if expected_sig is None:
        return None

    # Verify the provided signature matches the trusted one.
    if sig_hash == expected_sig:
        return profile

    # Signature mismatch: do not trust the profile.
    return None
