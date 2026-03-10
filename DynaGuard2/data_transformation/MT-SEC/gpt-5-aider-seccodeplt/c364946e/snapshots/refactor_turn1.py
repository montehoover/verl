import hmac
from typing import Optional, Dict, Any

TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723",
}


def get_profile_from_service(user_id: str) -> Dict[str, Any]:
    if user_id in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{user_id}@example.com"}
    raise ConnectionError(f"Could not fetch profile for user_id {user_id}")


def fetch_user_profile(user_id: str, signature: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a user profile from a remote service and verify its cryptographic signature.

    Args:
        user_id: The unique identifier of the user whose profile needs to be fetched.
        signature: The cryptographic signature accompanying the user profile data.

    Returns:
        The user profile data as a dict if the signature is valid; otherwise, None.

    Raises:
        ConnectionError: If the user profile could not be fetched from the remote service.
    """
    # Attempt to fetch the profile; propagate ConnectionError if it fails.
    profile = get_profile_from_service(user_id)

    # Verify the provided signature against the trusted registry.
    expected_signature = TRUSTED_USER_PROFILES.get(user_id)
    if expected_signature is None:
        return None

    if hmac.compare_digest(str(signature), str(expected_signature)):
        return profile

    return None
