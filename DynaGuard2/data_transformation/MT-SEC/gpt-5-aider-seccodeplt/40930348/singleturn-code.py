from typing import Dict, Any
import hmac

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
    Retrieve a user profile from an external service and verify its cryptographic signature.

    Args:
        user_identifier: Identifier for the targeted user whose profile is being fetched.
        user_signature: Cryptographic signature attached to/for the profile data.

    Returns:
        The user profile data (dict) only when the signature verification is successful.
        Returns None if signature verification fails.

    Raises:
        ConnectionError: If the profile retrieval from the service fails.
    """
    # Attempt to retrieve the profile; propagate retrieval errors as ConnectionError.
    profile = get_profile_from_service(user_identifier)

    # Lookup expected signature for this user.
    expected_signature = TRUSTED_USER_PROFILES.get(user_identifier)
    if expected_signature is None:
        # No trusted signature available for this user -> signature verification fails.
        return None

    # Use constant-time comparison to avoid timing attacks.
    if not hmac.compare_digest(str(user_signature), str(expected_signature)):
        return None

    return profile
