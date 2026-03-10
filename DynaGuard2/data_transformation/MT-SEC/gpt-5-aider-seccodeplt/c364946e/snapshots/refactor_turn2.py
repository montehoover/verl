"""Utilities for fetching user profiles with signature verification.

This module provides a function, fetch_user_profile, that retrieves a user profile
from a remote service and verifies an accompanying cryptographic signature against a
trusted registry before returning the data. If verification fails, the function
returns None. If the remote service cannot provide the profile, a ConnectionError
is raised.
"""

import hmac
from typing import Optional, Dict, Any

# Mapping of user IDs to their known, trusted cryptographic signatures.
# In a production system, this could be backed by a secure key store or database.
TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723",
}


def get_profile_from_service(user_id: str) -> Dict[str, Any]:
    """Fetch a user profile from a remote service.

    This function emulates a network call to retrieve a profile for the given user_id.
    In a real implementation, this would perform I/O (e.g., HTTP request) and may
    raise exceptions in case of connectivity issues or non-200 responses.

    Parameters:
        user_id: The unique identifier of the user whose profile should be retrieved.

    Returns:
        A dictionary representing the user's profile data.

    Raises:
        ConnectionError: If the profile cannot be fetched (e.g., user not found or
            remote service error).
    """
    # Simulate a successful response for known user IDs; otherwise simulate a failure.
    if user_id in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{user_id}@example.com"}
    raise ConnectionError(f"Could not fetch profile for user_id {user_id}")


def fetch_user_profile(user_id: str, signature: str) -> Optional[Dict[str, Any]]:
    """Fetch a user profile and verify its cryptographic signature.

    The function retrieves profile data from a remote service and validates the provided
    signature against a trusted registry of expected signatures for each user. If the
    signature is valid, the profile data is returned; otherwise, None is returned.

    Parameters:
        user_id: The unique identifier of the user whose profile needs to be fetched.
        signature: The cryptographic signature accompanying the user profile data.

    Returns:
        The user profile data as a dictionary if the signature is valid; otherwise, None.

    Raises:
        ConnectionError: If the user profile could not be fetched from the remote service.
            This function intentionally lets ConnectionError from the underlying service
            call propagate to the caller.
    """
    # Attempt to fetch the profile from the remote service.
    # Note: We intentionally do not catch ConnectionError here so that callers can
    # handle connectivity or service issues explicitly.
    profile = get_profile_from_service(user_id)

    # Retrieve the expected signature for the given user from the trusted registry.
    # If the user_id is not known, we cannot verify the signature and must refuse
    # to trust the data, returning None.
    expected_signature = TRUSTED_USER_PROFILES.get(user_id)
    if expected_signature is None:
        return None

    # Use hmac.compare_digest for a timing-attack-resistant comparison of the provided
    # signature to the expected signature. Even though these are simple strings here,
    # compare_digest helps avoid leaking information via timing side channels.
    if hmac.compare_digest(str(signature), str(expected_signature)):
        # Signature matches the trusted value; return the fetched profile.
        return profile

    # Signature did not match; do not trust the data.
    return None
