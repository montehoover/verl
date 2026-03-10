from typing import Dict, Any
import hashlib

# Sample user data for demonstration purposes
USER_DATA: Dict[str, Dict[str, Any]] = {
    "u123": {"name": "Alice", "age": 30, "email": "alice@example.com"},
    "u456": {"name": "Bob", "age": 25, "email": "bob@example.com"},
    "u789": {"name": "Charlie", "age": 28, "email": "charlie@example.com"},
}

def get_user_profile(user_id: str) -> Dict[str, Any]:
    """
    Retrieve the user profile from USER_DATA by user_id.

    Args:
        user_id: The ID of the user whose profile to retrieve.

    Returns:
        The profile data as a dictionary.

    Raises:
        KeyError: If the user_id does not exist in USER_DATA.
    """
    return USER_DATA[user_id]

def check_signature(data: str, signature: str) -> bool:
    """
    Compute the SHA-256 hash of the input data and compare it to the provided signature.

    Args:
        data: The input string to hash.
        signature: The expected hexadecimal SHA-256 hash string.

    Returns:
        True if the computed hash matches the provided signature, False otherwise.
    """
    computed_hash = hashlib.sha256(data.encode("utf-8")).hexdigest()
    return computed_hash == signature

TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}

# Simulated function for fetching user profile data from remote service
def get_profile_from_service(uid: str):
    if uid in TRUSTED_USER_PROFILES:
        return {"name": "Test Individual", "email": f"{uid}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for uid {uid}")

def retrieve_user_profile(uid: str, sig: str) -> Dict[str, Any]:
    """
    Retrieve a user profile from a remote service after verifying its cryptographic signature.

    Args:
        uid: The user identifier.
        sig: The provided cryptographic signature for the user.

    Returns:
        The profile data as a dictionary.

    Raises:
        KeyError: If the uid is not recognized (not present in TRUSTED_USER_PROFILES).
        ValueError: If the provided signature does not match the expected signature.
        ConnectionError: If the remote service fails to return the profile.
    """
    # Ensure the uid is recognized and has an expected signature
    if uid not in TRUSTED_USER_PROFILES:
        raise KeyError(f"Unrecognized uid {uid}")

    expected_sig = TRUSTED_USER_PROFILES[uid]
    if sig != expected_sig:
        raise ValueError("Signature mismatch for provided uid")

    # Fetch the profile from the remote service
    profile = get_profile_from_service(uid)
    return profile
