from typing import Dict, Any, Optional
import hashlib
import json

# Sample local user database for context
USER_DATABASE: Dict[str, Dict[str, Any]] = {
    "u_1001": {
        "id": "u_1001",
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "role": "admin",
        "active": True,
    },
    "u_1002": {
        "id": "u_1002",
        "name": "Bob Smith",
        "email": "bob@example.com",
        "role": "user",
        "active": False,
    },
    "u_1003": {
        "id": "u_1003",
        "name": "Carol Lee",
        "email": "carol@example.com",
        "role": "moderator",
        "active": True,
    },
}

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


def fetch_user_profile(user_id: str) -> Dict[str, Any]:
    """
    Retrieve the profile data for a given user_id from USER_DATABASE.

    Args:
        user_id: The unique identifier of the user.

    Returns:
        The profile data dictionary for the specified user.

    Raises:
        KeyError: If the user_id does not exist in USER_DATABASE.
    """
    # Will raise KeyError automatically if user_id is missing
    return USER_DATABASE[user_id]


def verify_signature(data: str, signature: str) -> bool:
    """
    Compute a SHA-256 hash of the input data and compare it to the provided signature.

    Args:
        data: The input string to hash.
        signature: The expected SHA-256 hex digest.

    Returns:
        True if the computed hash matches the provided signature, False otherwise.
    """
    computed = hashlib.sha256(data.encode("utf-8")).hexdigest()
    return computed == signature


def collect_user_data(uuid: str, sig_hash: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a user profile from a remote system and verify its cryptographic signature.

    Args:
        uuid: The user identifier to fetch remotely.
        sig_hash: The provided signature to verify against the profile contents.

    Returns:
        The profile data if the signature is valid; otherwise, None.

    Raises:
        ConnectionError: If the profile cannot be fetched from the remote service.
    """
    profile = get_profile_from_service(uuid)
    # Create a deterministic string representation of the profile for hashing
    data_str = json.dumps(profile, sort_keys=True, separators=(',', ':'))
    if verify_signature(data_str, sig_hash):
        return profile
    return None
