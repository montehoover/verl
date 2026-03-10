from typing import Any, Dict
import hashlib
import json

# Sample in-memory user database for context.
USER_DATABASE: Dict[str, Dict[str, Any]] = {
    "u_001": {"id": "u_001", "name": "Alice Johnson", "email": "alice@example.com", "age": 30},
    "u_002": {"id": "u_002", "name": "Bob Smith", "email": "bob@example.com", "age": 27},
    "u_003": {"id": "u_003", "name": "Charlie Kim", "email": "charlie@example.com", "age": 35},
}


def fetch_user_profile(user_id: str) -> Dict[str, Any]:
    """
    Retrieve a user's profile from USER_DATABASE by user_id.

    Args:
        user_id: The identifier of the user whose profile is to be fetched.

    Returns:
        The profile dictionary for the specified user.

    Raises:
        KeyError: If the user_id does not exist in USER_DATABASE.
    """
    if user_id in USER_DATABASE:
        return USER_DATABASE[user_id]
    raise KeyError(f"User ID '{user_id}' not found in USER_DATABASE.")


def verify_signature(data: str, signature: str) -> bool:
    """
    Compute the SHA-256 hash of the provided data and compare it to the given signature.

    Args:
        data: The input string to hash.
        signature: The expected hex-encoded SHA-256 digest.

    Returns:
        True if the computed digest matches the provided signature; otherwise, False.
    """
    computed_digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
    return computed_digest == signature.strip().lower()


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


def collect_user_data(user_identifier: str, user_signature: str) -> Dict[str, Any]:
    """
    Fetch a user profile from a remote service and verify its cryptographic signature.

    Steps:
      - Ensure the user_identifier is trusted and the provided signature matches the trusted signature.
      - Fetch the profile from the external service.
      - Serialize the profile deterministically and verify its signature using verify_signature.
      - Return the profile data if verification succeeds.

    Args:
        user_identifier: The external user identifier.
        user_signature: The claimed hex-encoded signature of the profile data.

    Returns:
        The fetched profile dictionary on successful verification.

    Raises:
        KeyError: If the user_identifier is not found in TRUSTED_USER_PROFILES.
        ValueError: If the provided signature does not match the trusted signature
                    or the computed signature does not match the provided signature.
        ConnectionError: If fetching the profile from the service fails.
    """
    if user_identifier not in TRUSTED_USER_PROFILES:
        raise KeyError(f"Unrecognized user_identifier '{user_identifier}'")

    trusted_signature = TRUSTED_USER_PROFILES[user_identifier].strip().lower()
    provided_signature = user_signature.strip().lower()

    if provided_signature != trusted_signature:
        raise ValueError("Provided signature does not match trusted signature for this user.")

    profile = get_profile_from_service(user_identifier)

    # Deterministic serialization for signing/verification.
    serialized_profile = json.dumps(profile, sort_keys=True, separators=(",", ":"))

    if not verify_signature(serialized_profile, provided_signature):
        raise ValueError("Signature verification failed for the retrieved profile data.")

    return profile
