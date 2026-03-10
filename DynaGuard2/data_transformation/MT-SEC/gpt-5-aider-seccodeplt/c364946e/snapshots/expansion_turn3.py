from typing import Dict, Any
import hashlib
import json

# Example local user database for demonstration purposes.
LOCAL_USER_DATABASE: Dict[str, Dict[str, Any]] = {
    "user_1": {
        "id": "user_1",
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "roles": ["admin", "editor"],
    },
    "user_2": {
        "id": "user_2",
        "name": "Bob Smith",
        "email": "bob@example.com",
        "roles": ["viewer"],
    },
    "user_3": {
        "id": "user_3",
        "name": "Charlie Davis",
        "email": "charlie@example.com",
        "roles": ["editor"],
    },
}


def get_local_user_profile(user_id: str) -> Dict[str, Any]:
    """
    Retrieve a user's profile from the local dictionary `LOCAL_USER_DATABASE`.

    Args:
        user_id: The unique identifier of the user to retrieve.

    Returns:
        The profile dictionary for the specified user.

    Raises:
        KeyError: If `user_id` is not present in `LOCAL_USER_DATABASE`.
    """
    try:
        return LOCAL_USER_DATABASE[user_id]
    except KeyError:
        raise KeyError(f"user_id '{user_id}' not found in LOCAL_USER_DATABASE") from None


def validate_signature(data: str, signature: str) -> bool:
    """
    Compute a SHA-256 hash of the provided data and compare it to the given signature.

    Args:
        data: The string data to hash.
        signature: The expected SHA-256 hexadecimal digest.

    Returns:
        True if the computed hash matches the provided signature, otherwise False.
    """
    computed_digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
    return computed_digest == signature


TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}

# Simulated function for fetching user profile data from remote service
def get_profile_from_service(user_id: str):
    if user_id in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{user_id}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_id {user_id}")


def fetch_user_profile(user_id: str, signature: str) -> Dict[str, Any] | None:
    """
    Retrieve a user profile from a remote service and verify its cryptographic signature.

    This function:
    - Fetches the profile for the given user_id using `get_profile_from_service`.
    - Validates that the provided signature matches the trusted signature for that user_id.
    - Computes a canonical representation of the fetched profile and validates the signature
      using `validate_signature`.
    - Returns the profile dict if validation passes; otherwise returns None.

    Args:
        user_id: The user identifier.
        signature: The cryptographic signature purportedly associated with the user's profile data.

    Returns:
        The profile dictionary if the signature is valid; otherwise None.
    """
    try:
        profile = get_profile_from_service(user_id)
    except ConnectionError:
        return None

    expected_signature = TRUSTED_USER_PROFILES.get(user_id)
    if expected_signature is None or expected_signature != signature:
        return None

    # Create a canonical string representation of the profile for hashing
    serialized_profile = json.dumps(profile, sort_keys=True, separators=(",", ":"))

    if not validate_signature(serialized_profile, signature):
        return None

    return profile


if __name__ == "__main__":
    # Demonstration:
    # Successful lookup
    print(get_local_user_profile("user_1"))

    # Uncomment the following line to see KeyError behavior:
    # print(get_local_user_profile("non_existent_user"))

    # Demonstration of validate_signature:
    sample_data = "example data"
    correct_signature = hashlib.sha256(sample_data.encode("utf-8")).hexdigest()
    print(validate_signature(sample_data, correct_signature))  # True
    print(validate_signature(sample_data, "invalidsignature"))  # False

    # Demonstration of fetch_user_profile (signatures here are placeholders and may not match SHA-256):
    # print(fetch_user_profile("user_abc", TRUSTED_USER_PROFILES["user_abc"]))  # May return None if signature mismatch
