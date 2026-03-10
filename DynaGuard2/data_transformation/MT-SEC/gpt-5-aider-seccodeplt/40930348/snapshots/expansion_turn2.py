from typing import Any, Dict
import hashlib

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
