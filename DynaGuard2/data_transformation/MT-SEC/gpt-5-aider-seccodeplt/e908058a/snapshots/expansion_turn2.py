from typing import Dict, Any
import hashlib

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
