from typing import Dict, Any
import hashlib
import hmac

# Sample in-memory user data store to illustrate structure.
# In a real system, this could be fetched from a database or remote API.
USER_DATA: Dict[str, Dict[str, Any]] = {
    "u_123": {
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "roles": ["admin", "editor"],
        "created_at": "2023-06-01T12:34:56Z",
        "active": True,
        "preferences": {
            "language": "en",
            "timezone": "UTC"
        },
    },
    "u_456": {
        "name": "Bob Smith",
        "email": "bob@example.com",
        "roles": ["user"],
        "created_at": "2024-01-15T09:22:10Z",
        "active": False,
        "preferences": {
            "language": "es",
            "timezone": "America/Mexico_City"
        },
    },
    "u_789": {
        "name": "Charlie Lee",
        "email": "charlie@example.com",
        "roles": ["support", "user"],
        "created_at": "2024-07-20T17:05:00Z",
        "active": True,
        "preferences": {
            "language": "en",
            "timezone": "America/Los_Angeles"
        },
    },
}


def retrieve_user_profile(user_id: str) -> Dict[str, Any]:
    """
    Retrieve a user profile by user_id from USER_DATA.

    Args:
        user_id: The unique identifier for the user.

    Returns:
        The profile dictionary for the given user_id.

    Raises:
        KeyError: If the user_id does not exist in USER_DATA.
    """
    try:
        return USER_DATA[user_id]
    except KeyError as e:
        raise KeyError(f"user_id '{user_id}' not found") from e


def validate_signature(data: str, signature: str) -> bool:
    """
    Validate that the provided signature matches the SHA-256 hex digest of data.

    Args:
        data: The input string to hash.
        signature: The expected signature as a hex string.

    Returns:
        True if the signature matches the computed hash; False otherwise
        or if any error occurs.
    """
    try:
        if not isinstance(data, str) or not isinstance(signature, str):
            return False

        computed = hashlib.sha256(data.encode("utf-8")).hexdigest()
        # Normalize provided signature to lowercase for hex comparison
        provided = signature.lower()

        return hmac.compare_digest(computed, provided)
    except Exception:
        return False
