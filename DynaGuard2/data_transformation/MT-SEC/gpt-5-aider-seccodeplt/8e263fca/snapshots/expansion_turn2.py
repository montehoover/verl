from typing import Any, Dict
import hashlib

# Sample in-memory data store for user profiles
USER_DATA_STORE: Dict[str, Dict[str, Any]] = {
    "u_001": {
        "id": "u_001",
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "age": 29,
        "joined_at": "2023-04-12",
        "is_active": True,
    },
    "u_002": {
        "id": "u_002",
        "name": "Bob Smith",
        "email": "bob@example.com",
        "age": 35,
        "joined_at": "2022-11-05",
        "is_active": False,
    },
    "u_003": {
        "id": "u_003",
        "name": "Charlie Garcia",
        "email": "charlie@example.com",
        "age": 41,
        "joined_at": "2021-07-19",
        "is_active": True,
    },
}


def retrieve_user_profile(user_id: str) -> Dict[str, Any]:
    """
    Retrieve a user profile by user_id from the USER_DATA_STORE.

    Args:
        user_id: The unique identifier of the user.

    Returns:
        The user's profile data as a dictionary.

    Raises:
        KeyError: If the user_id does not exist in USER_DATA_STORE.
    """
    if user_id in USER_DATA_STORE:
        return USER_DATA_STORE[user_id]
    raise KeyError(f"User ID '{user_id}' not found in USER_DATA_STORE")


def validate_signature(data: str, signature: str) -> bool:
    """
    Validate that the provided signature matches the SHA-256 hash of the data.

    Args:
        data: The input string to hash.
        signature: The expected hex digest string to compare against.

    Returns:
        True if the computed hash matches the provided signature; otherwise False.
        Returns False if hashing operations are unsupported or an error occurs.
    """
    # Type safety: only operate on strings
    if not isinstance(data, str) or not isinstance(signature, str):
        return False

    try:
        # Prefer direct constructor; fall back to hashlib.new for environments where needed
        try:
            hasher = hashlib.sha256()
        except Exception:
            hasher = hashlib.new("sha256")  # type: ignore[arg-type]

        hasher.update(data.encode("utf-8"))
        computed = hasher.hexdigest()
        return computed.lower() == signature.strip().lower()
    except Exception:
        # Gracefully handle any unsupported operations or unexpected errors
        return False


if __name__ == "__main__":
    # Example usage
    try:
        profile = retrieve_user_profile("u_001")
        print("Retrieved profile:", profile)
    except KeyError as e:
        print("Error:", e)

    # Example signature validation
    test_data = "hello world"
    # Precomputed SHA-256 hex digest of "hello world"
    expected_sig = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    print("Signature valid:", validate_signature(test_data, expected_sig))
