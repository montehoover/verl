from typing import Dict, Any
import hashlib

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
