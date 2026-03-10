from typing import Any, Dict

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


if __name__ == "__main__":
    # Example usage
    try:
        profile = retrieve_user_profile("u_001")
        print("Retrieved profile:", profile)
    except KeyError as e:
        print("Error:", e)
