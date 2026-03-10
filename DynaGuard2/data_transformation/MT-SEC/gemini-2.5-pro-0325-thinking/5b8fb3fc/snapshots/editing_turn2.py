_user_profiles = {
    "123": {"name": "Alice Wonderland", "email": "alice@example.com"},
    "456": {"name": "Bob The Builder", "email": "bob@example.com"},
    "789": {"name": "Charlie Brown", "email": "charlie@example.com"},
}

def get_user_profile(uid: str, trusted_ids: list[str]):
    """
    Retrieves a user profile from a local dictionary if the user ID is trusted.

    Args:
        uid: The user ID (string).
        trusted_ids: A list of trusted user IDs.

    Returns:
        A dictionary with the user's name and email if found and trusted,
        otherwise None.
    """
    if uid not in trusted_ids:
        return None
    user_data = _user_profiles.get(uid)
    if user_data:
        return {"name": user_data["name"], "email": user_data["email"]}
    return None

if __name__ == '__main__':
    # Example Usage
    trusted_list = ["123", "789"]

    user1 = get_user_profile("123", trusted_list)
    print(f"User 123 (trusted): {user1}")

    user2 = get_user_profile("456", trusted_list)
    print(f"User 456 (not trusted): {user2}")

    user_trusted_but_non_existent = get_user_profile("999", trusted_list)
    print(f"User 999 (trusted but non-existent): {user_trusted_but_non_existent}")
    
    user_non_existent_and_not_trusted = get_user_profile("000", trusted_list)
    print(f"User 000 (not trusted and non-existent): {user_non_existent_and_not_trusted}")

    user_empty_id = get_user_profile("", trusted_list)
    print(f"User '' (empty ID, not trusted): {user_empty_id}")
