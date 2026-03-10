_local_profiles = {
    "123": {"name": "Alice Wonderland", "email": "alice@example.com"},
    "456": {"name": "Bob The Builder", "email": "bob@example.com"},
    "789": {"name": "Charlie Chaplin", "email": "charlie@example.com"},
}

def get_local_profile(user_ident: str, trusted_ids: list[str]):
    """
    Retrieves a user profile from a local dictionary based on a user ID,
    if the user ID is in the list of trusted IDs.

    Args:
        user_ident: The ID of the user (str).
        trusted_ids: A list of trusted user IDs (list of str).

    Returns:
        A dictionary with the user's name and email, or None if not found
        or if the user_ident is not in trusted_ids.
    """
    if user_ident not in trusted_ids:
        return None
    return _local_profiles.get(user_ident)
