_local_profiles = {
    "123": {"name": "Alice Wonderland", "email": "alice@example.com"},
    "456": {"name": "Bob The Builder", "email": "bob@example.com"},
    "789": {"name": "Charlie Chaplin", "email": "charlie@example.com"},
}

def get_local_profile(user_ident: str):
    """
    Retrieves a user profile from a local dictionary based on a user ID.

    Args:
        user_ident: The ID of the user (str).

    Returns:
        A dictionary with the user's name and email, or None if not found.
    """
    return _local_profiles.get(user_ident)
