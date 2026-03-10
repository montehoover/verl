from typing import Optional, Dict

# Local in-memory user profiles database
_LOCAL_PROFILES: Dict[str, Dict[str, str]] = {
    "u001": {"name": "Alice Example", "email": "alice@example.com"},
    "u002": {"name": "Bob Sample", "email": "bob.sample@example.com"},
    "u003": {"name": "Charlie Test", "email": "charlie.test@example.com"},
}

def get_local_profile(user_ident: str) -> Optional[Dict[str, str]]:
    """
    Retrieve a user profile from a local dictionary.

    Args:
        user_ident: The user ID to look up.

    Returns:
        A dictionary with keys 'name' and 'email' if found; otherwise, None.
    """
    profile = _LOCAL_PROFILES.get(user_ident)
    if profile is None:
        return None
    # Return only the required fields
    return {"name": profile["name"], "email": profile["email"]}
