def build_user_url(base_url: str, path: str, user_id: str) -> str:
    """
    Build a user-specific URL by concatenating the base_url and path, then
    appending the user_id as a query parameter.

    Note: This function does not modify slashes between base_url and path.
    """
    url = f"{base_url}{path}"
    separator = '&' if '?' in url else '?'
    return f"{url}{separator}user_id={user_id}"
