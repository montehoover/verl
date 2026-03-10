def create_oauth_callback_url(base_url: str, callback_path: str) -> str:
    """
    Builds a URL for an OAuth callback process.

    Args:
        base_url: The main URL string.
        callback_path: The specific path to append to the base URL.

    Returns:
        A single complete URL string.
    """
    return base_url + callback_path
