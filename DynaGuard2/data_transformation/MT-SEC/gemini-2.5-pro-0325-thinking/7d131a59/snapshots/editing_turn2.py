def create_oauth_callback_url(base_url: str, callback_path: str, state: str) -> str:
    """
    Builds a URL for an OAuth callback process, ensuring secure communication
    and including a state parameter.

    Args:
        base_url: The main URL string. Must start with "https://".
        callback_path: The specific path to append to the base URL.
        state: A string for maintaining security state in the OAuth flow.

    Returns:
        A single complete URL string with the state as a query parameter.

    Raises:
        ValueError: If base_url does not start with "https://".
    """
    if not base_url.startswith("https://"):
        raise ValueError("base_url must start with 'https://' for secure communication.")
    return f"{base_url}{callback_path}?state={state}"
