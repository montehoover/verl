from urllib.parse import urljoin

def assemble_oauth_callback_url(root_url: str, path_for_callback: str) -> str:
    """
    Assembles an OAuth callback URL from a root URL and a callback path.

    Args:
        root_url: The base URL (e.g., "https://example.com").
        path_for_callback: The path for the callback (e.g., "/oauth/callback").

    Returns:
        The combined OAuth callback URL as a string.
    """
    # Ensure root_url ends with a slash if path_for_callback is not absolute
    # urljoin handles this correctly.
    return urljoin(root_url, path_for_callback)
