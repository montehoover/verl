from urllib.parse import urljoin, urlencode

def build_oauth_redirect_url(main_url: str, redirect_path: str, nonce: str) -> str:
    """
    Constructs a URL for handling OAuth callbacks.

    Args:
        main_url: The base URL of the application.
        redirect_path: The path for the redirect URI.
        nonce: The nonce or state parameter to include in the URL.

    Returns:
        The constructed OAuth redirect URL.
    """
    base_redirect_url = urljoin(main_url, redirect_path)
    query_params = urlencode({'state': nonce})
    return f"{base_redirect_url}?{query_params}"
