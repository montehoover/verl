import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def build_oauth_redirect_url(main_url: str, redirect_path: str, nonce: str) -> str:
    """
    Constructs a URL for handling OAuth callbacks, ensuring it's for an allowed domain.

    Args:
        main_url: The base URL of the application. Must start with "https://".
        redirect_path: The path for the redirect URI.
        nonce: The nonce or state parameter to include in the URL. Must not be empty.

    Returns:
        The constructed OAuth redirect URL.

    Raises:
        ValueError: If main_url does not start with "https://", if nonce is not provided,
                    or if the domain of main_url is not in ALLOWED_CALLBACK_DOMAINS.
    """
    if not main_url.startswith("https://"):
        raise ValueError("main_url must start with 'https://'")
    if not nonce:
        raise ValueError("nonce parameter must be provided")

    parsed_main_url = urllib.parse.urlparse(main_url)
    if parsed_main_url.hostname not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(f"Domain {parsed_main_url.hostname} is not an allowed callback domain.")

    base_redirect_url = urllib.parse.urljoin(main_url, redirect_path)
    query_params = urllib.parse.urlencode({'state': nonce})
    return f"{base_redirect_url}?{query_params}"
