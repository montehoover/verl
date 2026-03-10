import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}


def build_oauth_redirect_url(main_url: str, redirect_path: str, nonce: str) -> str:
    """
    Build the full OAuth callback URL.

    Args:
        main_url: The main/base URL of the application (e.g., "https://myapp.com").
        redirect_path: The callback path or a full URL (e.g., "/oauth/callback" or "https://auth.myapp.com/cb").
        nonce: The state parameter used to verify the OAuth transaction.

    Returns:
        The full OAuth callback URL as a string.

    Raises:
        ValueError: If the computed callback URL is invalid or points to an unauthorized domain.
    """
    if not isinstance(main_url, str) or not isinstance(redirect_path, str) or not isinstance(nonce, str):
        raise ValueError("main_url, redirect_path, and nonce must all be strings")

    # Build the callback URL
    combined_url = urllib.parse.urljoin(main_url, redirect_path)

    # Parse URLs for validation
    parsed_callback = urllib.parse.urlparse(combined_url)
    parsed_base = urllib.parse.urlparse(main_url)

    # Basic validity checks
    if not parsed_callback.scheme or not parsed_callback.netloc:
        raise ValueError("Callback URL is invalid: missing scheme or host")

    if parsed_callback.scheme not in {"http", "https"}:
        raise ValueError("Callback URL is invalid: unsupported scheme")

    # Domain authorization:
    # - Allow if callback host equals the base app host, OR
    # - Allow if callback host is explicitly whitelisted.
    callback_host = (parsed_callback.hostname or "").lower()
    base_host = (parsed_base.hostname or "").lower()

    if not callback_host:
        raise ValueError("Callback URL is invalid: missing host")

    if callback_host != base_host and callback_host not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(f"ValueError: Unauthorized callback domain: {callback_host}")

    # Ensure the state parameter is present and equals the provided nonce
    query_pairs = urllib.parse.parse_qsl(parsed_callback.query, keep_blank_values=True)
    query_dict = dict(query_pairs)
    query_dict["state"] = nonce
    new_query = urllib.parse.urlencode(query_dict, doseq=True)

    # Reconstruct the final URL
    final_url = urllib.parse.urlunparse((
        parsed_callback.scheme,
        parsed_callback.netloc,
        parsed_callback.path,
        parsed_callback.params,
        new_query,
        parsed_callback.fragment,
    ))
    return final_url
