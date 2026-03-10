import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}


def assemble_oauth_callback_url(root_url: str, path_for_callback: str, session_token: str) -> str:
    """
    Builds an OAuth callback URL for third-party authentication by combining a base URL,
    the callback path, and a state parameter for securing the flow.

    Args:
        root_url: The main URL of the application.
        path_for_callback: The path to execute OAuth callbacks.
        session_token: The state parameter used to verify the OAuth transaction.

    Returns:
        The full OAuth callback URL as a string.

    Raises:
        ValueError: If the callback URL points to an invalid or unauthorized domain.
    """
    # Compose the base + path into a full URL (path may be absolute and override base).
    combined_url = urllib.parse.urljoin(root_url, path_for_callback)

    # Validate the combined URL structure and authorized domain.
    parsed = urllib.parse.urlparse(combined_url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname:
        raise ValueError("Invalid callback URL: missing or unsupported scheme/host")

    if parsed.hostname not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(f"Unauthorized callback domain: {parsed.hostname}")

    # Merge existing query parameters and set/override the state parameter.
    query_params = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    query_params['state'] = session_token

    new_query = urllib.parse.urlencode(query_params, doseq=True)
    final_url = urllib.parse.urlunparse(parsed._replace(query=new_query))

    return final_url
