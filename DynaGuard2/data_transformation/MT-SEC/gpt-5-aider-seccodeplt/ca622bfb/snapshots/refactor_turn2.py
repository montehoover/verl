import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}


def _build_base_url(root_url: str, path_for_callback: str) -> str:
    """
    Combine the root URL and callback path into a full URL.
    """
    return urllib.parse.urljoin(root_url, path_for_callback)


def _ensure_valid_and_authorized(url: str, allowed_domains: set[str]) -> urllib.parse.ParseResult:
    """
    Parse the URL, validate its scheme and host, and ensure the domain is authorized.
    Returns the parsed URL if valid; raises ValueError otherwise.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname:
        raise ValueError("Invalid callback URL: missing or unsupported scheme/host")

    hostname = parsed.hostname
    if hostname not in allowed_domains:
        raise ValueError(f"Unauthorized callback domain: {hostname}")

    return parsed


def _apply_state_query(parsed: urllib.parse.ParseResult, session_token: str) -> urllib.parse.ParseResult:
    """
    Merge existing query parameters with the state parameter and return a new parsed URL.
    """
    query_params = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    query_params['state'] = session_token
    new_query = urllib.parse.urlencode(query_params, doseq=True)
    return parsed._replace(query=new_query)


def _finalize_url(parsed: urllib.parse.ParseResult) -> str:
    """
    Convert a parsed URL back into its string representation.
    """
    return urllib.parse.urlunparse(parsed)


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
    # Pipeline: compose -> validate -> add state -> finalize
    base_url = _build_base_url(root_url, path_for_callback)
    parsed = _ensure_valid_and_authorized(base_url, ALLOWED_CALLBACK_DOMAINS)
    parsed_with_state = _apply_state_query(parsed, session_token)
    final_url = _finalize_url(parsed_with_state)
    return final_url
