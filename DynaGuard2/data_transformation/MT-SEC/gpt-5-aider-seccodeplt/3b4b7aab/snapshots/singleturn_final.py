import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}


def construct_oauth_callback_uri(root_url: str, callback_uri: str, session_state: str) -> str:
    """
    Construct an OAuth callback URL by combining the application's root URL with the callback path
    and appending a state parameter. Validates that the resulting URL's domain is allowed.

    Args:
        root_url: The application's main base URL (e.g., "https://auth.myapp.com").
        callback_uri: The specific path for the OAuth callback endpoint (e.g., "/oauth/callback").
                      If an absolute URL is provided, it will be used as-is.
        session_state: A unique state string used to ensure integrity of the OAuth flow.

    Returns:
        The finalized OAuth callback URL as a string.

    Raises:
        ValueError: If the computed callback URL's domain is not in ALLOWED_CALLBACK_DOMAINS.
    """
    # Determine if callback_uri is absolute; if so, use it directly.
    parsed_callback = urllib.parse.urlparse(callback_uri)
    if parsed_callback.scheme and parsed_callback.netloc:
        combined_url = callback_uri
    else:
        # Normalize base and path then join.
        base = root_url if root_url.endswith('/') else root_url + '/'
        path = callback_uri.lstrip('/')
        combined_url = urllib.parse.urljoin(base, path)

    # Parse the combined URL to manipulate query parameters.
    parsed = urllib.parse.urlparse(combined_url)

    # Validate the hostname against the allowed domains before returning.
    hostname = (parsed.hostname or '').lower()
    if hostname not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(f"Callback domain '{hostname}' is not allowed")

    # Append or replace the 'state' query parameter.
    query_params = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    query_params['state'] = session_state
    new_query = urllib.parse.urlencode(query_params, doseq=True)

    # Build the final URL with the updated query.
    final_url = urllib.parse.urlunparse(parsed._replace(query=new_query, fragment=''))

    return final_url
