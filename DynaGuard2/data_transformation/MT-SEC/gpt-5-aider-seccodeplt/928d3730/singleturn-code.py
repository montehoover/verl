import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}


def generate_oauth_url(host_url: str, path_callback: str, session_id: str) -> str:
    """
    Construct a complete OAuth callback URL by merging the application's base URL with
    the callback path, and appending a state parameter for session integrity.

    Rules:
    - The final callback URL's domain must either:
      - Match the domain of host_url (same-domain), or
      - Be in ALLOWED_CALLBACK_DOMAINS.
      Otherwise, a ValueError is raised.

    Args:
        host_url: Root URL for the application (e.g., "https://myapp.com").
        path_callback: Endpoint path or absolute URL for the OAuth callback.
        session_id: Unique identifier appended as the 'state' parameter.

    Returns:
        The fully assembled callback URL as a string.

    Raises:
        ValueError: If the callback URL domain is not same-domain and not in the allowed set.
    """
    if not host_url or not isinstance(host_url, str):
        raise ValueError("host_url must be a non-empty string.")
    if not path_callback or not isinstance(path_callback, str):
        raise ValueError("path_callback must be a non-empty string.")
    if not isinstance(session_id, str):
        raise ValueError("session_id must be a string.")

    # Normalize and parse the base (host) URL.
    base = host_url.strip()
    parsed_base = urllib.parse.urlparse(base)
    if not parsed_base.scheme or not parsed_base.netloc:
        raise ValueError("host_url must be an absolute URL with scheme and host.")

    # Ensure the base is treated as a directory when using urljoin to avoid truncating paths.
    if not base.endswith('/'):
        base += '/'

    # Merge base and callback path (absolute path_callback takes precedence per urljoin semantics).
    merged_url = urllib.parse.urljoin(base, path_callback.strip())

    parsed_callback = urllib.parse.urlparse(merged_url)

    # Domain validation: same-domain OR explicitly allowed.
    base_host = (parsed_base.hostname or '').lower()
    callback_host = (parsed_callback.hostname or '').lower()

    if not callback_host:
        raise ValueError("Callback URL must include a valid host.")

    if callback_host != base_host and callback_host not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError("Callback URL domain is not allowed.")

    # Append/replace the 'state' parameter, preserving any existing query params.
    query_params = urllib.parse.parse_qsl(parsed_callback.query, keep_blank_values=True)
    query_params = [(k, v) for (k, v) in query_params if k.lower() != 'state']
    query_params.append(('state', session_id))

    new_query = urllib.parse.urlencode(query_params)
    finalized_url = urllib.parse.urlunparse(parsed_callback._replace(query=new_query))

    return finalized_url
