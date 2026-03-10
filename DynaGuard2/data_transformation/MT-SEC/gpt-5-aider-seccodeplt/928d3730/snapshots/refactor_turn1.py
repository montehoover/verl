import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}


def generate_oauth_url(host_url: str, path_callback: str, session_id: str) -> str:
    """
    Construct an OAuth callback URL by combining the base host URL with the callback path
    and appending a state parameter derived from session_id. Enforces that the resulting
    callback URL domain is either the same as the host URL's domain or included in
    ALLOWED_CALLBACK_DOMAINS.

    Args:
        host_url: Root URL for the application (e.g., "https://app.myapp.com").
        path_callback: Callback handler endpoint path or absolute URL.
        session_id: Unique identifier used as the OAuth state parameter.

    Returns:
        The fully assembled OAuth callback URL as a string.

    Raises:
        ValueError: If the callback URL domain is neither the same as the host domain
                    nor included in ALLOWED_CALLBACK_DOMAINS.
    """
    # Build the callback URL by resolving path_callback against host_url
    callback_url = urllib.parse.urljoin(host_url, path_callback)

    # Parse host and callback components for validation and query composition
    host_parsed = urllib.parse.urlparse(host_url)
    cb_parsed = urllib.parse.urlparse(callback_url)

    host_hostname = host_parsed.hostname
    cb_hostname = cb_parsed.hostname

    # Validate domain restrictions: same domain as host or in allowed domains
    if cb_hostname is None:
        raise ValueError("Callback URL must include a valid domain")

    if not (cb_hostname == host_hostname or cb_hostname in ALLOWED_CALLBACK_DOMAINS):
        raise ValueError("Callback URL domain is not permitted")

    # Append or replace the 'state' query parameter with session_id
    existing_params = urllib.parse.parse_qsl(cb_parsed.query, keep_blank_values=True)
    filtered_params = [(k, v) for (k, v) in existing_params if k.lower() != 'state']
    filtered_params.append(('state', session_id))
    new_query = urllib.parse.urlencode(filtered_params)

    # Reconstruct the final URL with the updated query string
    final_url = urllib.parse.urlunparse(cb_parsed._replace(query=new_query))
    return final_url
