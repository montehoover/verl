import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}


def _construct_callback_url(host_url: str, path_callback: str) -> str:
    """
    Pure function: Construct the callback URL from a host URL and a callback path/URL.
    """
    return urllib.parse.urljoin(host_url, path_callback)


def _validate_callback_domain(host_url: str, callback_url: str) -> str:
    """
    Pure function: Validate that the callback URL's domain is either the same as the host
    domain or present in ALLOWED_CALLBACK_DOMAINS. Returns the unmodified callback_url
    on success for easy pipeline chaining.
    """
    host_hostname = urllib.parse.urlparse(host_url).hostname
    cb_hostname = urllib.parse.urlparse(callback_url).hostname

    if cb_hostname is None:
        raise ValueError("Callback URL must include a valid domain")

    if not (cb_hostname == host_hostname or cb_hostname in ALLOWED_CALLBACK_DOMAINS):
        raise ValueError("Callback URL domain is not permitted")

    return callback_url


def _upsert_state_param(callback_url: str, session_id: str) -> str:
    """
    Pure function: Ensure the 'state' query parameter is set to session_id.
    If present, it is replaced; otherwise, it is added.
    """
    cb_parsed = urllib.parse.urlparse(callback_url)
    existing_params = urllib.parse.parse_qsl(cb_parsed.query, keep_blank_values=True)
    filtered_params = [(k, v) for (k, v) in existing_params if k.lower() != 'state']
    filtered_params.append(('state', session_id))
    new_query = urllib.parse.urlencode(filtered_params)
    return urllib.parse.urlunparse(cb_parsed._replace(query=new_query))


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
    # Pipeline: construct -> validate -> add state
    callback_url = _construct_callback_url(host_url, path_callback)
    validated_url = _validate_callback_domain(host_url, callback_url)
    final_url = _upsert_state_param(validated_url, session_id)
    return final_url
