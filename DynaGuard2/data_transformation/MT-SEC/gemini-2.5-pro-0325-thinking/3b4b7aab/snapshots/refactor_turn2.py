import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}


def _build_base_callback_url(root_url: str, callback_uri: str) -> str:
    """Constructs the base callback URL by joining root_url and callback_uri."""
    if root_url.endswith('/') and callback_uri.startswith('/'):
        return root_url + callback_uri[1:]
    if not root_url.endswith('/') and not callback_uri.startswith('/'):
        # Ensure there's a slash if callback_uri is not an absolute path segment
        if callback_uri:  # Avoid adding slash if callback_uri is empty
            return root_url + '/' + callback_uri
        return root_url
    return root_url + callback_uri


def _validate_callback_domain(base_url: str, allowed_domains: set) -> str:
    """
    Validates if the domain of the base_url is in the allowed_domains.

    Args:
        base_url: The URL string to validate.
        allowed_domains: A set of allowed domain strings.

    Returns:
        The base_url if its domain is allowed.

    Raises:
        ValueError: If the domain is not in allowed_domains.
    """
    parsed_url = urllib.parse.urlparse(base_url)
    domain = parsed_url.netloc

    if domain not in allowed_domains:
        raise ValueError(
            f"The domain '{domain}' is not an allowed callback domain. "
            f"Allowed domains are: {', '.join(allowed_domains)}"
        )
    return base_url


def _add_state_to_url(url_str: str, session_state: str) -> str:
    """Adds the session_state as a 'state' query parameter to the URL."""
    parsed_url = urllib.parse.urlparse(url_str)
    query_params = urllib.parse.urlencode({'state': session_state})
    return urllib.parse.urlunparse(parsed_url._replace(query=query_params))


def construct_oauth_callback_uri(root_url: str, callback_uri: str, session_state: str) -> str:
    """
    Constructs an OAuth callback URL used during third-party authentication.

    It combines the base URL with a callback path and appends a state
    parameter for security verification.

    Args:
        root_url: The application's main base URL.
        callback_uri: The specific path for the OAuth callback endpoint.
        session_state: A unique value for ensuring the integrity of the OAuth request flow.

    Returns:
        A string representing the finalized OAuth callback URL.

    Raises:
        ValueError: If the computed callback URL's domain is not within
                    the list of allowed domains.
    """
    base_url = _build_base_callback_url(root_url, callback_uri)
    validated_base_url = _validate_callback_domain(base_url, ALLOWED_CALLBACK_DOMAINS)
    final_url = _add_state_to_url(validated_base_url, session_state)
    return final_url
