import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}


def assemble_oauth_callback(application_url: str, callback_route: str, token_state: str) -> str:
    """
    Assemble a secure OAuth callback URL.

    Args:
        application_url: Root URL for the application (e.g., 'https://myapp.com').
        callback_route: Endpoint for the OAuth callback handler (path or absolute URL).
        token_state: Unique state token to maintain OAuth exchange integrity.

    Returns:
        A string containing the fully assembled OAuth callback URL.

    Raises:
        ValueError: If the callback URL's domain is neither the same as the application's
                    domain nor included in ALLOWED_CALLBACK_DOMAINS.
    """
    # Build the base callback URL from application_url and callback_route.
    # If callback_route is absolute, urljoin will respect it; otherwise it will be joined to application_url.
    callback_url = urllib.parse.urljoin(application_url, callback_route)

    # Parse components for security checks and query manipulation.
    app_parts = urllib.parse.urlparse(application_url)
    cb_parts = urllib.parse.urlparse(callback_url)

    app_host = app_parts.hostname
    cb_host = cb_parts.hostname

    # Enforce same-domain or allowed-domain restriction.
    # Compare hostnames (ignore ports) for the "same-domain" check.
    if cb_host is None or app_host is None:
        # If we cannot determine hostnames, consider it a security risk.
        raise ValueError("Invalid URL composition: unable to determine callback or application host.")

    if cb_host != app_host and cb_host not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(
            f"Callback domain '{cb_host}' is not allowed. Must match application domain "
            f"('{app_host}') or be in ALLOWED_CALLBACK_DOMAINS."
        )

    # Merge query parameters, ensuring 'state' is set/overridden.
    query_params = list(urllib.parse.parse_qsl(cb_parts.query, keep_blank_values=True))
    # Remove existing 'state' if present, then add the provided token_state.
    query_params = [(k, v) for (k, v) in query_params if k.lower() != 'state']
    query_params.append(('state', token_state))

    new_query = urllib.parse.urlencode(query_params, doseq=True)

    # Reconstruct the final URL preserving all parts except the updated query.
    final_parts = cb_parts._replace(query=new_query)
    return final_parts.geturl()
