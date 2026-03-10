import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}


def generate_sso_redirect(sso_endpoint: str, user_id: str, return_url: str) -> str:
    """
    Generate a Single Sign-On (SSO) redirect URL.

    Combines the SSO endpoint with user-specific parameters and a validated return URL.

    Args:
        sso_endpoint: The SSO service endpoint URL.
        user_id: The user's unique identifier.
        return_url: The URL to return to after SSO.

    Returns:
        A complete SSO redirect URL as a string.

    Raises:
        ValueError: If the return_url is not absolute or not from an allowed domain.
    """
    # Validate return_url is absolute and from an allowed domain
    parsed_return = urllib.parse.urlparse(return_url)
    if not parsed_return.scheme or not parsed_return.netloc:
        raise ValueError("return_url must be an absolute URL with a network location")

    hostname = parsed_return.hostname
    if hostname not in ALLOWED_RETURN_DOMAINS:
        raise ValueError("return_url is not from an allowed domain")

    # Parse the SSO endpoint and merge query parameters
    parsed_endpoint = urllib.parse.urlparse(sso_endpoint)

    # Preserve existing query params, but replace 'user' and 'return' if present
    existing_params = urllib.parse.parse_qsl(parsed_endpoint.query, keep_blank_values=True)
    filtered_params = [(k, v) for (k, v) in existing_params if k not in ("user", "return")]

    # Append required SSO parameters
    filtered_params.append(("user", user_id))
    filtered_params.append(("return", return_url))

    new_query = urllib.parse.urlencode(filtered_params, doseq=True)

    # Rebuild the final URL
    redirect_url = urllib.parse.urlunparse(
        (
            parsed_endpoint.scheme,
            parsed_endpoint.netloc,
            parsed_endpoint.path,
            parsed_endpoint.params,
            new_query,
            parsed_endpoint.fragment,
        )
    )
    return redirect_url
