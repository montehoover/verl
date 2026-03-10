import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}


def generate_sso_redirect(sso_endpoint: str, user_id: str, return_url: str) -> str:
    """
    Generate a Single Sign-On (SSO) redirect URL.

    Combines the given SSO endpoint with user-specific parameters and a validated return URL.

    Args:
        sso_endpoint: The SSO service endpoint URL.
        user_id: The user's unique identifier.
        return_url: The URL to return to after SSO.

    Returns:
        The complete SSO redirect URL as a string.

    Raises:
        ValueError: If the return_url is not from an allowed domain or is not a valid absolute URL.
    """
    # Validate return_url is absolute and from an allowed domain
    ru = urllib.parse.urlsplit(return_url)
    if not ru.scheme or not ru.netloc:
        raise ValueError("return_url must be an absolute URL with scheme and host")

    host = ru.hostname.lower() if ru.hostname else None
    if host not in ALLOWED_RETURN_DOMAINS:
        raise ValueError("return_url is not from an allowed domain")

    # Parse the SSO endpoint and merge query parameters
    ep = urllib.parse.urlsplit(sso_endpoint)
    existing_qs = dict(urllib.parse.parse_qsl(ep.query, keep_blank_values=True))

    # Add/override with required SSO params
    existing_qs.update({
        "user_id": user_id,
        "return_url": return_url,
    })

    new_query = urllib.parse.urlencode(existing_qs, doseq=True)
    new_ep = ep._replace(query=new_query)

    return urllib.parse.urlunsplit(new_ep)
