import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}


def build_sso_url(sso_path: str, uid: str, redirect_url: str) -> str:
    """
    Construct an SSO redirect URL.

    Args:
        sso_path: The SSO endpoint URL.
        uid: The unique user ID.
        redirect_url: The return URL after SSO completes.

    Returns:
        A string containing the full SSO redirect URL.

    Raises:
        ValueError: If the redirect_url's domain is not in the approved domains list.
    """
    # Validate redirect_url domain
    redirect_parsed = urllib.parse.urlparse(redirect_url)
    hostname = redirect_parsed.hostname
    if not hostname or hostname not in ALLOWED_RETURN_DOMAINS:
        raise ValueError("The redirect_url's domain is not allowed.")

    # Parse the base SSO path and merge query parameters
    sso_parsed = urllib.parse.urlparse(sso_path)
    existing_params = dict(urllib.parse.parse_qsl(sso_parsed.query, keep_blank_values=True))

    # Set/override required parameters
    existing_params["uid"] = uid
    existing_params["redirect_url"] = redirect_url

    new_query = urllib.parse.urlencode(existing_params, doseq=True)

    # Rebuild the SSO URL with updated query parameters
    full_sso_url = urllib.parse.urlunparse((
        sso_parsed.scheme,
        sso_parsed.netloc,
        sso_parsed.path,
        sso_parsed.params,
        new_query,
        sso_parsed.fragment
    ))

    return full_sso_url
