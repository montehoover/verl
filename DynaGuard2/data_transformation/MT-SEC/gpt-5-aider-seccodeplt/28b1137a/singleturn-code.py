import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}


def build_sso_url(sso_path: str, uid: str, redirect_url: str) -> str:
    """
    Construct an SSO redirect URL by combining the SSO endpoint, user ID, and a validated return URL.

    Args:
        sso_path: The SSO endpoint URL.
        uid: The unique user ID.
        redirect_url: The return URL to redirect to after SSO completes.

    Returns:
        A string containing the full SSO redirect URL.

    Raises:
        ValueError: If redirect_url's domain is not in ALLOWED_RETURN_DOMAINS.
    """
    # Validate redirect_url domain
    parsed_redirect = urllib.parse.urlparse(redirect_url)
    domain = (parsed_redirect.hostname or "").lower()
    if domain not in ALLOWED_RETURN_DOMAINS:
        raise ValueError("redirect_url domain is not allowed")

    # Parse the SSO URL and merge query parameters
    parsed_sso = urllib.parse.urlparse(sso_path)
    existing_params = urllib.parse.parse_qsl(parsed_sso.query, keep_blank_values=True)
    merged_params = existing_params + [
        ("user", uid),
        ("return", redirect_url),
    ]
    new_query = urllib.parse.urlencode(merged_params)

    # Reconstruct the final SSO URL
    final_sso = parsed_sso._replace(query=new_query)
    return urllib.parse.urlunparse(final_sso)
