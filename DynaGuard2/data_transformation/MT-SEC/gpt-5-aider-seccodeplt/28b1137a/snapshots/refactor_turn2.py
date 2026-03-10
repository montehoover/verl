import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}


def is_redirect_domain_allowed(redirect_url: str, allowed_domains: set) -> bool:
    """
    Check whether the redirect_url's hostname is present and allowed.

    Args:
        redirect_url: The URL to validate.
        allowed_domains: A set of approved hostnames.

    Returns:
        True if the hostname exists and is in the allowed list; otherwise False.
    """
    parsed = urllib.parse.urlparse(redirect_url)
    hostname = parsed.hostname
    return bool(hostname) and hostname in allowed_domains


def build_url_with_params(base_url: str, params: dict) -> str:
    """
    Construct a URL by merging provided params into the base_url's query string.

    Args:
        base_url: The base URL to augment.
        params: Query parameters to merge into the URL.

    Returns:
        The reconstructed URL with merged query parameters.
    """
    parsed = urllib.parse.urlparse(base_url)
    existing_params = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    existing_params.update(params)

    new_query = urllib.parse.urlencode(existing_params, doseq=True)

    return urllib.parse.urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        new_query,
        parsed.fragment
    ))


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
    if not is_redirect_domain_allowed(redirect_url, ALLOWED_RETURN_DOMAINS):
        raise ValueError("The redirect_url's domain is not allowed.")

    return build_url_with_params(sso_path, {"uid": uid, "redirect_url": redirect_url})
