from urllib import parse as urlparse

APPROVED_DOMAINS = ['trusted-site.com', 'partner-domain.org']


def generate_redirect_link(app_url: str, redirect_param: str) -> str:
    """
    Generates a safe redirect URL after login.

    Arguments:
    - app_url: Base URL of the application (e.g., "https://myapp.com").
    - redirect_param: A path (e.g., "/dashboard") or a full URL (e.g., "https://trusted-site.com/path").

    Returns:
    - The final redirect URL as a string.

    Raises:
    - ValueError: if the final URL does not remain within the application's domain
      or an allowed domain, or if an invalid URL is provided.

    Example:
    >>> generate_redirect_link("https://myapp.com", "/dashboard")
    'https://myapp.com/dashboard'
    """
    # Basic validation and normalization of inputs
    if not isinstance(app_url, str) or not app_url.strip():
        raise ValueError("app_url must be a non-empty string")
    if not isinstance(redirect_param, str) or not redirect_param.strip():
        # Default to root of the app if redirect_param is empty/blank
        redirect_param = "/"

    app_url = app_url.strip()
    redirect_param = redirect_param.strip()

    # Parse and validate the base app URL
    try:
        base = urlparse.urlparse(app_url)
    except Exception as e:
        raise ValueError("Invalid app_url") from e

    if base.scheme not in ("http", "https") or not base.hostname:
        raise ValueError("app_url must be an absolute http(s) URL")

    base_host = base.hostname.lower().rstrip(".")

    # Build the allowed domains set (include app domain + approved domains)
    allowed_domains = {base_host}
    for d in APPROVED_DOMAINS:
        if isinstance(d, str) and d.strip():
            allowed_domains.add(d.strip().lower().rstrip("."))

    def is_allowed(hostname: str) -> bool:
        if not hostname:
            return False
        host = hostname.lower().rstrip(".")
        # Allow exact domain or subdomains of any allowed domain
        for dom in allowed_domains:
            if host == dom or host.endswith("." + dom):
                return True
        return False

    # Resolve the final URL from redirect_param against app_url
    parsed_redirect = urlparse.urlparse(redirect_param)

    if parsed_redirect.scheme:
        # Absolute URL with scheme provided
        final_url = redirect_param
    else:
        if redirect_param.startswith("//"):
            # Scheme-relative URL (e.g., //trusted-site.com/path)
            final_url = f"{base.scheme}:{redirect_param}"
        else:
            # Relative or absolute path; join to base
            base_for_join = app_url if app_url.endswith("/") else app_url + "/"
            final_url = urlparse.urljoin(base_for_join, redirect_param)

    # Ensure final URL is absolute and validate it
    parsed_final = urlparse.urlparse(final_url)
    if not parsed_final.scheme or not parsed_final.netloc:
        # If still not absolute, join with base again
        base_for_join = app_url if app_url.endswith("/") else app_url + "/"
        final_url = urlparse.urljoin(base_for_join, final_url)
        parsed_final = urlparse.urlparse(final_url)

    if parsed_final.scheme not in ("http", "https"):
        raise ValueError("Unsafe redirect target: unsupported URL scheme")

    if not is_allowed(parsed_final.hostname):
        raise ValueError("Unsafe redirect target: domain not allowed")

    return final_url
