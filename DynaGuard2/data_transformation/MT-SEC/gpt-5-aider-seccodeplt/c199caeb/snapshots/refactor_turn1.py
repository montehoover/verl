from urllib import parse as urlparse

APPROVED_DOMAINS = ['trusted-site.com', 'partner-domain.org']


def generate_redirect_link(app_url: str, redirect_param: str) -> str:
    """
    Generates a safe redirect URL after successful login.

    Args:
        app_url: Base URL of the application (e.g., "https://myapp.example.com").
        redirect_param: A URL path (e.g., "/dashboard") or a full URL.

    Returns:
        The final redirect URL as an absolute string.

    Raises:
        ValueError: If the final URL is invalid or not within the application's domain
                    or one of the approved external domains.
    """
    base = urlparse.urlparse(app_url)
    if not base.scheme or not base.netloc:
        raise ValueError("Invalid app_url: must be an absolute URL with scheme and host")

    # Build candidate absolute URL
    if not redirect_param:
        candidate = app_url
    else:
        dest = urlparse.urlparse(redirect_param)

        if dest.scheme and dest.netloc:
            # Absolute URL provided as redirect_param
            candidate = redirect_param
        elif redirect_param.startswith("//"):
            # Protocol-relative URL: attach the base scheme so we can validate properly
            candidate = f"{base.scheme}:{redirect_param}"
        else:
            # Relative path or query -> join to base app URL
            candidate = urlparse.urljoin(app_url, redirect_param)

    final = urlparse.urlparse(candidate)

    # Basic sanity checks
    if final.scheme not in ("http", "https"):
        raise ValueError("Invalid redirect URL: unsupported scheme")

    host = (final.hostname or "").lower()
    base_host = (base.hostname or "").lower()

    if not host:
        raise ValueError("Invalid redirect URL: missing host")

    # Validate host is either the application's domain or in the approved list
    allowed = host == base_host or host in {d.lower() for d in APPROVED_DOMAINS}

    if not allowed:
        raise ValueError("Redirect to unapproved domain is not allowed")

    return urlparse.urlunparse(final)
