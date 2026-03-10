from urllib import parse as urlparse

APPROVED_DOMAINS = ['trusted-site.com', 'partner-domain.org']


def _validate_base_url(app_url: str) -> urlparse.ParseResult:
    """
    Parse and validate the base application URL.
    Raises ValueError if invalid.
    """
    base = urlparse.urlparse(app_url)
    if not base.scheme or not base.netloc:
        raise ValueError("Invalid app_url: must be an absolute URL with scheme and host")
    return base


def _construct_candidate_url(base: urlparse.ParseResult, app_url: str, redirect_param: str) -> str:
    """
    Construct the candidate absolute URL to redirect to, based on the base app URL
    and the user-provided redirect parameter. This function is pure and does not
    perform domain validation.
    """
    if not redirect_param:
        return app_url

    dest = urlparse.urlparse(redirect_param)

    if dest.scheme and dest.netloc:
        # Absolute URL provided as redirect_param
        return redirect_param
    elif redirect_param.startswith("//"):
        # Protocol-relative URL: attach the base scheme so we can validate properly
        return f"{base.scheme}:{redirect_param}"
    else:
        # Relative path or query -> join to base app URL
        return urlparse.urljoin(app_url, redirect_param)


def _is_allowed_host(host: str, base_host: str) -> bool:
    """
    Determine if the host is allowed: either matches the application's host or
    is present in the approved domains list (case-insensitive).
    """
    if not host:
        return False
    host = host.lower()
    base_host = (base_host or "").lower()
    approved = {d.lower() for d in APPROVED_DOMAINS}
    return host == base_host or host in approved


def _validate_final_destination(base: urlparse.ParseResult, candidate_url: str) -> urlparse.ParseResult:
    """
    Validate the final redirect destination:
    - Scheme must be http or https
    - Host must exist
    - Host must match application host or be in approved domains
    Raises ValueError if invalid.
    """
    final = urlparse.urlparse(candidate_url)

    if final.scheme not in ("http", "https"):
        raise ValueError("Invalid redirect URL: unsupported scheme")

    host = (final.hostname or "").lower()
    if not host:
        raise ValueError("Invalid redirect URL: missing host")

    base_host = (base.hostname or "").lower()
    if not _is_allowed_host(host, base_host):
        raise ValueError("Redirect to unapproved domain is not allowed")

    return final


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
    base = _validate_base_url(app_url)
    candidate = _construct_candidate_url(base, app_url, redirect_param)
    final = _validate_final_destination(base, candidate)
    return urlparse.urlunparse(final)
